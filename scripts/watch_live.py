"""Live stream trade-state watcher: continuous frames -> YOLO classifier -> hysteresis -> events.

Replay an ingested session's frames to validate the pipeline end-to-end, or point
it at a live stream URL (yt-dlp pipes the video; ffmpeg decodes raw frames).
    Emits position-open / position-closed events to the console and optionally to a
Discord relay-channel webhook. The classifier answers only open vs not_open;
trade-detail extraction (UI detector + OCR) is a later phase.

Usage:
    # Replay validation against an ingested session (no sleep, fast):
    python scripts/watch_live.py --frames-dir data/raw/<session>/frames `
        --model runs/classify/runs/vision/<run>/weights/best.pt `
        --roi 0.0,0.08,0.78,0.82 --source-id channel-b-replay-test

    # Watch a live stream (classifies every frame at --fps):
    python scripts/watch_live.py --url "https://www.youtube.com/watch?v=LIVEID" `
        --model runs/classify/runs/vision/<run>/weights/best.pt `
        --roi 0.0,0.08,0.78,0.82 --source-id channel-b-live `
        --discord-webhook "$env:SSV_DISCORD_WEBHOOK"

Discord webhook URLs are supplied at runtime via argument or environment variable
SSV_DISCORD_WEBHOOK and are never stored in the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
for path in (ROOT / "src", SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def parse_timestamp(value: str) -> float:
    parts = [float(p) for p in value.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def format_timestamp(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_roi(value: str | None) -> tuple[float, float, float, float] | None:
    if not value:
        return None
    parts = [float(v) for v in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height")
    return parts


def crop_roi(image: Any, roi: tuple[float, float, float, float] | None) -> Any:
    if roi is None:
        return image
    height, width = image.shape[:2]
    x, y, box_w, box_h = roi
    return image[int(y * height):int((y + box_h) * height),
                 int(x * width):int((x + box_w) * width)]


def build_anchor_tagger(
    template_path: Path,
    region: tuple[float, float, float, float],
    threshold: float,
    label: str,
):
    """Per-streamer platform anchor: tag frames whose anchor region matches the template.

    Example: channel-b executes only on TopstepX (large icon top-left); other streamers
    execute in TradingView. The anchor is optional per-streamer profile config and
    never changes the pooled classifier - it only tags frames/events with the
    visible execution platform. Templates are resolution-bound; build them at the
    same resolution the live source decodes (default 1280x720).
    """
    import cv2

    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise SystemExit(f"could not read anchor template: {template_path}")
    x, y, box_w, box_h = region

    def tag(full_image: Any) -> str | None:
        height, width = full_image.shape[:2]
        crop = full_image[int(y * height):int((y + box_h) * height),
                          int(x * width):int((x + box_w) * width)]
        if crop.size == 0:
            return None
        if crop.ndim == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if crop.shape[0] < template.shape[0] or crop.shape[1] < template.shape[1]:
            return None
        score = float(cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED).max())
        return label if score >= threshold else None

    return tag


class StateHysteresis:
    """Emit state transitions only after N consecutive confirming frames.

    A single low-confidence frame inside an open window (obscured position
    badge) must not close the position; a single high-confidence frame while
    flat must not open one. Long data gaps reset the confirmation runs.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        open_confirm: int = 2,
        close_confirm: int = 5,
        gap_seconds: float = 30.0,
    ) -> None:
        self.threshold = threshold
        self.open_confirm = open_confirm
        self.close_confirm = close_confirm
        self.gap_seconds = gap_seconds
        self.state = "flat"
        self._open_run = 0
        self._close_run = 0
        self._last_t: float | None = None

    def update(self, open_prob: float, t: float) -> dict[str, Any] | None:
        if self._last_t is not None and t - self._last_t > self.gap_seconds:
            self._open_run = 0
            self._close_run = 0
        self._last_t = t
        if open_prob >= self.threshold:
            self._open_run += 1
            self._close_run = 0
        else:
            self._close_run += 1
            self._open_run = 0

        event: dict[str, Any] | None = None
        if self.state == "flat" and self._open_run >= self.open_confirm:
            self.state = "open"
            event = {"event": "position_open", "at": t, "open_prob": round(open_prob, 4)}
        elif self.state == "open" and self._close_run >= self.close_confirm:
            self.state = "flat"
            event = {"event": "position_closed", "at": t, "open_prob": round(open_prob, 4)}

        if event is not None:
            self._open_run = 0
            self._close_run = 0
        return event


def replay_frames(
    frames_dir: Path,
    roi: tuple[float, float, float, float] | None,
    speed: float,
    anchor=None,
) -> Iterator[tuple[Any, float, str, str | None]]:
    import cv2

    manifest_path = frames_dir / "frames.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda r: parse_timestamp(r["video_timestamp"]))
    prev_t: float | None = None
    for row in rows:
        t = parse_timestamp(row["video_timestamp"])
        if speed > 0 and prev_t is not None:
            time.sleep(max(0.0, (t - prev_t) / speed))
        prev_t = t
        image = cv2.imread(str(frames_dir / Path(row["image_path"]).name))
        if image is None:
            continue
        yield image, t, row["video_timestamp"], (anchor(image) if anchor else None)


def resolve_live_stream_url(url: str) -> str:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "-g",
            "-f",
            "311/298/136/95/94/bv*[height<=720]/bv*/best",
            "--no-playlist",
            url,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in result.stdout.splitlines():
        if line.strip().startswith("http"):
            return line.strip()
    return url


def resolve_webhook_url(
    explicit_webhook: str | None,
    routes_config: Path | None,
    source_id: str,
) -> str | None:
    if explicit_webhook:
        return explicit_webhook
    if routes_config and routes_config.is_file():
        data = json.loads(routes_config.read_text(encoding="utf-8"))
        routes = data.get("routes", data) if isinstance(data, dict) else {}
        if source_id in routes:
            return routes[source_id]
        prefix = source_id.split("-")[0].split(":")[0]
        if prefix in routes:
            return routes[prefix]
    return os.environ.get("SSV_DISCORD_WEBHOOK")


def live_frames(
    url: str,
    fps: float,
    width: int,
    height: int,
    roi: tuple[float, float, float, float] | None,
    reconnect_seconds: float,
    anchor=None,
) -> Iterator[tuple[Any, float, str, str | None]]:
    import numpy as np

    frame_bytes = width * height * 3
    while True:
        # Pipe the stream straight from yt-dlp: a bare DASH manifest URL is not
        # always readable by ffmpeg (headers/expiry), but yt-dlp handles that.
        downloader = subprocess.Popen(
            [
                sys.executable, "-m", "yt_dlp",
                "-f", "298/136/bv*[height<=720]/bv*/best",
                "--no-playlist",
                "-o", "-",
                url,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        ffmpeg = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", "pipe:0",
                "-vf", f"fps={fps},scale={width}:{height}",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
            ],
            stdin=downloader.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if downloader.stdout is not None:
            downloader.stdout.close()
        assert ffmpeg.stdout is not None
        try:
            while True:
                buf = ffmpeg.stdout.read(frame_bytes)
                if not buf or len(buf) < frame_bytes:
                    break
                frame = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3))
                now = time.time()
                yield frame, now, datetime.now(timezone.utc).isoformat(timespec="seconds"), (anchor(frame) if anchor else None)
        finally:
            ffmpeg.kill()
            downloader.kill()
        print(f"[watch_live] stream ended; reconnecting in {reconnect_seconds}s", flush=True)
        time.sleep(reconnect_seconds)


def post_discord(webhook_url: str, content: str, timeout_seconds: float = 5.0) -> None:
    if not webhook_url.startswith(("https://discord.com/api/webhooks/", "https://discordapp.com/api/webhooks/")):
        raise ValueError("discord webhook must be an HTTPS Discord webhook URL")
    payload = json.dumps({"content": content}).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "StreamSignalVision/0.2"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(f"Discord webhook returned HTTP {response.status}")


def format_event(event: dict[str, Any], stamp: str, source_id: str) -> str:
    if event["event"] == "position_open":
        return (
            f"[{source_id}] POSITION OPEN at {stamp} "
            f"(open_prob {event['open_prob']:.2f}; trade details pending detector+OCR phase)"
        )
    return f"[{source_id}] POSITION CLOSED at {stamp} (open_prob {event['open_prob']:.2f})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", default=None, help="Live stream URL (yt-dlp compatible)")
    source.add_argument("--frames-dir", type=Path, default=None, help="Replay an ingested session frames directory")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--roi", type=parse_roi, default=None)
    parser.add_argument("--source-id", default="live-stream")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fps", type=float, default=2.0, help="Live decode/classify cadence")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--anchor-template", type=Path, default=None,
                        help="Optional per-streamer platform anchor template PNG (see scripts/platform_anchor.py)")
    parser.add_argument("--anchor-region", type=parse_roi, default=None,
                        help="Anchor search region, normalized x,y,width,height (default 0.0,0.0,0.22,0.16)")
    parser.add_argument("--anchor-threshold", type=float, default=0.72)
    parser.add_argument("--anchor-label", default="topstepx",
                        help="Tag applied to frames/events whose anchor region matches the template")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--open-confirm", type=int, default=2)
    parser.add_argument("--close-confirm", type=int, default=5)
    parser.add_argument("--gap-seconds", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=0.0, help="Replay speed multiplier (0 = no sleep)")
    parser.add_argument("--reconnect-seconds", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-latest", type=Path, default=None, help="Save latest decoded frame to this path")
    parser.add_argument("--read-details", action="store_true",
                        help="On position-open events, read trade details from the frame via the local VLM "
                             "(red/green box: direction, entry, SL, TP, instrument, qty). Adds ~3-8s per event.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--events-out", type=Path, default=None)
    parser.add_argument("--discord-webhook", default=None, help="Discord relay-channel webhook URL")
    parser.add_argument(
        "--routes-config",
        type=Path,
        default=None,
        help="Optional JSON mapping streamer/source_id -> Discord webhook URL",
    )
    args = parser.parse_args()

    webhook = resolve_webhook_url(args.discord_webhook, args.routes_config, args.source_id)

    anchor = None
    if args.anchor_template is not None:
        anchor = build_anchor_tagger(
            args.anchor_template,
            args.anchor_region or (0.0, 0.0, 0.22, 0.16),
            args.anchor_threshold,
            args.anchor_label,
        )

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    open_id = next((idx for idx, name in model.names.items() if name == "open"), 0)
    gate = StateHysteresis(args.threshold, args.open_confirm, args.close_confirm, args.gap_seconds)

    if args.frames_dir is not None:
        if not (args.frames_dir / "frames.jsonl").is_file():
            raise SystemExit(f"frames manifest not found: {args.frames_dir / 'frames.jsonl'}")
        source_iter = replay_frames(args.frames_dir, args.roi, args.speed, anchor)
    else:
        source_iter = live_frames(args.url, args.fps, args.width, args.height, args.roi, args.reconnect_seconds, anchor)

    n_frames = 0
    n_open_frames = 0
    n_anchor_frames = 0
    events: list[dict[str, Any]] = []
    started = time.time()
    for image_full, t, stamp, anchor_tag in source_iter:
        if args.save_latest is not None:
            import cv2

            args.save_latest.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save_latest), image_full)
        image = crop_roi(image_full, args.roi)
        try:
            results = model.predict([image], imgsz=args.imgsz, device=args.device, verbose=False)
        except NotImplementedError:
            args.device = "cpu"
            results = model.predict([image], imgsz=args.imgsz, device="cpu", verbose=False)
        open_prob = float(results[0].probs.data[open_id])
        n_frames += 1
        if open_prob >= args.threshold:
            n_open_frames += 1
        if anchor_tag:
            n_anchor_frames += 1
        if args.verbose:
            print(f"{stamp} open_prob={open_prob:.3f} state={gate.state}" + (f" anchor={anchor_tag}" if anchor_tag else ""))
        event = gate.update(open_prob, t)
        if event is not None:
            message = format_event(event, stamp, args.source_id)
            if anchor_tag:
                message += f" [{anchor_tag}]"
            details = None
            if args.read_details and event["event"] == "position_open":
                try:
                    from trade_details import format_details, read_trade_details

                    details = read_trade_details(image_full)
                    message += "\n" + format_details(details)
                except Exception as exc:
                    message += f"\ndetails unavailable ({type(exc).__name__})"
            print(message, flush=True)
            events.append({**event, "timestamp": stamp, "anchor": anchor_tag, "details": details})
            if webhook:
                post_discord(webhook, message)
        if args.max_frames is not None and n_frames >= args.max_frames:
            break

    elapsed = time.time() - started
    print(
        f"[watch_live] done: frames={n_frames} open_frames={n_open_frames} "
        f"anchor_frames={n_anchor_frames} events={len(events)} elapsed={elapsed:.1f}s "
        f"({(n_frames / elapsed):.1f} frames/s sustained)"
    )
    if args.events_out:
        args.events_out.parent.mkdir(parents=True, exist_ok=True)
        args.events_out.write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n", encoding="utf-8"
        )
        print(f"[watch_live] wrote {len(events)} event(s) to {args.events_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
