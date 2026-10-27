"""Mirror a streamer's POSITION number to Discord within about a second.

This ignores stop-loss and take-profit entirely. It watches one small panel that
shows the open position (positive = long, negative = short, 0 = flat), reads the
number with a tiny digit CNN, and posts the trade needed to match it.

Usage:
    python scripts/watch_position.py --url "<live>" --instrument MNQ \
        --source-id blastoise-live --routes-config <private>/webhook_routes.json
    python scripts/watch_position.py --url "<live>" --dry-run --fps 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "runs" / "digits_cnn.pt"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_digits import DigitNet  # noqa: E402
from watch_live import live_frames  # noqa: E402


def load_model() -> DigitNet:
    if not MODEL_PATH.is_file():
        raise SystemExit("digit model missing; run scripts/train_digits.py first")
    model = DigitNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def read_position(frame, roi, model) -> int | None:
    """Read the signed position integer from the panel region."""
    h, w = frame.shape[:2]
    x0 = int(roi[0] * w)
    y0 = int(roi[1] * h)
    x1 = int((roi[0] + roi[2]) * w)
    y1 = int((roi[1] + roi[3]) * h)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height_px, width_px = crop.shape[:2]
    blobs = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bh < max(12, int(height_px * 0.35)) or bw < 4:
            continue
        # A digit never fills the crop or touches its edges; that pattern means
        # the panel is covered by an overlay and the read is meaningless.
        if x <= 1 or y <= 1 or x + bw >= width_px - 1 or y + bh >= height_px - 1:
            continue
        if bh / max(1, bw) < 0.7:
            continue
        blobs.append((x, y, bw, bh))
    if not blobs:
        return None
    blobs.sort(key=lambda b: b[0])

    max_h = max(b[3] for b in blobs)
    negative = False
    digits: list[int] = []
    for x, y, bw, bh in blobs:
        if bh < max_h * 0.5:
            continue
        if bw > bh * 1.6:
            negative = True
            continue
        glyph = mask[y:y + bh, x:x + bw]
        side = max(bw, bh)
        square = np.zeros((side, side), dtype=np.uint8)
        square[(side - bh) // 2:(side - bh) // 2 + bh, (side - bw) // 2:(side - bw) // 2 + bw] = glyph
        digit_img = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(digit_img).float().unsqueeze(0).unsqueeze(0) / 255.0
        tensor = (tensor - 0.1307) / 0.3081
        with torch.no_grad():
            digit = int(model(tensor).argmax(dim=1).item())
        digits.append(digit)
    if not digits:
        return None
    value = 0
    for digit in digits:
        value = value * 10 + digit
    return -value if negative else value


def format_change(previous: int | None, current: int, instrument: str) -> str | None:
    if previous is None:
        previous = 0
    if current == previous:
        return None
    name = f" {instrument}" if instrument else ""
    if current == 0:
        return f"**FLAT{name}** (was {previous:+d})"
    delta = abs(current) - abs(previous)
    if previous == 0 or (current > 0) != (previous > 0):
        side = "BUY" if current > 0 else "SELL"
        if previous != 0:
            close_side = "SELL" if previous > 0 else "BUY"
            return (f"**{side} {abs(current)}{name} at market** "
                    f"(flip: {close_side} {abs(previous)} first; now {current:+d})")
        return f"**{side} {abs(current)}{name} at market** (now {current:+d})"
    if delta > 0:
        return f"**BUY {delta}{name} at market** (now {current:+d})"
    if delta < 0:
        return f"**SELL {abs(delta)}{name} at market** (now {current:+d})"
    return None


def format_commands(previous: int, current: int, instrument: str) -> list[str]:
    """Translate a position change into the NT8 relay command lines."""
    inst = instrument.upper()
    if current == previous:
        return []
    if current == 0:
        return ["FLATTEN"]
    if previous == 0:
        side = "long" if current > 0 else "short"
        return [f"OPEN|{inst}|{side}|{abs(current)}|0|"]
    flip = (current > 0) != (previous > 0)
    if flip:
        side = "long" if current > 0 else "short"
        return ["FLATTEN", f"OPEN|{inst}|{side}|{abs(current)}|0|"]
    delta = abs(current) - abs(previous)
    if delta > 0:
        side = "long" if current > 0 else "short"
        return [f"OPEN|{inst}|{side}|{delta}|0|"]
    if delta < 0:
        return [f"CLOSE|{abs(delta)}"]
    return []


class NT8Bridge:
    def __init__(self, host: str = "127.0.0.1", port: int = 7777, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def send(self, line: str) -> str:
        import socket

        with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
            sock.sendall((line + "\n").encode())
            sock.settimeout(self.timeout)
            try:
                data = sock.recv(200)
            except Exception:
                return ""
        return data.decode(errors="replace").strip()


def post_discord(webhook: str, content: str) -> None:
    body = json.dumps({"content": content}).encode()
    request = urllib.request.Request(
        webhook, data=body,
        headers={"Content-Type": "application/json", "User-Agent": "StreamSignalVision/0.4"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        response.status


def resolve_webhook(direct: str | None, routes_config: str | None, source_id: str) -> str | None:
    if direct:
        return direct
    if not routes_config:
        return None
    path = Path(routes_config)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    routes = data.get("routes", data)
    if source_id in routes:
        return routes[source_id]
    return routes.get(source_id.split("-")[0])


def direct_frames(url: str, fps: float, width: int, height: int):
    """Read frames from any ffmpeg-playable URL (e.g. a Discord/HLS stream)."""
    import subprocess

    frame_bytes = width * height * 3
    process = subprocess.Popen(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "2",
            "-i", url,
            "-vf", f"fps={fps},scale={width}:{height}",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    assert process.stdout is not None
    try:
        while True:
            buf = process.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3))
            yield frame, time.time(), "", None
    finally:
        process.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--roi", default="0.795,0.487,0.042,0.067",
                        help="normalized x,y,w,h of the position number")
    parser.add_argument("--instrument", default="MNQ")
    parser.add_argument("--source-id", default="position-live")
    parser.add_argument("--routes-config", default=None)
    parser.add_argument("--discord-webhook", default=None)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--confirm", type=int, default=2,
                        help="matching reads required before posting (2 at 5fps = 400ms)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--direct", action="store_true",
                        help="treat --url as a direct ffmpeg stream (Discord/HLS feed)")
    parser.add_argument("--nt-host", default="127.0.0.1")
    parser.add_argument("--nt-port", type=int, default=7777)
    parser.add_argument("--nt-instrument", default="MNQ")
    parser.add_argument("--no-nt", action="store_true", help="do not send to NinjaTrader")
    parser.add_argument("--nt-dry-run", action="store_true", help="print NT8 commands without sending")
    args = parser.parse_args()

    roi = tuple(float(v) for v in args.roi.split(","))
    model = load_model()
    webhook = resolve_webhook(args.discord_webhook, args.routes_config, args.source_id)
    bridge = NT8Bridge(args.nt_host, args.nt_port)
    print(f"[position] NT8 target {args.nt_host}:{args.nt_port} instrument={args.nt_instrument}"
          f"{' (dry run)' if args.nt_dry_run or args.no_nt else ''}", flush=True)

    if args.direct:
        print("[position] direct stream mode", flush=True)
        frames = direct_frames(args.url, args.fps, args.width, args.height)
    else:
        frames = live_frames(args.url, args.fps, args.width, args.height, None, 5.0, None)
    previous = 0
    candidate = None
    candidate_count = 0
    started = time.time()
    print(f"[position] watching roi={args.roi} fps={args.fps} confirm={args.confirm}", flush=True)

    for frame, _, stamp, _ in frames:
        value = read_position(frame, roi, model)
        if value is None:
            continue
        if args.verbose:
            print(f"{stamp} read {value:+d}", flush=True)
        if value == candidate:
            candidate_count += 1
        else:
            candidate = value
            candidate_count = 1
        if candidate_count < args.confirm:
            continue
        if value != previous:
            message = format_change(previous, value, args.instrument)
            commands = format_commands(previous, value, args.nt_instrument)
            previous = value
            if message:
                print(f"{stamp} {message}", flush=True)
            if commands:
                for command in commands:
                    if args.nt_dry_run or args.no_nt:
                        print(f"  nt> {command}", flush=True)
                        continue
                    try:
                        reply = bridge.send(command)
                        print(f"  nt> {command}  <- {reply}", flush=True)
                    except Exception as exc:
                        print(f"  nt error: {exc}", flush=True)
            if message and webhook and not args.dry_run:
                try:
                    post_discord(webhook, f"[{args.source_id}] {message}")
                except Exception as exc:
                    print(f"  discord error: {exc}", flush=True)
        if args.max_seconds and time.time() - started > args.max_seconds:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
