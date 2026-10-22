"""Automated streamer onboarding, training, and back-catalogue evaluation orchestrator.

Designed so a human or an AI agent can point at a streamer's channel or VOD URL
with a single Python command. Orchestrates the existing scripts in scripts/:
  1. Discover back-catalogue VODs via yt-dlp (`discover_channel_vods`)
  2. Transcribe speech cues via `scripts/transcribe_audio.py`
  3. Extract sparse cue-window frames via `scripts/extract_cue_frames.py`
  4. Auto-label boxes + state via `scripts/label_cue_windows.py` (`vlm_label.py`)
  5. Generate timeline + rendered box previews (`make_timeline.py`, `render_annotations.py`)
  6. Prepare & train the pooled YOLO state classifier (`prepare_state_classifier.py`,
     `train_state_classifier.py`, `evaluate_state_classifier.py`) and optional
     pooled YOLO UI detector (`prepare_yolo.py`, `train_detector.py`)
  7. Pull the next unseen VOD from the streamer's back catalogue and measure how
     accurately the trained YOLO model mirrors the trades on screen, mining any
     disagreements as hard examples for the next round.

Usage examples:
  # Dry-run back-catalogue discovery for a streamer channel:
  python scripts/train_channel.py --channel "https://www.youtube.com/@channel-a/streams" \
      --streamer channel-a --platform tradingview --dry-run

  # Ingest, auto-label, and train on a specific VOD URL:
  python scripts/train_channel.py --url "https://www.youtube.com/watch?v=..." \
      --streamer channel-b --platform topstepx --roi "0.0,0.08,0.78,0.82"

  # Run an既存の local session directory through evaluation + hard-example mining:
  python scripts/train_channel.py --session-dir data/raw/channel-b-topstepx-vod-01 \
      --streamer channel-b --platform topstepx --roi "0.0,0.08,0.78,0.82" \
      --mode evaluate --model runs/classify/runs/vision/channel-a-open-not-open-v11/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
VALID_YOLO_KINDS = frozenset(
    {
        "entry_marker",
        "stop_line",
        "target_line",
        "quantity_badge",
        "position_panel",
        "order_ticket",
    }
)


def parse_timestamp(value: str) -> float:
    parts = [float(part) for part in value.split(":")]
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


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip()).strip("-").lower()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text((content + "\n") if content else "", encoding="utf-8")


def seed_pooled_state_intervals(data_root: Path, pooled_path: Path) -> list[dict[str, Any]]:
    """Merge session-local reviewed intervals into the pooled private interval file."""
    rows = read_jsonl(pooled_path)
    known = {
        (r.get("source_id"), r.get("start"), r.get("end"), r.get("trade_state"), r.get("split"))
        for r in rows
    }
    for source_path in sorted(data_root.rglob("state_intervals.jsonl")):
        if source_path.resolve() == pooled_path.resolve():
            continue
        for row in read_jsonl(source_path):
            key = (row.get("source_id"), row.get("start"), row.get("end"), row.get("trade_state"), row.get("split"))
            if key not in known:
                rows.append(row)
                known.add(key)
    write_jsonl(pooled_path, rows)
    return rows


def _write_stream(text: str, stream) -> None:
    """Write text to a stream without crashing on cp1252 consoles (Ultralytics UTF-8 bars)."""
    try:
        stream.write(text)
        stream.flush()
    except UnicodeEncodeError:
        encoding = stream.encoding or "ascii"
        stream.write(text.encode(encoding, "replace").decode(encoding, "replace"))
        stream.flush()


def run_cmd(
    command: list[str],
    *,
    check: bool = True,
    echo_stdout: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    print(f">> {' '.join(str(part) for part in command)}")
    result = subprocess.run(
        [str(part) for part in command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if echo_stdout and result.stdout:
        _write_stream(result.stdout, sys.stdout)
    if check and result.returncode != 0:
        if result.stderr:
            _write_stream(result.stderr, sys.stderr)
        raise SystemExit(result.returncode)
    return result


def ollama_ready(url: str = "http://localhost:11434/api/tags") -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def unload_ollama_model(
    url: str = "http://localhost:11434/api/generate",
    model: str = "qwen2.5vl:7b",
) -> None:
    """Ask Ollama to release VRAM so CUDA whisper/ultralytics do not OOM on shared GPUs."""
    try:
        body = json.dumps({"model": model, "keep_alive": 0}).encode()
        request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()
    except Exception:
        pass


def load_ledger(ledger_path: Path) -> dict[str, Any]:
    if ledger_path.is_file():
        return json.loads(ledger_path.read_text(encoding="utf-8"))
    return {"sessions": {}, "vods_by_streamer": {}}


def save_ledger(ledger_path: Path, ledger: dict[str, Any]) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def discover_channel_vods(channel_url: str, max_vods: int | None = 10, playlist_start: int = 1) -> list[dict[str, Any]]:
    """Fetch VOD metadata from a channel page; omit max_vods to list the full page."""
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--flat-playlist",
        "--dump-json",
    ]
    if playlist_start > 1:
        cmd.extend(["--playlist-start", str(playlist_start)])
    if max_vods is not None:
        cmd.extend(["--playlist-end", str(playlist_start + max_vods - 1)])
    cmd.append(channel_url)
    result = run_cmd(cmd, check=True, echo_stdout=False)
    vods: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        vod_id = item.get("id")
        url = item.get("url") or item.get("webpage_url")
        if vod_id and url and not str(url).startswith("http"):
            url = f"https://www.youtube.com/watch?v={vod_id}"
        if vod_id and url:
            vods.append(
                {
                    "vod_id": str(vod_id),
                    "url": str(url),
                    "title": item.get("title") or "",
                    "duration": item.get("duration"),
                    "live_status": item.get("live_status"),
                    "is_live": bool(item.get("is_live")),
                }
            )
    return vods


def select_unseen_vods(
    vods: list[dict[str, Any]],
    processed_vod_ids: set[str],
    count: int = 1,
) -> list[dict[str, Any]]:
    """Return the first `count` VODs whose vod_id is not yet in processed_vod_ids."""
    chosen: list[dict[str, Any]] = []
    for vod in vods:
        if vod["vod_id"] not in processed_vod_ids:
            chosen.append(vod)
            if len(chosen) >= count:
                break
    return chosen


def derive_state_intervals_from_annotations(
    annotations: list[dict[str, Any]],
    frames_manifest: list[dict[str, Any]],
    *,
    source_id: str,
    split: str = "train",
    max_gap_seconds: float = 5.0,
) -> list[dict[str, Any]]:
    """Collapse per-frame VLM trade_state drafts into contiguous state_intervals rows."""
    stamp_map = {
        Path(row["image_path"]).name: row["video_timestamp"]
        for row in frames_manifest
        if row.get("image_path") and row.get("video_timestamp")
    }
    timed: list[tuple[float, str, str]] = []
    for row in annotations:
        name = Path(str(row.get("image_path", ""))).name
        stamp = stamp_map.get(name)
        if not stamp:
            continue
        ev = row.get("event") or {}
        state = str(ev.get("trade_state") or "unknown").lower()
        if state not in ("open", "pending", "flat"):
            continue
        # Enforce project rule: order_ticket alone without filled-position evidence is pending, not open,
        # and VLM records flagged with sanity issues in needs_review cannot auto-promote to open.
        kinds = {str(el.get("kind")) for el in (row.get("elements") or [])}
        has_pos = bool(kinds & {"position_panel", "entry_marker", "quantity_badge"})
        if state == "open" and ("order_ticket" in kinds and not has_pos):
            state = "pending"
        if state == "open" and (row.get("needs_review") or not has_pos):
            continue
        timed.append((parse_timestamp(stamp), stamp, state))

    timed.sort(key=lambda item: item[0])
    intervals: list[dict[str, Any]] = []
    for sec, stamp, state in timed:
        if (
            intervals
            and intervals[-1]["trade_state"] == state
            and sec - intervals[-1]["_end_sec"] <= max_gap_seconds
        ):
            intervals[-1]["end"] = stamp
            intervals[-1]["_end_sec"] = sec
        else:
            intervals.append(
                {
                    "source_id": source_id,
                    "start": stamp,
                    "end": stamp,
                    "trade_state": state,
                    "split": split,
                    "reason": f"Auto-derived from {source_id} cue-window VLM state ({state}).",
                    "_end_sec": sec,
                }
            )
    for row in intervals:
        row.pop("_end_sec", None)
    return intervals


def sanitize_yolo_records(
    annotations: list[dict[str, Any]],
    *,
    frames_dir: Path,
    dataset_root: Path,
    session_id: str,
    platform_id: str,
) -> list[dict[str, Any]]:
    """Normalize VLM annotation records so scripts/prepare_yolo.py accepts them cleanly."""
    clean_rows: list[dict[str, Any]] = []
    root_resolved = dataset_root.resolve()
    for row in annotations:
        name = Path(str(row.get("image_path", ""))).name
        img_file = (frames_dir / name).resolve()
        if not img_file.is_file() or not img_file.is_relative_to(root_resolved):
            continue
        rel_img = img_file.relative_to(root_resolved).as_posix()
        valid_els: list[dict[str, Any]] = []
        for el in row.get("elements") or []:
            kind = el.get("kind")
            bbox = el.get("bbox")
            if kind not in VALID_YOLO_KINDS or not isinstance(bbox, list) or len(bbox) != 4:
                continue
            try:
                x, y, w, h = (float(v) for v in bbox)
            except (TypeError, ValueError):
                continue
            if 0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1 and x + w <= 1 and y + h <= 1:
                valid_els.append({"kind": kind, "bbox": [x, y, w, h], "text": el.get("text")})
        clean_rows.append(
            {
                "image_path": rel_img,
                "platform_id": platform_id if row.get("platform_id") in (None, "", "other") else row["platform_id"],
                "session_id": session_id,
                "elements": valid_els,
                "event": row.get("event"),
            }
        )
    return clean_rows


def ensure_three_session_ids_for_bootstrap(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """When fewer than 3 sessions exist, partition chronologically into <session>-s1/s2/s3 blocks.

    This satisfies split_by_session()'s >=3 session requirement in src/stream_signal_vision/dataset.py
    without leaking random adjacent frames across splits or modifying dataset.py.
    """
    sessions = sorted({str(r.get("session_id", "")) for r in records if r.get("session_id")})
    if len(sessions) >= 3 or len(records) < 3:
        return records
    ordered = sorted(records, key=lambda r: (str(r.get("session_id", "")), str(r.get("image_path", ""))))
    n = len(ordered)
    cut1 = max(1, int(round(n * 0.70)))
    cut2 = max(cut1 + 1, int(round(n * 0.85)))
    if cut2 >= n:
        cut1 = max(1, n - 2)
        cut2 = n - 1
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(ordered):
        copy_row = dict(row)
        base = str(row.get("session_id", "session"))
        if idx < cut1:
            copy_row["session_id"] = f"{base}-part1"
        elif idx < cut2:
            copy_row["session_id"] = f"{base}-part2"
        else:
            copy_row["session_id"] = f"{base}-part3"
        out.append(copy_row)
    return out


def download_vod_audio(url: str, session_dir: Path) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    for existing in session_dir.glob("source_audio.*"):
        if existing.is_file():
            return existing
    target_template = session_dir / "source_audio.%(ext)s"
    run_cmd(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "-f",
            "139/ba[ext=m4a]/ba",
            "--no-progress",
            "-o",
            str(target_template),
            "--no-playlist",
            url,
        ]
    )
    matches = list(session_dir.glob("source_audio.*"))
    if matches:
        return matches[0]
    raise SystemExit(f"yt-dlp did not produce an audio file in {session_dir}")


def download_vod_video(url: str, session_dir: Path, quality: str = "298/136/bv*[height<=720][protocol=https]/bv*[height<=720]/best[height<=720]/best") -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    target = session_dir / "source.mp4"
    if target.is_file():
        return target
    run_cmd(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "-f",
            quality,
            "--no-progress",
            "-o",
            str(target),
            "--no-playlist",
            url,
        ]
    )
    if target.is_file():
        return target
    matches = list(session_dir.glob("source.*"))
    if matches:
        return matches[0]
    raise SystemExit(f"yt-dlp did not produce a video file in {session_dir}")


def ingest_and_label_session(
    *,
    session_dir: Path,
    session_id: str,
    platform: str,
    url: str | None = None,
    media_path: Path | None = None,
    start: str | None = None,
    end: str | None = None,
    cleanup_media: bool = False,
) -> dict[str, Path]:
    """Run Whisper cue transcription, sparse cue-frame extraction, VLM labeling, and timeline export."""
    session_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = session_dir / "transcript.jsonl"
    frames_dir = session_dir / "frames"
    manifest_path = frames_dir / "frames.jsonl"
    annotations_path = session_dir / "cue-visuals.jsonl"
    timeline_path = session_dir / "timeline.txt"
    previews_dir = session_dir / "previews"

    audio_media = media_path
    if not transcript_path.is_file():
        if audio_media is None and url:
            audio_media = download_vod_audio(url, session_dir)
        if audio_media is None or not audio_media.is_file():
            raise SystemExit(f"media file required to transcribe {session_id}")
        unload_ollama_model()
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "transcribe_audio.py"),
            str(audio_media),
            "--out",
            str(transcript_path),
        ]
        if start:
            cmd.extend(["--start", start])
        if end:
            cmd.extend(["--end", end])
        run_cmd(cmd)

    video_media = media_path
    if not manifest_path.is_file():
        if video_media is None and url:
            video_media = download_vod_video(url, session_dir)
        if video_media is None or not video_media.is_file():
            raise SystemExit(f"video file required to extract cue frames for {session_id}")
        run_cmd(
            [
                sys.executable,
                str(SCRIPTS_DIR / "extract_cue_frames.py"),
                str(video_media),
                str(transcript_path),
                str(session_dir),
                "--platform",
                platform,
                "--session-id",
                session_id,
            ]
        )

    if cleanup_media:
        for candidate in [*session_dir.glob("source.*"), *session_dir.glob("source_audio.*")]:
            if candidate.is_file():
                candidate.unlink(missing_ok=True)

    if not annotations_path.is_file() and ollama_ready():
        run_cmd(
            [
                sys.executable,
                str(SCRIPTS_DIR / "label_cue_windows.py"),
                str(frames_dir),
                str(transcript_path),
                "--out",
                str(annotations_path),
                "--session-id",
                session_id,
            ]
        )

    if annotations_path.is_file() and read_jsonl(annotations_path):
        tl = run_cmd(
            [
                sys.executable,
                str(SCRIPTS_DIR / "make_timeline.py"),
                str(annotations_path),
                "--manifest",
                str(manifest_path),
            ],
            check=False,
        )
        timeline_path.write_text(tl.stdout, encoding="utf-8")
        run_cmd(
            [
                sys.executable,
                str(SCRIPTS_DIR / "render_annotations.py"),
                str(annotations_path),
                str(frames_dir),
                str(previews_dir),
                "--limit",
                "50",
            ],
            check=False,
        )

    return {
        "transcript": transcript_path,
        "frames_dir": frames_dir,
        "manifest": manifest_path,
        "annotations": annotations_path,
        "timeline": timeline_path,
    }


def evaluate_session_with_classifier(
    *,
    model_path: Path,
    frames_dir: Path,
    roi: str | None = None,
    annotations_path: Path | None = None,
    open_threshold: float = 0.5,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Run the YOLO state classifier over a session's frames and compare against VLM/verified states."""
    import cv2
    from ultralytics import YOLO

    manifest = read_jsonl(frames_dir / "frames.jsonl")
    manifest.sort(key=lambda r: parse_timestamp(r["video_timestamp"]))
    roi_tuple = tuple(float(v) for v in roi.split(",")) if roi else None

    model = YOLO(str(model_path))
    open_id = next((idx for idx, name in model.names.items() if name == "open"), 0)

    predictions: list[dict[str, Any]] = []
    for start in range(0, len(manifest), batch_size):
        chunk = manifest[start : start + batch_size]
        imgs = []
        valid_rows = []
        for row in chunk:
            img_path = frames_dir / Path(row["image_path"]).name
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            if roi_tuple is not None:
                h, w = img.shape[:2]
                x, y, bw, bh = roi_tuple
                img = img[int(y * h) : int((y + bh) * h), int(x * w) : int((x + bw) * w)]
            imgs.append(img)
            valid_rows.append(row)
        if not imgs:
            continue
        results = model.predict(imgs, imgsz=512, device="cpu", verbose=False)
        for row, res in zip(valid_rows, results):
            prob = float(res.probs.data[open_id])
            predictions.append(
                {
                    "image_path": Path(row["image_path"]).name,
                    "video_timestamp": row["video_timestamp"],
                    "timestamp_sec": parse_timestamp(row["video_timestamp"]),
                    "open_prob": round(prob, 4),
                    "predicted_state": "open" if prob >= open_threshold else "not_open",
                }
            )

    # Compare against VLM/verified annotations if available (excluding sanity-flagged open guesses)
    ref_map: dict[str, str] = {}
    if annotations_path and annotations_path.is_file():
        for row in read_jsonl(annotations_path):
            ev = row.get("event") or {}
            st = str(ev.get("trade_state") or "unknown").lower()
            if st == "open" and row.get("needs_review"):
                continue
            if st in ("open", "pending", "flat"):
                ref_map[Path(str(row["image_path"])).name] = "open" if st == "open" else "not_open"

    compared = 0
    matches = 0
    disagreements: list[dict[str, Any]] = []
    for pred in predictions:
        ref = ref_map.get(pred["image_path"])
        if ref is None:
            continue
        compared += 1
        if ref == pred["predicted_state"]:
            matches += 1
        else:
            disagreements.append({**pred, "reference_state": ref})

    open_frames = [p for p in predictions if p["predicted_state"] == "open"]
    spans: list[dict[str, Any]] = []
    for item in open_frames:
        sec = float(item["timestamp_sec"])
        if spans and sec - float(spans[-1]["_end_sec"]) <= 2.0:
            spans[-1]["end"] = item["video_timestamp"]
            spans[-1]["_end_sec"] = sec
            spans[-1]["frames"] += 1
            spans[-1]["max_open_prob"] = max(spans[-1]["max_open_prob"], item["open_prob"])
        else:
            spans.append(
                {
                    "start": item["video_timestamp"],
                    "end": item["video_timestamp"],
                    "frames": 1,
                    "max_open_prob": item["open_prob"],
                    "first_frame": item["image_path"],
                    "_end_sec": sec,
                }
            )
    for span in spans:
        span.pop("_end_sec", None)

    return {
        "frames_evaluated": len(predictions),
        "predicted_open_frames": len(open_frames),
        "predicted_open_spans": spans,
        "reference_compared": compared,
        "reference_agreement": round(matches / compared, 4) if compared else None,
        "disagreements": disagreements,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default=None, help="Streamer channel/streams URL for back-catalogue discovery")
    parser.add_argument("--url", default=None, help="Single VOD URL to ingest/train/evaluate")
    parser.add_argument("--session-dir", type=Path, default=None, help="Existing local session directory")
    parser.add_argument("--streamer", required=True, help="Short streamer identifier (e.g. channel-a, channel-b)")
    parser.add_argument("--platform", required=True, choices=("tradingview", "topstepx", "tradovate", "wealthcharts", "mt5"))
    parser.add_argument("--roi", default=None, help="Normalized x,y,width,height chart crop for this streamer")
    parser.add_argument("--open-stride", type=int, default=10, help="Seconds between sampled open frames")
    parser.add_argument("--not-open-stride", type=int, default=1, help="Seconds between sampled flat/pending frames")
    parser.add_argument("--edge-window", type=float, default=12.0, help="Sample every frame near interval boundaries")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"), help="Local private raw data root")
    parser.add_argument("--intervals", type=Path, default=None, help="Path to pooled state_intervals.jsonl")
    parser.add_argument("--dataset-out", type=Path, default=Path("data/derived/state-prototype"))
    parser.add_argument("--model", type=Path, default=Path("yolo26n-cls.pt"), help="Classifier base or trained checkpoint")
    parser.add_argument("--run-name", default=None, help="Training run name")
    parser.add_argument("--mode", choices=("discover", "ingest", "train", "evaluate", "loop"), default="loop")
    parser.add_argument("--max-vods", type=int, default=10)
    parser.add_argument("--train-detector", action="store_true", help="Also export and train the pooled YOLO object detector")
    parser.add_argument("--cleanup-media", action="store_true", help="Delete full downloaded VOD mp4 after cue extraction")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    streamer = safe_slug(args.streamer)
    ledger_path = args.data_root / "channel_ledger.json"
    ledger = load_ledger(ledger_path)

    if args.mode == "discover" or (args.channel and args.dry_run):
        if not args.channel:
            parser.error("--channel is required for discovery")
        vods = discover_channel_vods(args.channel, max_vods=args.max_vods)
        seen = set(ledger.get("vods_by_streamer", {}).get(streamer, []))
        print(f"Discovered {len(vods)} VOD(s) for {streamer} ({len(seen)} already processed):")
        for vod in vods:
            status = "SEEN" if vod["vod_id"] in seen else "NEW"
            print(f"  [{status}] {vod['vod_id']}  {vod['url']}  {vod['title']}")
        return 0

    # Resolve target session(s)
    session_dir = args.session_dir
    vod_url = args.url
    vod_id = None
    if session_dir is None:
        if vod_url is None and args.channel:
            vods = discover_channel_vods(args.channel, max_vods=args.max_vods)
            seen = set(ledger.get("vods_by_streamer", {}).get(streamer, []))
            unseen = select_unseen_vods(vods, seen, count=1)
            if not unseen:
                print(f"All discovered VODs for {streamer} have already been processed.")
                return 0
            vod_url = unseen[0]["url"]
            vod_id = unseen[0]["vod_id"]
        elif vod_url:
            vod_id = safe_slug(vod_url.split("v=")[-1].split("/")[-1][:16])
        else:
            parser.error("Provide --channel, --url, or --session-dir")
        session_id = f"{streamer}-{args.platform}-{vod_id}"
        session_dir = args.data_root / session_id
    else:
        session_id = session_dir.name

    artifacts = ingest_and_label_session(
        session_dir=session_dir,
        session_id=session_id,
        platform=args.platform,
        url=vod_url,
        cleanup_media=args.cleanup_media,
    )

    if args.mode == "ingest":
        if vod_id:
            if args.roi:
                ledger.setdefault("rois", {})[streamer] = args.roi
            ledger.setdefault("vods_by_streamer", {}).setdefault(streamer, [])
            if vod_id not in ledger["vods_by_streamer"][streamer]:
                ledger["vods_by_streamer"][streamer].append(vod_id)
            ledger.setdefault("sessions", {})[session_id] = {
                "streamer": streamer,
                "platform": args.platform,
                "vod_id": vod_id,
                "url": vod_url,
                "session_dir": str(session_dir),
            }
            save_ledger(ledger_path, ledger)
        return 0

    if args.mode == "evaluate":
        eval_report = evaluate_session_with_classifier(
            model_path=args.model,
            frames_dir=artifacts["frames_dir"],
            roi=args.roi,
            annotations_path=artifacts["annotations"],
        )
        out_path = session_dir / "evaluation_report.json"
        out_path.write_text(json.dumps(eval_report, indent=2) + "\n", encoding="utf-8")
        print(
            f"Evaluated {session_id}: frames={eval_report['frames_evaluated']} "
            f"predicted_open={eval_report['predicted_open_frames']} "
            f"compared={eval_report['reference_compared']} "
            f"agreement={eval_report['reference_agreement']}"
        )
        print(f"Saved evaluation report to {out_path}")
        return 0

    # Record session and ROI in ledger before building pooled sources
    if args.roi:
        ledger.setdefault("rois", {})[streamer] = args.roi
    if vod_id:
        ledger.setdefault("vods_by_streamer", {}).setdefault(streamer, [])
        if vod_id not in ledger["vods_by_streamer"][streamer]:
            ledger["vods_by_streamer"][streamer].append(vod_id)
    ledger.setdefault("sessions", {})[session_id] = {
        "streamer": streamer,
        "platform": args.platform,
        "vod_id": vod_id,
        "url": vod_url,
        "session_dir": str(session_dir),
    }
    save_ledger(ledger_path, ledger)

    # Train / Loop mode: update state_intervals.jsonl if needed and run prepare + train + evaluate
    intervals_path = args.intervals or (args.data_root / "state_intervals.jsonl")
    existing_intervals = seed_pooled_state_intervals(args.data_root, intervals_path)
    derived = derive_state_intervals_from_annotations(
        read_jsonl(artifacts["annotations"]),
        read_jsonl(artifacts["manifest"]),
        source_id=session_id,
        split="train",
    )
    if derived and not any(r.get("source_id") == session_id for r in existing_intervals):
        existing_intervals.extend(derived)
        write_jsonl(intervals_path, existing_intervals)
        print(f"Added {len(derived)} auto-derived state interval(s) for {session_id} -> {intervals_path}")

    if existing_intervals:
        if args.dataset_out.exists():
            shutil.rmtree(args.dataset_out)
        prep_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "prepare_state_classifier.py"),
            str(intervals_path),
            "--output",
            str(args.dataset_out),
        ]
        # Each frames directory is keyed ONLY by its session id so a session's
        # frames can never match another VOD's intervals at overlapping timestamps.
        rois_map = dict(ledger.get("rois", {}))
        for sid, sinfo in ledger.get("sessions", {}).items():
            s_streamer = sinfo.get("streamer", sid)
            if s_streamer in rois_map:
                rois_map[sid] = rois_map[s_streamer]
            s_dir = Path(sinfo["session_dir"])
            for candidate_frames in [s_dir / "frames", *s_dir.glob("segment_*/frames")]:
                if (candidate_frames / "frames.jsonl").is_file():
                    prep_cmd.extend(["--source", f"{sid}={candidate_frames}"])
        for key, roi_val in rois_map.items():
            prep_cmd.extend(["--roi", f"{key}={roi_val}"])
        prep_cmd.extend([
            "--open-stride", str(args.open_stride),
            "--not-open-stride", str(args.not_open_stride),
            "--edge-window", str(args.edge_window),
        ])
        run_cmd(prep_cmd)

        run_name = args.run_name or f"pooled-{session_id}-cls"
        run_cmd(
            [
                sys.executable,
                str(SCRIPTS_DIR / "train_state_classifier.py"),
                str(args.dataset_out),
                "--model",
                str(args.model),
                "--name",
                run_name,
            ]
        )

    if args.train_detector and artifacts["annotations"].is_file():
        clean = sanitize_yolo_records(
            read_jsonl(artifacts["annotations"]),
            frames_dir=artifacts["frames_dir"],
            dataset_root=REPO_ROOT,
            session_id=session_id,
            platform_id=args.platform,
        )
        if len(clean) >= 3:
            pooled_manifest = args.data_root / "pooled_annotations.jsonl"
            write_jsonl(pooled_manifest, ensure_three_session_ids_for_bootstrap(clean))
            yolo_out = Path("data/derived/pooled-yolo")
            run_cmd(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "prepare_yolo.py"),
                    str(pooled_manifest),
                    "--root",
                    str(REPO_ROOT),
                    "--output",
                    str(yolo_out),
                ]
            )
            run_cmd(
                [
                    sys.executable,
                    str(SCRIPTS_DIR / "train_detector.py"),
                    str(yolo_out / "dataset.yaml"),
                    "--name",
                    f"{streamer}-ui-detector",
                ]
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
