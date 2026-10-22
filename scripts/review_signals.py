"""Build the trade-signal review spreadsheet (CSV) for line-by-line owner review.

One row per confluence-confirmed trade window across all ingested sessions:
video URL, signal timestamp, window end, the curated direction (speech/VLM
confluence), and a fresh red/green-box read taken at the entry frame. Review
columns are left blank for the owner to fill line by line.

Reads (all private):
- <data-root>/channel-a-tradingview-vod-01/state_intervals.jsonl  (curated windows)
- <data-root>/channel_ledger.json                               (VOD ids/URLs)
- <data-root>/signal_notes.json                                 (curated directions + evidence quotes)

Usage:
    python scripts/review_signals.py --data-root <private raw root> `
        --out <private raw root>/review/trade_signals_review.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from trade_details import read_trade_details


def parse_timestamp(value: str) -> float:
    parts = [float(p) for p in value.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def entry_frame(session_dir: Path, window_start: float, offset: float = 8.0):
    """Find the frame nearest window_start+offset across the session's frame manifests."""
    import cv2

    best = None
    want = window_start + offset
    for manifest_path in [session_dir / "frames" / "frames.jsonl", *sorted(session_dir.glob("segment_*/frames/frames.jsonl"))]:
        for row in read_jsonl(manifest_path):
            t = parse_timestamp(row["video_timestamp"])
            distance = abs(t - want)
            if best is None or distance < best[0]:
                best = (distance, manifest_path.parent / Path(row["image_path"]).name, t)
    if best is None:
        return None, None
    _, frame_path, frame_t = best
    image = cv2.imread(str(frame_path))
    return image, frame_t


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offset", type=float, default=8.0, help="seconds after window start to read details")
    args = parser.parse_args()

    intervals_path = args.data_root / "channel-a-tradingview-vod-01" / "state_intervals.jsonl"
    ledger = json.loads((args.data_root / "channel_ledger.json").read_text(encoding="utf-8"))
    notes_path = args.data_root / "signal_notes.json"
    notes = json.loads(notes_path.read_text(encoding="utf-8")) if notes_path.is_file() else {}

    sessions_by_id = ledger.get("sessions", {})
    rows = []
    for interval in read_jsonl(intervals_path):
        if interval.get("trade_state") != "open":
            continue
        session_dir = Path(str(sessions_by_id.get(interval["source_id"], {}).get("session_dir", "")))
        if not session_dir.is_dir():
            continue
        vod_id = sessions_by_id.get(interval["source_id"], {}).get("vod_id")
        start_t = parse_timestamp(interval["start"])

        note = notes.get(interval["source_id"], {}).get(interval["start"], {})
        image, frame_t = entry_frame(session_dir, start_t, args.offset)
        box = {}
        if image is not None:
            try:
                box = read_trade_details(image)
            except Exception as exc:
                box = {"description": f"box read failed: {exc}"}

        rows.append({
            "session": interval["source_id"],
            "video_url": f"https://www.youtube.com/watch?v={vod_id}" if vod_id else "",
            "signal_time": interval["start"],
            "read_time": format_ts(frame_t) if frame_t is not None else "",
            "window_end": interval["end"],
            "my_direction": note.get("direction") or "",
            "entry_type": note.get("entry_type") or "",
            "evidence": (note.get("evidence") or interval.get("reason") or "")[:160],
            "box_direction": box.get("direction") or "",
            "box_instrument": box.get("instrument") or "",
            "box_entry": box.get("entry_price") or "",
            "box_sl": box.get("stop_price") or "",
            "box_tp": box.get("target_price") or "",
            "box_qty": box.get("quantity") or "",
            "box_flags": "; ".join(box.get("notes", [])) if isinstance(box, dict) else "",
            "verified": "yes" if note.get("verified") else "",
            "review_verdict": "",
            "actual_direction": "",
            "actual_entry_type": "",
            "actual_entry": "",
            "actual_sl": "",
            "actual_tp": "",
            "actual_qty": "",
            "notes": "",
        })

    def sort_key(row):
        return (row["session"], parse_timestamp(row["signal_time"]))

    rows.sort(key=sort_key)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["session"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} signal rows -> {args.out}")
    return 0


def format_ts(seconds: float | None) -> str:
    if seconds is None:
        return ""
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


if __name__ == "__main__":
    raise SystemExit(main())
