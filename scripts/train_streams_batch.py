"""Train from the next small batch of completed videos on a YouTube Streams page."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from train_channel import discover_channel_vods, load_ledger

REPO_ROOT = Path(__file__).resolve().parents[1]


def is_completed_vod(vod: dict[str, Any]) -> bool:
    status = str(vod.get("live_status") or "").lower()
    if vod.get("is_live") or status in {"is_live", "is_upcoming", "is_upcoming_live"}:
        return False
    return vod.get("duration") is not None or status == "was_live"


def next_unseen_batch(
    vods: list[dict[str, Any]], seen_ids: set[str], limit: int
) -> list[dict[str, Any]]:
    chosen = []
    seen = {value.lower() for value in seen_ids}
    queued: set[str] = set()
    for vod in vods:
        vod_id = str(vod.get("vod_id") or "")
        normalized = vod_id.lower()
        if not is_completed_vod(vod) or not normalized or normalized in seen or normalized in queued:
            continue
        chosen.append(vod)
        queued.add(normalized)
        if len(chosen) >= limit:
            break
    return chosen


def safe_print(text: str) -> None:
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "ascii"
        print(text.encode(encoding, "replace").decode(encoding, "replace"), flush=True)


def run_vod(args: argparse.Namespace, vod: dict[str, Any], model: str) -> tuple[int, str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "train_channel.py"),
        "--url",
        str(vod["url"]),
        "--streamer",
        args.streamer,
        "--platform",
        args.platform,
        "--data-root",
        str(args.data_root),
        "--mode",
        args.mode,
    ]
    if args.roi:
        command.extend(["--roi", args.roi])
    if model:
        command.extend(["--model", model])
    if args.run_prefix:
        command.extend(["--run-name", f"{args.run_prefix}-{vod['vod_id']}"])
    if not args.keep_media:
        command.append("--cleanup-media")
    if args.train_detector:
        command.append("--train-detector")

    safe_print("$ " + subprocess.list2cmdline(command))
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    latest_model = model
    assert process.stdout is not None
    for line in process.stdout:
        text = line.rstrip()
        safe_print(text)
        match = re.search(r"best checkpoint:\s*(.+?)\s*$", text, re.IGNORECASE)
        if match:
            candidate = Path(match.group(1).strip().strip('"'))
            if not candidate.is_absolute():
                candidate = REPO_ROOT / candidate
            if candidate.is_file():
                latest_model = str(candidate)
    return process.wait(), latest_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", required=True, help="YouTube channel /streams URL")
    parser.add_argument("--streamer", required=True)
    parser.add_argument("--platform", required=True, choices=("tradingview", "topstepx", "tradovate", "wealthcharts", "mt5"))
    parser.add_argument("--roi", default=None)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--model", default="")
    parser.add_argument("--mode", choices=("loop", "ingest"), default="loop")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--run-prefix", default="streams-batch")
    parser.add_argument("--keep-media", action="store_true")
    parser.add_argument("--train-detector", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    ledger = load_ledger(args.data_root / "channel_ledger.json")
    seen_ids = set(ledger.get("vods_by_streamer", {}).get(args.streamer.lower(), []))
    safe_print(f"Discovering Streams page metadata: {args.channel}")
    vods = discover_channel_vods(args.channel, max_vods=None)
    completed_count = sum(is_completed_vod(vod) for vod in vods)
    batch = next_unseen_batch(vods, seen_ids, args.batch_size)
    safe_print(
        f"Found {len(vods)} entries, {completed_count} completed VODs, "
        f"{len(seen_ids)} ledger IDs, next batch={len(batch)}."
    )
    if not batch:
        safe_print("No unseen completed videos in this batch.")
        return 0

    model = args.model
    for index, vod in enumerate(batch, 1):
        safe_print(f"[batch {index}/{len(batch)}] {vod['vod_id']} | {vod['title']}")
        if args.dry_run:
            continue
        code, model = run_vod(args, vod, model)
        if code != 0:
            safe_print(f"Stopping batch after failure on {vod['vod_id']} (exit {code}).")
            return code
        if model:
            safe_print(f"Next run checkpoint: {model}")
    safe_print(f"Batch complete: {len(batch)} video(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
