"""Train from the next small batch of completed videos on a YouTube Streams page."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timedelta, timezone
import json
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
    vods: list[dict[str, Any]], seen_ids: set[str], limit: int,
    deferred_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    chosen = []
    seen = {value.lower() for value in seen_ids | (deferred_ids or set())}
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


def _deferred_path(data_root: Path) -> Path:
    return data_root / "batch_deferred.json"


def load_deferred_ids(data_root: Path, streamer: str) -> set[str]:
    path = _deferred_path(data_root)
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    now = datetime.now(timezone.utc)
    result = set()
    for vod_id, entry in data.get(streamer.lower(), {}).items():
        try:
            until = datetime.fromisoformat(entry["deferred_until"])
        except (KeyError, TypeError, ValueError):
            continue
        if until > now:
            result.add(str(vod_id).lower())
    return result


def defer_vod(data_root: Path, streamer: str, vod_id: str, reason: str, hours: float = 24) -> None:
    path = _deferred_path(data_root)
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        data = {}
    until = datetime.now(timezone.utc) + timedelta(hours=hours)
    data.setdefault(streamer.lower(), {})[vod_id.lower()] = {
        "deferred_until": until.isoformat(),
        "reason": reason,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_external_confusion(output: str) -> dict[str, float] | None:
    match = re.search(r"confusion=(\{.*\})", output)
    if not match:
        return None
    try:
        matrix = ast.literal_eval(match.group(1))
        tp_open = int(matrix["open"]["open"])
        fn_open = int(matrix["open"]["not_open"])
        fp_open = int(matrix["not_open"]["open"])
        tn = int(matrix["not_open"]["not_open"])
    except (ValueError, TypeError, KeyError, SyntaxError):
        return None
    total = tp_open + fn_open + fp_open + tn
    return {
        "accuracy": (tp_open + tn) / max(1, total),
        "open_recall": tp_open / max(1, tp_open + fn_open),
        "false_open": float(fp_open),
        "negative_count": float(fp_open + tn),
    }


def candidate_is_not_worse(candidate: dict[str, float], baseline: dict[str, float]) -> bool:
    return (
        candidate["false_open"] <= baseline["false_open"]
        and candidate["open_recall"] >= baseline["open_recall"]
        and candidate["accuracy"] >= baseline["accuracy"]
    )


def evaluate_external(model: str, dataset: Path, log_fn=None) -> dict[str, float] | None:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate_state_classifier.py"),
        model,
        str(dataset),
        "--split",
        "external",
        "--device",
        "cpu",
        "--max-errors",
        "0",
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in result.stdout.splitlines():
        (log_fn or safe_print)(line)
    if result.returncode != 0:
        (log_fn or safe_print)(f"External evaluation failed for {model}: {result.stderr[-500:]}")
        return None
    return parse_external_confusion(result.stdout)


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
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--until-exhausted", action="store_true",
                        help="continue in five-video batches until no unseen completed VODs remain")
    parser.add_argument("--skip-vod-id", action="append", default=[],
                        help="defer this VOD for 24 hours; repeat the option for multiple IDs")
    parser.add_argument("--run-prefix", default="streams-batch")
    parser.add_argument("--keep-media", action="store_true")
    parser.add_argument("--train-detector", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_batches < 1:
        parser.error("--max-batches must be at least 1")

    safe_print(f"Discovering Streams page metadata: {args.channel}")
    vods = discover_channel_vods(args.channel, max_vods=None)
    completed_count = sum(is_completed_vod(vod) for vod in vods)
    model = args.model
    for vod_id in args.skip_vod_id:
        defer_vod(args.data_root, args.streamer, vod_id, "manually deferred by operator")
    batch_number = 0
    while args.until_exhausted or batch_number < args.max_batches:
        ledger = load_ledger(args.data_root / "channel_ledger.json")
        seen_ids = set(ledger.get("vods_by_streamer", {}).get(args.streamer.lower(), []))
        deferred_ids = load_deferred_ids(args.data_root, args.streamer)
        batch = next_unseen_batch(vods, seen_ids, args.batch_size, deferred_ids)
        safe_print(
            f"Found {len(vods)} entries, {completed_count} completed VODs, "
            f"{len(seen_ids)} ledger IDs, {len(deferred_ids)} deferred, next batch={len(batch)}."
        )
        if not batch:
            safe_print("No unseen completed videos remain in this Streams page snapshot.")
            break
        batch_number += 1
        for index, vod in enumerate(batch, 1):
            safe_print(f"[batch {batch_number} {index}/{len(batch)}] {vod['vod_id']} | {vod['title']}")
            if args.dry_run:
                continue
            previous_model = model
            code, candidate = run_vod(args, vod, model)
            if code != 0:
                defer_vod(args.data_root, args.streamer, vod["vod_id"], f"train_channel exit {code}")
                safe_print(f"Deferred {vod['vod_id']} for 24 hours after exit {code}; continuing this batch.")
                continue
            if args.mode == "loop" and candidate and previous_model and candidate != previous_model:
                dataset = REPO_ROOT / "data" / "derived" / "state-prototype"
                baseline_metrics = evaluate_external(previous_model, dataset)
                candidate_metrics = evaluate_external(candidate, dataset)
                if baseline_metrics and candidate_metrics and candidate_is_not_worse(candidate_metrics, baseline_metrics):
                    model = candidate
                    safe_print(f"Accepted checkpoint {candidate}: held-out metrics did not regress.")
                else:
                    model = previous_model
                    safe_print(f"Kept previous checkpoint {model}: candidate failed the held-out non-regression gate.")
            elif candidate:
                model = candidate
            if model:
                safe_print(f"Next run checkpoint: {model}")
        safe_print(f"Batch {batch_number} complete: {len(batch)} video(s).")
        if args.dry_run:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
