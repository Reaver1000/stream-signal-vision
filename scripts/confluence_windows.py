"""Automated confluence curation for ingested VOD sessions.

Proposes trade windows by clustering VLM open frames and requiring spoken
in-trade evidence plus a spoken exit, extracts dense 1 fps segments around
each confirmed window, and rewrites the session's rows in the pooled
state_intervals.jsonl. Unconfirmed VLM open clusters are quarantined as
`unknown` so they never enter training without owner review. Sessions that
already have curated rows are skipped unless --force is given.

Usage:
    # Proposal report only (no writes):
    python scripts/confluence_windows.py data/raw/<session> `
        --intervals data/raw/channel-a-tradingview-vod-01/state_intervals.jsonl --dry-run

    # Curate with detected proposals (extracts dense segments, deletes source.mp4):
    python scripts/confluence_windows.py data/raw/<session> `
        --intervals data/raw/channel-a-tradingview-vod-01/state_intervals.jsonl

    # Curate with explicit reviewed windows (overrides detection):
    python scripts/confluence_windows.py data/raw/<session> `
        --intervals data/raw/channel-a-tradingview-vod-01/state_intervals.jsonl `
        --windows "01:13:18-01:16:51,01:21:00-01:25:35" --split train
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_channel import derive_state_intervals_from_annotations, read_jsonl, write_jsonl

IN_TRADE_CUES = ("stop_update", "target_update", "exit_warning", "position_claim")


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


def vlm_open_clusters(
    annotations: list[dict[str, Any]],
    frames_manifest: list[dict[str, Any]],
    cluster_gap_s: float = 180.0,
) -> list[dict[str, Any]]:
    stamp_by_name = {
        Path(r["image_path"]).name: parse_timestamp(r["video_timestamp"])
        for r in frames_manifest
        if r.get("image_path") and r.get("video_timestamp")
    }
    times: list[float] = []
    for row in annotations:
        ev = row.get("event") or {}
        if str(ev.get("trade_state", "")).lower() != "open":
            continue
        t = stamp_by_name.get(Path(str(row.get("image_path", ""))).name)
        if t is not None:
            times.append(t)
    times.sort()
    clusters: list[dict[str, Any]] = []
    for t in times:
        if clusters and t - clusters[-1]["end"] <= cluster_gap_s:
            clusters[-1]["end"] = t
            clusters[-1]["frames"].append(t)
        else:
            clusters.append({"start": t, "end": t, "frames": [t]})
    return clusters


def detect_window_proposals(
    clusters: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
    min_cluster: int = 2,
    max_exit_gap: float = 600.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (confirmed_windows, rejected_clusters) from VLM clusters + speech."""
    windows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for cluster in clusters:
        if len(cluster["frames"]) < min_cluster:
            rejected.append({**cluster, "reason": "fewer than min_cluster VLM open frames"})
            continue
        exit_row = next(
            (
                r for r in transcript
                if r.get("cue") == "exit_claim"
                and cluster["end"] < r["start_sec"]
                and r["start_sec"] - cluster["end"] <= max_exit_gap
            ),
            None,
        )
        if exit_row is None:
            rejected.append({**cluster, "reason": "no spoken exit within window"})
            continue
        in_trade = [
            r for r in transcript
            if r.get("cue") in IN_TRADE_CUES
            and cluster["start"] <= r["start_sec"] <= exit_row["start_sec"]
        ]
        if not in_trade:
            rejected.append({**cluster, "reason": "no in-trade speech between opens and exit"})
            continue
        windows.append(
            {
                "start": cluster["start"],
                "end": exit_row["start_sec"],
                "n_vlm_opens": len(cluster["frames"]),
                "exit_at": exit_row["start_sec"],
                "exit_text": exit_row.get("text", ""),
                "in_trade_cues": in_trade,
            }
        )
    windows.sort(key=lambda w: w["start"])
    merged: list[dict[str, Any]] = []
    for w in windows:
        if merged and w["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], w["end"])
            merged[-1]["n_vlm_opens"] += w["n_vlm_opens"]
            merged[-1]["in_trade_cues"].extend(w["in_trade_cues"])
        else:
            merged.append(dict(w))
    return merged, rejected


def parse_windows_arg(value: str) -> list[dict[str, Any]]:
    windows = []
    for part in value.split(","):
        start_s, sep, end_s = part.strip().partition("-")
        if not sep:
            raise argparse.ArgumentTypeError(f"window must be START-END: {part}")
        windows.append({"start": parse_timestamp(start_s), "end": parse_timestamp(end_s)})
    return windows


def extract_segment(
    session_dir: Path,
    index: int,
    seg_start: float,
    seg_end: float,
    platform: str,
    source: Path,
) -> Path | None:
    if not source.is_file():
        print(f"  [warn] source video missing; skipping dense segment for window {index}")
        return None
    seg_dir = session_dir / f"segment_auto_{index:02d}"
    frames_dir = seg_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(seg_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", format_timestamp(seg_start), "-to", format_timestamp(seg_end),
        "-i", str(source), "-vf", "fps=1", "-start_number", "1",
        str(frames_dir / "frame_%05d.png"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [warn] ffmpeg failed for segment {index}: {result.stderr[-300:]}")
        return None
    manifest = subprocess.run(
        [
            sys.executable, str(SCRIPTS_DIR / "manifest_ffmpeg.py"), str(seg_dir),
            "--start", format_timestamp(seg_start), "--fps", "1.0", "--platform", platform,
        ],
        capture_output=True, text=True,
    )
    if manifest.returncode != 0:
        print(f"  [warn] manifest failed for segment {index}")
        return None
    return seg_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--intervals", type=Path, required=True)
    parser.add_argument("--windows", default=None, help="explicit 'START-END,START-END' reviewed windows")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--pre-seconds", type=float, default=30.0)
    parser.add_argument("--post-seconds", type=float, default=15.0)
    parser.add_argument("--cluster-gap", type=float, default=180.0)
    parser.add_argument("--min-cluster", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-curate even if session rows already exist")
    parser.add_argument("--keep-video", action="store_true")
    args = parser.parse_args()

    sid = args.session_dir.name
    transcript = read_jsonl(args.session_dir / "transcript.jsonl")
    annotations = read_jsonl(args.session_dir / "cue-visuals.jsonl")
    frames_manifest = read_jsonl(args.session_dir / "frames" / "frames.jsonl")
    session_meta = {}
    session_json = args.session_dir / "session.json"
    if session_json.is_file():
        session_meta = json.loads(session_json.read_text(encoding="utf-8"))
    platform = session_meta.get("platform_id", "topstepx")

    existing_rows = read_jsonl(args.intervals)
    already = [r for r in existing_rows if r.get("source_id") == sid]
    if already and not args.force and not args.dry_run:
        print(f"[{sid}] already has {len(already)} curated row(s); skipping (use --force to redo)")
        return 0

    frame_times = [parse_timestamp(r["video_timestamp"]) for r in frames_manifest]
    first_t, last_t = (min(frame_times), max(frame_times)) if frame_times else (0.0, 0.0)
    clusters = vlm_open_clusters(annotations, frames_manifest, args.cluster_gap)

    print(f"[{sid}] cues={sum(1 for r in transcript if r.get('cue') != 'none')} "
          f"vlm_rows={len(annotations)} vlm_open_clusters={len(clusters)}")
    for idx, c in enumerate(clusters):
        print(f"  cluster {idx}: {format_timestamp(c['start'])}..{format_timestamp(c['end'])} ({len(c['frames'])} opens)")

    if args.windows:
        windows = parse_windows_arg(args.windows)
        for w in windows:
            w.setdefault("n_vlm_opens", sum(
                1 for c in clusters if w["start"] <= c["start"] <= w["end"] or w["start"] <= c["end"] <= w["end"]
            ))
            w.setdefault("exit_text", "(explicit reviewed window)")
            w.setdefault("in_trade_cues", [])
        rejected = [c for c in clusters if not any(w["start"] <= c["start"] and c["end"] <= w["end"] for w in windows)]
    else:
        windows, rejected = detect_window_proposals(clusters, transcript, args.min_cluster)

    for idx, w in enumerate(windows, 1):
        print(f"  WINDOW {idx}: {format_timestamp(w['start'])} -> {format_timestamp(w['end'])} "
              f"(opens={w['n_vlm_opens']}, exit: {w['exit_text'][:70]})")
        for cue in w["in_trade_cues"][:6]:
            print(f"      {cue.get('start')} [{cue.get('cue')}] {cue.get('text', '')[:90]}")
    for c in rejected:
        print(f"  QUARANTINED cluster {format_timestamp(c['start'])}..{format_timestamp(c['end'])}: {c.get('reason', 'outside reviewed windows')}")

    if args.dry_run:
        print(f"[{sid}] dry run; no writes")
        return 0

    spans: list[dict[str, Any]] = []
    for idx, w in enumerate(windows, 1):
        seg_start = max(first_t, w["start"] - args.pre_seconds)
        seg_end = min(last_t, w["end"] + args.post_seconds)
        spans.append({"seg_start": seg_start, "seg_end": seg_end, "w_start": w["start"], "w_end": w["end"]})

    source = args.session_dir / "source.mp4"
    for idx, span in enumerate(spans, 1):
        extract_segment(args.session_dir, idx, span["seg_start"], span["seg_end"], platform, source)
        print(f"  segment_auto_{idx:02d}: {format_timestamp(span['seg_start'])}..{format_timestamp(span['seg_end'])}")

    derived = derive_state_intervals_from_annotations(
        annotations, frames_manifest, source_id=sid, split=args.split,
    )
    rows: list[dict[str, Any]] = []
    for row in derived:
        st, en = parse_timestamp(row["start"]), parse_timestamp(row["end"])
        if any(st <= s["seg_end"] and en >= s["seg_start"] for s in spans):
            continue
        if row["trade_state"] == "open":
            rows.append({**row, "trade_state": "unknown",
                         "reason": "Unconfirmed isolated VLM open claim; excluded pending owner review."})
        else:
            rows.append(row)
    for span in spans:
        pre_end = span["w_start"] - 1
        post_start = span["w_end"] + 1
        if pre_end >= span["seg_start"]:
            rows.append({"source_id": sid, "start": format_timestamp(span["seg_start"]),
                         "end": format_timestamp(pre_end), "trade_state": "flat", "split": args.split,
                         "reason": "Pre-entry chart before confluence-confirmed trade window."})
        rows.append({"source_id": sid, "start": format_timestamp(span["w_start"]),
                     "end": format_timestamp(span["w_end"]), "trade_state": "open", "split": args.split,
                     "reason": f"Confluence-confirmed trade window ({platform})."})
        if post_start <= span["seg_end"]:
            rows.append({"source_id": sid, "start": format_timestamp(post_start),
                         "end": format_timestamp(span["seg_end"]), "trade_state": "flat", "split": args.split,
                         "reason": "Post-exit chart after confluence-confirmed trade window."})

    kept = [r for r in existing_rows if r.get("source_id") != sid]
    kept.extend(rows)
    kept.sort(key=lambda r: (str(r.get("source_id", "")), parse_timestamp(r["start"])))
    write_jsonl(args.intervals, kept)

    if source.is_file() and not args.keep_video:
        source.unlink()

    n_open = len([r for r in rows if r["trade_state"] == "open"])
    n_unk = len([r for r in rows if r["trade_state"] == "unknown"])
    print(f"[{sid}] curated {len(rows)} rows ({n_open} open-window, {n_unk} unknown); "
          f"intervals file now has {len(kept)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
