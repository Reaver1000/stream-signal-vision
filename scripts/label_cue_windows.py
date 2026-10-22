"""Focus VLM frame labeling around timestamped ASR cues.

This does not treat speech as a signal. It selects nearby video frames for
visual confirmation, then delegates frame labeling to vlm_label.py.

Usage:
    python scripts/label_cue_windows.py <frames_dir> <transcript.jsonl> \
        --out cue-visuals.jsonl --dry-run
    python scripts/label_cue_windows.py <frames_dir> <transcript.jsonl> \
        --out cue-visuals.jsonl
"""
import argparse
import bisect
import json
import statistics
import subprocess
import sys
from pathlib import Path


def parse_timestamp(value):
    parts = [float(p) for p in value.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def merge_windows(windows, max_gap=0):
    merged = []
    for start, end, cue in sorted(windows, key=lambda item: item[0]):
        if merged and start <= merged[-1][1] + max_gap:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2].append(cue)
        else:
            merged.append([start, end, [cue]])
    return merged


def evenly_spaced_indices(length, count):
    if count <= 0 or length <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [(length - 1) // 2]
    return [round(i * (length - 1) / (count - 1)) for i in range(count)]


def allocate_window_limits(window_count, keyframe_budget):
    selected_count = min(window_count, keyframe_budget)
    indices = evenly_spaced_indices(window_count, selected_count)
    if not indices:
        return []
    base, remainder = divmod(keyframe_budget, len(indices))
    return [(index, base + (position < remainder)) for position, index in enumerate(indices)]


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frames", type=Path)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--before", type=float, default=6.0)
    parser.add_argument("--after", type=float, default=8.0)
    parser.add_argument("--merge-gap", type=float, default=3.0)
    parser.add_argument("--min-logprob", type=float, default=-1.5)
    parser.add_argument("--max-no-speech", type=float, default=0.6)
    parser.add_argument("--max-keyframes", type=int, default=120,
                        help="Maximum VLM-labelled keyframes for this VOD (spread across cue windows)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = args.manifest or args.frames / "frames.jsonl"
    frame_rows = read_jsonl(manifest)
    frame_rows.sort(key=lambda row: parse_timestamp(row["video_timestamp"]))
    times = [parse_timestamp(row["video_timestamp"]) for row in frame_rows]
    if not frame_rows:
        raise SystemExit(f"no frame records in {manifest}")

    cues = []
    for row in read_jsonl(args.transcript):
        if row.get("cue", "none") == "none":
            continue
        if float(row.get("avg_logprob", 0)) < args.min_logprob:
            continue
        if float(row.get("no_speech_prob", 1)) > args.max_no_speech:
            continue
        start = max(times[0], float(row["start_sec"]) - args.before)
        end = min(times[-1], float(row["end_sec"]) + args.after)
        left = bisect.bisect_left(times, start)
        right = bisect.bisect_right(times, end)
        if left < right:
            cue = {"time": row.get("start"), "cue": row["cue"],
                   "text": row.get("text", "")}
            cues.append((left, right, cue))

    intervals = [b - a for a, b in zip(times, times[1:]) if b > a]
    frame_rate = 1.0 / statistics.median(intervals) if intervals else 1.0
    merged = merge_windows(cues, max_gap=args.merge_gap * frame_rate)

    if not merged:
        print("no transcript cues passed the confidence filters")
        return

    if args.max_keyframes < 1:
        parser.error("--max-keyframes must be at least 1")

    labeler = Path(__file__).with_name("vlm_label.py")
    session_id = args.session_id or args.frames.parent.name
    args.out.parent.mkdir(parents=True, exist_ok=True)
    window_limits = allocate_window_limits(len(merged), args.max_keyframes)
    print(f"{len(merged)} merged cue window(s); VLM label budget={sum(limit for _, limit in window_limits)}")
    for window_index, limit in window_limits:
        start, end, cue_group = merged[window_index]
        cue_text = " | ".join(f"{c['time']} {c['cue']}: {c['text']}" for c in cue_group)
        print(f"frames[{start}:{end}] {cue_text}")
        if args.dry_run:
            continue
        command = [sys.executable, str(labeler), str(args.frames),
                   "--out", str(args.out), "--session-id", session_id,
                   "--start", str(start), "--end", str(end), "--limit", str(limit)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
