"""Extract sparse frames from local VODs only around timestamped speech cues.

Requires ffmpeg on PATH. This is an offline training/review helper; the live
video path remains continuous and does not screenshot-poll.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess


def parse_timestamp(value):
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def format_timestamp(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def merge_intervals(intervals, gap_seconds=0):
    merged = []
    for start, end, cues in sorted(intervals, key=lambda item: item[0]):
        if merged and start <= merged[-1][1] + gap_seconds:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2].extend(cues)
        else:
            merged.append([start, end, list(cues)])
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media", type=Path)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--before", type=float, default=6.0)
    parser.add_argument("--after", type=float, default=8.0)
    parser.add_argument("--merge-gap", type=float, default=3.0)
    parser.add_argument("--fps", type=float, default=0.25,
                        help="Sparse cue-window sampling rate; default 0.25 fps (one frame every 4 seconds)")
    parser.add_argument("--image-format", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--jpeg-qscale", type=int, default=9,
                        help="FFmpeg JPEG q:v scale (2=best/largest, 31=worst/smallest)")
    parser.add_argument("--min-logprob", type=float, default=-1.5)
    parser.add_argument("--max-no-speech", type=float, default=0.6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is required on PATH")
    if not args.media.is_file() and not args.dry_run:
        raise SystemExit(f"media not found: {args.media}")

    cues = []
    with args.transcript.open(encoding="utf-8") as transcript:
        for line in transcript:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("cue", "none") == "none":
                continue
            if float(row.get("avg_logprob", 0)) < args.min_logprob:
                continue
            if float(row.get("no_speech_prob", 1)) > args.max_no_speech:
                continue
            start = max(0, math.floor(float(row["start_sec"]) - args.before))
            end = math.ceil(float(row["end_sec"]) + args.after)
            cues.append((start, end, [{"time": row.get("start"),
                                      "cue": row["cue"],
                                      "text": row.get("text", "")}]))

    windows = merge_intervals(cues, args.merge_gap)
    if not windows:
        print("no transcript cues passed the confidence filters")
        return
    if args.dry_run:
        for start, end, cue_group in windows:
            print(f"{format_timestamp(start)}–{format_timestamp(end)} "
                  f"{len(cue_group)} cue(s): " + " | ".join(c["text"] for c in cue_group))
        return

    frames_dir = args.session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if any(frames_dir.glob("frame_*.png")) or any(frames_dir.glob("frame_*.jpg")):
        raise SystemExit(f"frame directory is not empty; use a new session dir: {frames_dir}")
    records = []
    extension = "jpg" if args.image_format == "jpg" else "png"
    for start, end, cue_group in windows:
        start_index = int(round(start * args.fps))
        pattern = frames_dir / f"frame_%07d.{extension}"
        command = [ffmpeg, "-hide_banner", "-loglevel", "error",
                   "-ss", format_timestamp(start), "-to", format_timestamp(end),
                   "-i", str(args.media), "-vf", f"fps={args.fps}",
                   "-start_number", str(start_index)]
        if args.image_format == "jpg":
            if not 2 <= args.jpeg_qscale <= 31:
                parser.error("--jpeg-qscale must be between 2 and 31")
            command.extend(["-q:v", str(args.jpeg_qscale)])
        command.append(str(pattern))
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise SystemExit(f"ffmpeg failed for {format_timestamp(start)}: {result.stderr[-1000:]}")
        for frame_index in range(start_index, int(math.ceil(end * args.fps))):
            name = f"frame_{frame_index:07d}.{extension}"
            if (frames_dir / name).is_file():
                seconds = frame_index / args.fps
                records.append({"image_path": name,
                                "video_timestamp": format_timestamp(seconds),
                                "platform_id": args.platform,
                                "session_id": args.session_id})

    records.sort(key=lambda row: row["video_timestamp"])
    (frames_dir / "frames.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n",
        encoding="utf-8",
    )
    (args.session_dir / "session.json").write_text(json.dumps({
        "session_id": args.session_id,
        "platform_id": args.platform,
        "sampling": "speech-cued sparse VOD frames",
        "fps": args.fps,
        "image_format": args.image_format,
        "jpeg_qscale": args.jpeg_qscale if args.image_format == "jpg" else None,
        "frames": len(records),
        "cue_windows": len(windows),
    }, indent=2), encoding="utf-8")
    print(f"extracted {len(records)} frames from {len(windows)} speech-cued windows to {frames_dir}")


if __name__ == "__main__":
    main()
