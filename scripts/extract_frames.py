"""Extract frames from a YouTube/Twitch VOD for training.

Uses yt-dlp to download the video locally, then OpenCV to sample frames at
a configurable rate. Frames are written under data/raw/<session>/ and are
gitignored. No frames are uploaded or committed.

Usage:
    python scripts/extract_frames.py --url "https://www.youtube.com/watch?v=..." \
        --platform tradingview --session tv-session-01 \
        --start 00:10:00 --end 00:20:00 --fps 2
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def parse_timestamp(value: str) -> float:
    """Parse HH:MM:SS or MM:SS to seconds."""
    parts = value.split(":")
    if len(parts) == 3:
        h, m, s = (float(p) for p in parts)
        return h * 3600 + m * 60 + s
    if len(parts) == 2:
        m, s = (float(p) for p in parts)
        return m * 60 + s
    return float(parts[0])


def download_video(url: str, output_dir: Path, quality: str) -> Path:
    """Download video with yt-dlp at reduced quality (720p is plenty for labels)."""
    video_path = output_dir / "source.mp4"
    if video_path.is_file():
        print(f"Video already downloaded: {video_path}")
        return video_path

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", quality,
        "-o", str(video_path),
        "--no-playlist",
        url,
    ]
    print(f"Downloading: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"yt-dlp stderr (last 5 lines):\n{result.stderr[-500:]}")
        raise RuntimeError(f"yt-dlp failed with exit code {result.returncode}")
    if not video_path.is_file():
        # yt-dlp sometimes appends extension
        candidates = list(output_dir.glob("source.*"))
        if candidates:
            return candidates[0]
        raise RuntimeError("yt-dlp completed but no output file found")
    return video_path


def extract_frames(video_path: Path, output_dir: Path, start: float, end: float, fps: float) -> int:
    """Sample frames from a video segment using OpenCV."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError("Install capture support first: pip install -e '.[capture]'")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0

    start_frame = int(start * video_fps)
    end_frame = int(end * video_fps) if end > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps))

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "frames.jsonl"
    manifest = manifest_path.open("a", encoding="utf-8")

    count = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    while frame_idx < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        timestamp_sec = frame_idx / video_fps
        timestamp_str = f"{int(timestamp_sec // 3600):02d}:{int((timestamp_sec % 3600) // 60):02d}:{int(timestamp_sec % 60):02d}"
        filename = f"frame_{frame_idx:08d}_{timestamp_str.replace(':', '')}.png"
        filepath = output_dir / filename
        cv2.imwrite(str(filepath), frame)

        record = {
            "image_path": str(filepath.resolve()),
            "platform_id": "",
            "session_id": "",
            "source_url": "",
            "video_timestamp": timestamp_str,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "width": frame.shape[1],
            "height": frame.shape[0],
            "elements": [],
            "event": None,
        }
        manifest.write(json.dumps(record) + "\n")
        count += 1
        frame_idx += frame_interval

    manifest.close()
    cap.release()
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="YouTube/Twitch VOD URL")
    parser.add_argument("--platform", required=True, choices=(
        "tradingview", "topstepx", "tradovate", "wealthcharts", "mt5"))
    parser.add_argument("--session", required=True, help="Session identifier for the dataset")
    parser.add_argument("--source-name", default="", help="Streamer/channel name for attribution")
    parser.add_argument("--start", default="00:00:00", help="Start timestamp (HH:MM:SS or MM:SS)")
    parser.add_argument("--end", default="", help="End timestamp (empty = end of video)")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second to sample (default 2)")
    parser.add_argument("--quality", default="best[height<=720]", help="yt-dlp format string")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="Output root for frames")
    args = parser.parse_args()

    end = parse_timestamp(args.end) if args.end else 0
    session_dir = args.data_dir / args.platform / args.session
    session_dir.mkdir(parents=True, exist_ok=True)

    print(f"Platform: {args.platform}")
    print(f"Session: {args.session}")
    print(f"Segment: {args.start} -> {args.end or 'end'} @ {args.fps} fps")
    print(f"Output: {session_dir}")
    print()

    video_path = download_video(args.url, session_dir, args.quality)
    print(f"Video: {video_path}")

    start = parse_timestamp(args.start)
    count = extract_frames(video_path, session_dir / "frames", start, end, args.fps)
    print(f"\nExtracted {count} frames to {session_dir / 'frames'}")
    print(f"Manifest: {session_dir / 'frames' / 'frames.jsonl'}")
    print("\nNext step: annotate the frames with UI element bounding boxes and trade events,")
    print("then run: python scripts/prepare_yolo.py <manifest.jsonl> --output data/derived/<platform>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
