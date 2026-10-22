"""Write a frames.jsonl manifest for frames extracted by ffmpeg fps filter.

Maps frame_%05d.png (starting at 1) to HH:MM:SS video timestamps given the
segment offset. Usage:
    python scripts/manifest_ffmpeg.py data/raw/<session> --start 00:10:00 \
        --fps 1.0 --platform tradingview
"""
import argparse
import json
from pathlib import Path


def parse_ts(v):
    parts = [float(p) for p in v.split(":")]
    return 3600 * parts[0] + 60 * parts[1] + parts[2]


def fmt_ts(sec):
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", type=Path)
    ap.add_argument("--start", required=True)
    ap.add_argument("--fps", default=1.0)
    ap.add_argument("--platform")
    a = ap.parse_args()
    frames_dir = a.session_dir / "frames"
    t0 = parse_ts(a.start)
    pngs = sorted(frames_dir.glob("frame_*.png"))
    with (frames_dir / "frames.jsonl").open("w", encoding="utf-8") as f:
        for i, p in enumerate(pngs):
            t = t0 + i / float(a.fps)
            f.write(json.dumps({"image_path": p.name,
                                "video_timestamp": fmt_ts(t),
                                "platform_id": a.platform}) + "\n")
    (a.session_dir / "session.json").write_text(json.dumps({
        "platform_id": a.platform, "start": a.start, "fps": a.fps,
        "frames": len(pngs)}, indent=2), encoding="utf-8")
    print(f"manifest for {len(pngs)} frames, {a.start} onward")


if __name__ == "__main__":
    main()
