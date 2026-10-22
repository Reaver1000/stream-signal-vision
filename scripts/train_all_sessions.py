#!/usr/bin/env python3
"""Train visual state classifier on ALL channel-a sessions."""

import json
from pathlib import Path

ROOT = Path("C:/Users/david/Documents/Default Project/stream-signal-vision")

# Source directories for each channel-a session
sources = [
    ("channel-a-tradingview-vod-01", ROOT / "data" / "raw" / "channel-a-tradingview-vod-01" / "frames"),
    ("channel-a-tradingview-sess01", ROOT / "data" / "raw" / "channel-a-tradingview-sess01" / "frames"),
    ("channel-a-tradingview-sess02", ROOT / "data" / "raw" / "channel-a-tradingview-sess02" / "frames"),
    ("channel-a-tradingview-sess03", ROOT / "data" / "raw" / "channel-a-tradingview-sess03" / "frames"),
    ("channel-a-tradingview-sess04", ROOT / "data" / "raw" / "channel-a-tradingview-sess04" / "frames"),
    ("channel-a-tradingview-sess05", ROOT / "data" / "raw" / "channel-a-tradingview-sess05" / "frames"),
    ("channel-a-tradingview-sess06", ROOT / "data" / "raw" / "channel-a-tradingview-sess06" / "frames"),
    ("channel-a-tradingview-sess07", ROOT / "data" / "raw" / "channel-a-tradingview-sess07" / "frames"),
    ("channel-a-tradingview-sess08", ROOT / "data" / "raw" / "channel-a-tradingview-sess08" / "frames"),
    ("channel-a-tradingview-sess09", ROOT / "data" / "raw" / "channel-a-tradingview-sess09" / "frames"),
    ("channel-a-tradingview-sess10", ROOT / "data" / "raw" / "channel-a-tradingview-sess10" / "frames"),
    ("channel-a-tradingview-sess11", ROOT / "data" / "raw" / "channel-a-tradingview-sess11" / "frames"),
    ("channel-a-tradingview-sess12", ROOT / "data" / "raw" / "channel-a-tradingview-sess12" / "frames"),
]

intervals_path = ROOT / "data" / "raw" / "channel-a-tradingview-vod-01" / "state_intervals.jsonl"
output_dir = ROOT / "data" / "derived" / "channel-a-state-prototype"


def parse_timestamp(value):
    parts = [float(p) for p in value.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def main():
    import cv2
    from ultralytics import YOLO

    # Read intervals
    print("Reading state intervals...")
    with open(intervals_path) as f:
        intervals = []
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row["start_sec"] = parse_timestamp(row["start"])
            row["end_sec"] = parse_timestamp(row["end"])
            intervals.append(row)
    print(f"Read {len(intervals)} state intervals")

    # Collect all frames from all sessions
    frame_records = []
    for source_id, frames_dir in sources:
        if not frames_dir.is_dir():
            print(f"WARNING: {frames_dir} does not exist, skipping")
            continue
        manifest = frames_dir / "frames.jsonl"
        if not manifest.is_file():
            print(f"WARNING: No frames manifest for {source_id}, skipping")
            continue
        with manifest.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                timestamp = parse_timestamp(row["video_timestamp"])
                frame_records.append({
                    "image_path": str(frames_dir / Path(row["image_path"]).name),
                    "timestamp": timestamp,
                    "session_id": source_id,
                })

    # Assign each frame to an interval
    frame_rows_with_state = []
    for row in frame_records:
        ts = row["timestamp"]
        assigned = False
        for interval in intervals:
            if interval["start_sec"] <= ts <= interval["end_sec"]:
                row["trade_state"] = interval["trade_state"]
                row["split"] = interval["split"]
                assigned = True
                break
        if not assigned:
            row["trade_state"] = "unknown"
        frame_rows_with_state.append(row)

    # Count
    train_open = sum(1 for r in frame_rows_with_state if r["trade_state"] == "open" and r.get("split") == "train")
    train_not_open = sum(1 for r in frame_rows_with_state if r["trade_state"] == "not_open" and r.get("split") == "train")
    val_open = sum(1 for r in frame_rows_with_state if r["trade_state"] == "open" and r.get("split") == "val")
    val_not_open = sum(1 for r in frame_rows_with_state if r["trade_state"] == "not_open" and r.get("split") == "val")

    print(f"\nDataset summary:")
    print(f"  Train: {train_open} open, {train_not_open} not_open")
    print(f"  Val:   {val_open} open, {val_not_open} not_open")

    # Copy images to dataset directory
    (output_dir / "train" / "open").mkdir(parents=True, exist_ok=True)
    (output_dir / "train" / "not_open").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "open").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "not_open").mkdir(parents=True, exist_ok=True)

    import shutil
    count = {"train_open": 0, "train_not_open": 0, "val_open": 0, "val_not_open": 0}
    for row in frame_rows_with_state:
        if row["trade_state"] not in ("open", "flat", "pending"):
            continue
        src = Path(row["image_path"])
        if row["split"] == "train":
            if row["trade_state"] == "open":
                dst = output_dir / "train" / "open" / f"{row['session_id']}_{src.name}"
                count["train_open"] += 1
            else:
                dst = output_dir / "train" / "not_open" / f"{row['session_id']}_{src.name}"
                count["train_not_open"] += 1
        elif row["split"] == "val":
            if row["trade_state"] == "open":
                dst = output_dir / "val" / "open" / f"{row['session_id']}_{src.name}"
                count["val_open"] += 1
            else:
                dst = output_dir / "val" / "not_open" / f"{row['session_id']}_{src.name}"
                count["val_not_open"] += 1
        else:
            continue
        shutil.copy2(str(src), str(dst))

    print(f"\nDataset prepared:")
    print(f"  Train: {count['train_open']} open, {count['train_not_open']} not_open")
    print(f"  Val:   {count['val_open']} open, {count['val_not_open']} not_open")

    # Train the model
    print("\nTraining visual state classifier...")
    from ultralytics import YOLO
    model = YOLO("yolo26n-cls.pt")
    model.train(
        data=str(output_dir),
        epochs=20,
        imgsz=512,
        batch=16,
        device="cpu",
        project=str(ROOT / "runs" / "classify" / "runs" / "vision"),
        name="channel-a-open-not-open-v17",
        seed=42,
        deterministic=True,
    )

    print("\nTraining complete!")
    best = Path("runs/classify/runs/vision/channel-a-open-not-open-v17/weights/best.pt")
    if best.is_file():
        print(f"Best checkpoint: {best}")
    else:
        print("WARNING: best.pt not found at expected path")


if __name__ == "__main__":
    main()