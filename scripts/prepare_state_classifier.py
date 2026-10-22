"""Prepare a private visual-only open/not-open classification prototype.

State intervals come from owner-verified events. This intentionally makes a
temporal prototype split; it is not an independent-session accuracy claim.
The classifier sees only frames, never ASR text.

Usage:
    python scripts/prepare_state_classifier.py intervals.jsonl `
      --source channel-a=path/to/channel-a/frames `
      --source channel-a=path/to/channel-a/segment_50_87/frames `
      --source channel-b=path/to/channel-b/frames `
      --roi channel-a=0.02,0.11,0.69,0.87 `
      --output data/derived/channel-a-state-prototype
"""
import argparse
import json
from pathlib import Path
import re


SPLITS = ("train", "val", "test")
CLASSES = ("open", "not_open")


def output_split(split):
    return "external" if split == "test" else split


def parse_timestamp(value):
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def parse_mapping(value, name):
    key, sep, raw = value.partition("=")
    if not sep or not key or not raw:
        raise ValueError(f"{name} must be KEY=VALUE")
    return key, raw


def interval_for(source_id, timestamp, intervals):
    matches = [
        row for row in intervals
        if row["source_id"] == source_id
        and row["start_sec"] <= timestamp <= row["end_sec"]
    ]
    if len(matches) > 1:
        raise ValueError(f"overlapping state intervals for {source_id} at {timestamp}")
    return matches[0] if matches else None


def should_sample(timestamp, interval, stride, edge_window):
    from_start = timestamp - interval["start_sec"]
    to_end = interval["end_sec"] - timestamp
    return (
        from_start <= edge_window
        or to_end <= edge_window
        or int(round(from_start)) % stride == 0
    )


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value)


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("intervals", type=Path)
    parser.add_argument("--source", action="append", required=True,
                        help="SOURCE_ID=frames_dir; repeat for frame segments")
    parser.add_argument("--roi", action="append", default=[],
                        help="SOURCE_ID=x,y,width,height in normalized coordinates")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--open-stride", type=int, default=10,
                        help="sample one open frame per N seconds")
    parser.add_argument("--not-open-stride", type=int, default=1,
                        help="sample one flat/pending frame per N seconds")
    parser.add_argument("--edge-window", type=float, default=2.0,
                        help="sample every frame near interval boundaries")
    parser.add_argument("--img-quality", type=int, default=92)
    args = parser.parse_args()

    if args.open_stride <= 0 or args.not_open_stride <= 0:
        parser.error("sampling strides must be positive")

    import cv2

    intervals = read_jsonl(args.intervals)
    for row in intervals:
        row["start_sec"] = parse_timestamp(row["start"])
        row["end_sec"] = parse_timestamp(row["end"])
        if row.get("split") not in SPLITS or row.get("trade_state") not in ("open", "flat", "pending", "unknown"):
            raise SystemExit(f"invalid state interval: {row}")

    try:
        sources = [parse_mapping(value, "--source") for value in args.source]
        roi_specs = [parse_mapping(value, "--roi") for value in args.roi]
        rois = {key: tuple(float(v) for v in raw.split(",")) for key, raw in roi_specs}
    except ValueError as exc:
        parser.error(str(exc))
    for source_id, roi in rois.items():
        if len(roi) != 4 or not (0 <= roi[0] < 1 and 0 <= roi[1] < 1 and 0 < roi[2] <= 1 and 0 < roi[3] <= 1):
            parser.error(f"invalid normalized ROI for {source_id}")

    for split in ("train", "val", "external"):
        for class_name in CLASSES:
            (args.output / split / class_name).mkdir(parents=True, exist_ok=True)

    metadata = []
    skipped = 0
    for source_index, (source_id, raw_dir) in enumerate(sources):
        frames_dir = Path(raw_dir)
        manifest = frames_dir / "frames.jsonl"
        if not manifest.is_file():
            raise SystemExit(f"frames manifest not found: {manifest}")
        roi = rois.get(source_id, (0.0, 0.0, 1.0, 1.0))
        for frame in read_jsonl(manifest):
            timestamp = parse_timestamp(frame["video_timestamp"])
            interval = interval_for(source_id, timestamp, intervals)
            if interval is None or interval["trade_state"] == "unknown":
                continue
            state = interval["trade_state"]
            stride = args.open_stride if state == "open" else args.not_open_stride
            if not should_sample(timestamp, interval, stride, args.edge_window):
                continue
            image_path = frames_dir / Path(frame["image_path"]).name
            image = cv2.imread(str(image_path))
            if image is None:
                skipped += 1
                continue
            height, width = image.shape[:2]
            x, y, crop_w, crop_h = roi
            x1, y1 = int(x * width), int(y * height)
            x2, y2 = int((x + crop_w) * width), int((y + crop_h) * height)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                skipped += 1
                continue
            class_name = "open" if state == "open" else "not_open"
            stamp = frame["video_timestamp"].replace(":", "")
            stem = f"{safe_name(source_id)}_{source_index}_{stamp}_{Path(frame['image_path']).stem}"
            target = args.output / output_split(interval["split"]) / class_name / f"{stem}.jpg"
            cv2.imwrite(str(target), crop, [cv2.IMWRITE_JPEG_QUALITY, args.img_quality])
            metadata.append({
                "output_path": target.relative_to(args.output).as_posix(),
                "source_id": source_id,
                "image_path": str(image_path),
                "video_timestamp": frame["video_timestamp"],
                "trade_state": state,
                "class_name": class_name,
                "split": interval["split"],
                "interval_reason": interval.get("reason"),
                "roi": list(roi),
            })

    if not metadata:
        raise SystemExit("no frames matched the supplied state intervals")
    (args.output / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in metadata) + "\n",
        encoding="utf-8",
    )
    print(f"prepared {len(metadata)} visual state crops; skipped {skipped} unreadable frames")
    for split in ("train", "val", "external"):
        counts = {name: len(list((args.output / split / name).glob("*.jpg"))) for name in CLASSES}
        print(f"  {split}: {counts}")
    print("This temporal prototype split is not an independent-session generalization benchmark.")


if __name__ == "__main__":
    main()
