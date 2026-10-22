"""Detect the red/green trade box drawn on charts and extract trade-detail confluence.

A working/filled position on most charting surfaces (TradingView position
tool, TopstepX order lines, etc.) renders as a box with a green profit region
and a red loss region sharing an entry edge:

- green region ABOVE the entry edge -> long (buy)
- green region BELOW the entry edge -> short (sell)
- the red region's far edge -> usual stop-loss level
- the green region's far edge -> usual take-profit level

This is confluence, not gospel truth. The box alone can be a hypothetical
plan; only boxes that coexist with position-state evidence (open badge/panel,
spoken confirmation) become signaled trades. Levels are extracted as
normalized edge positions; price labels on the axis are saved as crops for
OCR/VLM reading.

Usage:
    python scripts/trade_box.py <frames_dir> --roi 0.0,0.08,0.78,0.82 `
        --start 01:13:00 --end 01:17:10 --save-crops <dir> --jsonl out.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


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


def parse_roi(value: str | None):
    if not value:
        return None
    parts = [float(v) for v in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height")
    return parts


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def find_trade_boxes(image, roi=None, min_area_frac=0.012, min_width_frac=0.06):
    """Return trade boxes as dicts with direction and normalized edge positions."""
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    x, y, box_w, box_h = roi or (0.0, 0.0, 1.0, 1.0)
    crop = image[int(y * height):int((y + box_h) * height),
                 int(x * width):int((x + box_w) * width)]
    ch, cw = crop.shape[:2]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green_mask = (((hue >= 35) & (hue <= 85) & (sat >= 40) & (val >= 50)) * 255).astype("uint8")
    red_mask = ((((hue <= 10) | (hue >= 165)) & (sat >= 40) & (val >= 50)) * 255).astype("uint8")

    close_kernel = np.ones((11, 11), np.uint8)
    open_kernel = np.ones((5, 5), np.uint8)
    min_area = min_area_frac * ch * cw
    min_w = min_width_frac * cw

    regions = {}
    for name, mask in (("green", green_mask), ("red", red_mask)):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        found = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            rx, ry, rw, rh = cv2.boundingRect(contour)
            if rw * rh < min_area or rw < min_w:
                continue
            if cv2.contourArea(contour) / (rw * rh) < 0.5:
                continue
            found.append((rx, ry, rw, rh))
        regions[name] = found

    y_gap_tolerance = max(8, int(0.025 * ch))
    boxes = []
    for gx, gy, gw, gh in regions["green"]:
        for rx, ry, rw, rh in regions["red"]:
            overlap = min(gx + gw, rx + rw) - max(gx, rx)
            if overlap < 0.5 * min(gw, rw):
                continue
            if gy < ry and abs((gy + gh) - ry) <= y_gap_tolerance:
                boxes.append({
                    "direction": "long",
                    "entry_y": (gy + gh) / ch,
                    "tp_y": gy / ch,
                    "sl_y": (ry + rh) / ch,
                    "bbox": [gx / cw, gy / ch, gw / cw, (gh + rh) / ch],
                    "area_frac": (gw * (gh + rh)) / (ch * cw),
                })
            elif ry < gy and abs((ry + rh) - gy) <= y_gap_tolerance:
                boxes.append({
                    "direction": "short",
                    "entry_y": (ry + rh) / ch,
                    "tp_y": (gy + gh) / ch,
                    "sl_y": ry / ch,
                    "bbox": [gx / cw, ry / ch, gw / cw, (gh + rh) / ch],
                    "area_frac": (gw * (gh + rh)) / (ch * cw),
                })
    boxes.sort(key=lambda b: -b["area_frac"])
    return boxes[:3], (x, y, box_w, box_h), (ch, cw)


def save_axis_crops(image, box, roi_geom, crop_size, out_dir: Path, stem: str):
    import cv2

    _, (rx, ry, rw, rh) = None, roi_geom
    height, width = image.shape[:2]
    x0, y0 = int(rx * width), int(ry * height)
    out_dir.mkdir(parents=True, exist_ok=True)
    axis_x1, axis_x2 = int(0.80 * width), width
    for edge in ("entry", "sl", "tp"):
        y_norm = box[f"{edge}_y"]
        cy = y0 + int(y_norm * crop_size[0])
        y1, y2 = max(0, cy - 14), min(height, cy + 14)
        strip = image[y1:y2, axis_x1:axis_x2]
        cv2.imwrite(str(out_dir / f"{stem}_{edge}.png"), strip)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames_dir", type=Path)
    parser.add_argument("--roi", type=parse_roi, default=None)
    parser.add_argument("--start", default=None, help="HH:MM:SS window start")
    parser.add_argument("--end", default=None, help="HH:MM:SS window end")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--save-crops", type=Path, default=None)
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--min-area-frac", type=float, default=0.012)
    args = parser.parse_args()

    import cv2

    manifest = read_jsonl(args.frames_dir / "frames.jsonl")
    manifest.sort(key=lambda r: parse_timestamp(r["video_timestamp"]))
    t0 = parse_timestamp(args.start) if args.start else None
    t1 = parse_timestamp(args.end) if args.end else None

    detections = []
    for index, row in enumerate(manifest):
        t = parse_timestamp(row["video_timestamp"])
        if t0 is not None and t < t0:
            continue
        if t1 is not None and t > t1:
            break
        if index % args.stride:
            continue
        image = cv2.imread(str(args.frames_dir / Path(row["image_path"]).name))
        if image is None:
            continue
        boxes, roi_geom, crop_size = find_trade_boxes(image, args.roi, args.min_area_frac)
        if boxes:
            best = boxes[0]
            detections.append({
                "timestamp": row["video_timestamp"],
                "timestamp_sec": t,
                "direction": best["direction"],
                "entry_y": round(best["entry_y"], 4),
                "sl_y": round(best["sl_y"], 4),
                "tp_y": round(best["tp_y"], 4),
                "area_frac": round(best["area_frac"], 4),
            })
            if args.save_crops:
                save_axis_crops(image, best, roi_geom, crop_size, args.save_crops,
                                Path(row["image_path"]).stem)
        elif args.save_crops:
            pass

    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.jsonl.write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in detections) + "\n",
            encoding="utf-8",
        )

    if not detections:
        print("[trade_box] no trade boxes detected in window")
        return 0

    directions = [d["direction"] for d in detections]
    majority = statistics.mode(directions)
    coverage = len(detections)
    print(f"[trade_box] detections={coverage} majority={majority} "
          f"long_votes={directions.count('long')} short_votes={directions.count('short')}")
    first, last = detections[0], detections[-1]
    print(f"  span {first['timestamp']}..{last['timestamp']} "
          f"entry_y median={statistics.median(d['entry_y'] for d in detections):.3f} "
          f"sl_y median={statistics.median(d['sl_y'] for d in detections):.3f} "
          f"tp_y median={statistics.median(d['tp_y'] for d in detections):.3f}")

    edges = [(d["timestamp"], d["sl_y"], d["tp_y"]) for d in detections]
    sl_moves = [(a[0], b[0]) for a, b in zip(edges, edges[1:]) if abs(b[1] - a[1]) > 0.004]
    tp_moves = [(a[0], b[0]) for a, b in zip(edges, edges[1:]) if abs(b[2] - a[2]) > 0.004]
    if sl_moves:
        print(f"  SL moved {len(sl_moves)}x, first at {sl_moves[0][1]}")
    if tp_moves:
        print(f"  TP moved {len(tp_moves)}x, first at {tp_moves[0][1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
