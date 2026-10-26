"""Detect horizontal order/position markers on trading charts.

The rule that separates a position from an indicator: a position/order has a
filled box (or tag) attached to a horizontal line that spans much of the chart.
Indicators draw lines but not that attached box.

Output per marker: colour, y row, line extent, whether a box is attached, and
the box rectangle. Prices are resolved separately from the y-axis scale.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Strict ranges: used to judge whether a line is solid or dotted, so candle
# wicks and faint drawings must not count as line pixels.
COLOR_RANGES = {
    "green": [((35, 120, 90), (90, 255, 255))],
    "red": [((0, 130, 130), (8, 255, 255)), ((172, 130, 130), (179, 255, 255))],
    "blue": [((100, 130, 100), (130, 255, 255))],
    "amber": [((15, 120, 120), (32, 255, 255))],
}

# Loose ranges: used to find order pills, which some platforms draw in muted
# maroon/navy rather than saturated colours.
LOOSE_RANGES = {
    "green": [((32, 80, 50), (95, 255, 255))],
    "red": [((0, 70, 50), (14, 255, 255)), ((166, 70, 50), (179, 255, 255))],
    "blue": [((90, 60, 50), (140, 255, 255))],
    "amber": [((12, 80, 80), (32, 255, 255))],
}


def color_mask_range(hsv: np.ndarray, ranges: dict, color: str) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges.get(color, []):
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    return mask


def crop_roi(image: np.ndarray, roi) -> tuple[np.ndarray, tuple[int, int]]:
    if roi is None:
        return image, (0, 0)
    h, w = image.shape[:2]
    x, y, rw, rh = roi
    x0, y0 = int(x * w), int(y * h)
    x1, y1 = int((x + rw) * w), int((y + rh) * h)
    return image[y0:y1, x0:x1], (x0, y0)


def color_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    return mask


def suppress_pure_yellow(mask: np.ndarray, hsv: np.ndarray) -> np.ndarray:
    yellow = cv2.inRange(hsv, np.array((25, 200, 200), np.uint8), np.array((35, 255, 255), np.uint8))
    return cv2.subtract(mask, yellow)


def find_line_rows(mask: np.ndarray, min_line_frac: float, width: int) -> list[dict]:
    bridged = cv2.dilate(mask, np.ones((1, 21), np.uint8))
    row_counts = (bridged > 0).sum(axis=1)
    threshold = max(20, int(width * min_line_frac))
    rows = np.where(row_counts >= threshold)[0]
    if rows.size == 0:
        return []
    groups: list[list[int]] = []
    for row in rows:
        if groups and row - groups[-1][-1] <= 3:
            groups[-1].append(int(row))
        else:
            groups.append([int(row)])
    lines = []
    for group in groups:
        if group[0] <= 3 or group[-1] >= mask.shape[0] - 3:
            continue
        y = int(round(sum(group) / len(group)))
        band = bridged[max(0, group[0] - 1): group[-1] + 2, :]
        cols = np.where((band > 0).any(axis=0))[0]
        if cols.size == 0:
            continue
        lines.append({
            "y": y,
            "y0": group[0],
            "y1": group[-1],
            "x0": int(cols.min()),
            "x1": int(cols.max()),
            "width_frac": round(float(cols.size) / width, 3),
        })
    return lines


def find_attached_box(mask: np.ndarray, line: dict, width: int, height: int) -> dict | None:
    """A position/order box is a solid block at the right end of the line."""
    right_zone_start = max(line["x1"] - int(width * 0.16), 0)
    zone = mask[max(0, line["y"] - 14): min(height, line["y"] + 15), right_zone_start: width]
    if zone.size == 0:
        return None
    col_counts = (zone > 0).sum(axis=0)
    solid_cols = col_counts >= max(6, int(zone.shape[0] * 0.35))
    if solid_cols.sum() < 24:
        return None
    xs = np.where(solid_cols)[0]
    box_w = int(xs.max() - xs.min() + 1)
    if box_w < 24 or box_w > int(width * 0.18):
        return None
    row_counts = (zone[:, xs.min(): xs.max() + 1] > 0).sum(axis=1)
    solid_rows = np.where(row_counts >= max(12, int(box_w * 0.4)))[0]
    if solid_rows.size < 10:
        return None
    box_x0 = right_zone_start + int(xs.min())
    box_x1 = right_zone_start + int(xs.max())
    box_y0 = max(0, line["y"] - 14) + int(solid_rows.min())
    box_y1 = max(0, line["y"] - 14) + int(solid_rows.max())
    return {
        "x0": box_x0,
        "y0": box_y0,
        "x1": box_x1,
        "y1": box_y1,
        "w": box_x1 - box_x0 + 1,
        "h": box_y1 - box_y0 + 1,
    }


def detect_markers(image: np.ndarray, roi=None, min_line_frac: float = 0.28,
                   ignore_yellow: bool = False) -> list[dict]:
    chart, offset = crop_roi(image, roi)
    height, width = chart.shape[:2]
    hsv = cv2.cvtColor(chart, cv2.COLOR_BGR2HSV)
    markers: list[dict] = []
    for color in COLOR_RANGES:
        mask = color_mask(hsv, color)
        if ignore_yellow:
            mask = suppress_pure_yellow(mask, hsv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 9), np.uint8))
        for line in find_line_rows(mask, min_line_frac, width):
            if line["width_frac"] < min_line_frac:
                continue
            box = find_attached_box(mask, line, width, height)
            markers.append({
                "color": color,
                "y": line["y"] + offset[1],
                "x0": line["x0"] + offset[0],
                "x1": line["x1"] + offset[0],
                "width_frac": line["width_frac"],
                "box": None if box is None else {
                    "x0": box["x0"] + offset[0], "y0": box["y0"] + offset[1],
                    "x1": box["x1"] + offset[0], "y1": box["y1"] + offset[1],
                    "w": box["w"], "h": box["h"],
                },
            })
    markers.sort(key=lambda m: (m["y"], m["color"]))
    return markers


def annotate(image: np.ndarray, markers: list[dict]) -> np.ndarray:
    out = image.copy()
    palette = {"green": (0, 255, 0), "red": (0, 0, 255), "blue": (255, 128, 0), "amber": (0, 200, 255)}
    for m in markers:
        cv2.line(out, (m["x0"], m["y"]), (m["x1"], m["y"]), palette.get(m["color"], (255, 255, 255)), 2)
        if m["box"]:
            b = m["box"]
            cv2.rectangle(out, (b["x0"], b["y0"]), (b["x1"], b["y1"]), (0, 255, 255), 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", type=Path)
    ap.add_argument("--roi", default=None, help="normalized x,y,w,h chart crop")
    ap.add_argument("--min-line-frac", type=float, default=0.28)
    ap.add_argument("--ignore-yellow", action="store_true",
                    help="mask pure-yellow annotation overlays (for reviewed fixtures)")
    ap.add_argument("--debug-out", type=Path, default=None)
    args = ap.parse_args()
    roi = tuple(float(v) for v in args.roi.split(",")) if args.roi else None
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"cannot read {args.image}")
    markers = detect_markers(image, roi, args.min_line_frac, args.ignore_yellow)
    print(json.dumps(markers, indent=2))
    if args.debug_out:
        args.debug_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.debug_out), annotate(image, markers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
