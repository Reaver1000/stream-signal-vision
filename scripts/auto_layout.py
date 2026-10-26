"""Locate the price axis and chart area without per-streamer coordinates.

Every trading platform draws a column of similar numeric labels beside the
candles. We find text-like blobs, cluster them into vertical columns, OCR only
the few strongest columns, and keep the one whose numbers form a consistent
linear price scale. That column is the price axis; the chart is the region
beside it.

This makes the reader layout-agnostic: TopstepX, TradingView, Tradovate and
anything similar resolve the same way.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from price_axis import calibrate, read_axis_labels


def text_components(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find small bright glyph-like blobs on any background brightness.

    A plain brightness threshold merges the whole page into one blob when the
    screenshot has pale margins, so use a top-hat (local contrast) filter.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    mask = (tophat >= 28).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 5 <= h <= 46 and 8 <= w <= 260 and w * h >= 40:
            boxes.append((x, y, w, h))
    return boxes


def cluster_columns(boxes: list[tuple[int, int, int, int]], tolerance: int = 22) -> list[dict]:
    columns: list[dict] = []
    for x, y, w, h in sorted(boxes, key=lambda b: b[0] + b[2] / 2):
        center = x + w / 2
        placed = False
        for column in columns:
            if abs(center - column["center"]) <= tolerance:
                column["boxes"].append((x, y, w, h))
                column["center"] = (column["center"] + center) / 2
                placed = True
                break
        if not placed:
            columns.append({"center": center, "boxes": [(x, y, w, h)]})
    return columns


def column_roi(column: dict, image_shape) -> tuple[float, float, float, float]:
    height, width = image_shape[:2]
    xs = [b[0] for b in column["boxes"]]
    xe = [b[0] + b[2] for b in column["boxes"]]
    ys = [b[1] for b in column["boxes"]]
    ye = [b[1] + b[3] for b in column["boxes"]]
    x0 = max(0, min(xs) - 6)
    x1 = min(width, max(xe) + 6)
    y0 = max(0, min(ys) - 10)
    y1 = min(height, max(ye) + 10)
    return (
        x0 / width,
        y0 / height,
        max(0.005, (x1 - x0) / width),
        max(0.01, (y1 - y0) / height),
    )


def find_axis(image: np.ndarray, max_candidates: int = 8) -> dict | None:
    columns = cluster_columns(text_components(image))
    ranked = sorted(columns, key=lambda c: len(c["boxes"]), reverse=True)[:max_candidates]
    best = None
    for column in ranked:
        if len(column["boxes"]) < 4:
            continue
        roi = column_roi(column, image.shape)
        labels = read_axis_labels(image, roi)
        scale = calibrate(labels)
        if not scale or scale["labels"] < 4:
            continue
        score = (scale["labels"], -scale["max_residual"])
        if best is None or score > best["score"]:
            best = {"score": score, "axis_roi": roi, "scale": scale, "boxes": len(column["boxes"])}
    if best is None:
        return None

    height, width = image.shape[:2]
    ax, ay, aw, ah = best["axis_roi"]
    axis_x0 = int(ax * width)
    axis_x1 = int((ax + aw) * width)
    axis_y0 = int(ay * height)
    axis_y1 = int((ay + ah) * height)

    if axis_x0 > width / 2:
        chart_x0 = max(0, axis_x0 - int(width * 0.75))
        chart_x1 = axis_x0
        axis_side = "right"
    else:
        chart_x0 = axis_x1
        chart_x1 = min(width, axis_x1 + int(width * 0.75))
        axis_side = "left"

    chart_roi = (
        chart_x0 / width,
        axis_y0 / height,
        max(0.05, (chart_x1 - chart_x0) / width),
        max(0.05, (axis_y1 - axis_y0) / height),
    )
    tag_roi = (
        chart_roi[0] + chart_roi[2] * 0.70,
        chart_roi[1],
        chart_roi[2] * 0.30,
        chart_roi[3],
    )
    return {
        "axis_roi": best["axis_roi"],
        "chart_roi": chart_roi,
        "tag_roi": tag_roi,
        "scale": best["scale"],
        "axis_side": axis_side,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", type=Path)
    ap.add_argument("--debug-out", type=Path, default=None)
    args = ap.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"cannot read {args.image}")
    result = find_axis(image)
    if result is None:
        print("no price axis found")
        return 1
    print(json.dumps({
        "axis_side": result["axis_side"],
        "axis_roi": [round(v, 4) for v in result["axis_roi"]],
        "chart_roi": [round(v, 4) for v in result["chart_roi"]],
        "axis_labels": result["scale"]["labels"],
        "max_residual": round(result["scale"]["max_residual"], 3),
        "price_range": [round(v, 2) for v in result["scale"]["price_range"]],
    }, indent=2))
    if args.debug_out:
        out = image.copy()
        h, w = image.shape[:2]
        ax, ay, aw, ah = result["axis_roi"]
        cv2.rectangle(out, (int(ax * w), int(ay * h)), (int((ax + aw) * w), int((ay + ah) * h)), (0, 255, 0), 2)
        cx, cy, cw, ch = result["chart_roi"]
        cv2.rectangle(out, (int(cx * w), int(cy * h)), (int((cx + cw) * w), int((cy + ch) * h)), (0, 255, 255), 2)
        args.debug_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.debug_out), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
