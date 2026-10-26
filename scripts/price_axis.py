"""Read the chart price axis and map any pixel row to a price.

Order tags are tiny and OCR-unreliable; the axis grid labels are larger, evenly
spaced and drawn white-on-black, so we calibrate the y->price scale from them
and then interpolate the price of every detected order line from its row.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

_OCR = None


def get_ocr():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    return _OCR


def parse_price(text: str) -> float | None:
    cleaned = text.replace(" ", "").replace("O", "0").replace("o", "0")
    cleaned = re.sub(r"[^0-9,.\-]", "", cleaned)
    if not cleaned:
        return None
    if cleaned.count(",") == 1 and cleaned.count(".") <= 1:
        cleaned = cleaned.replace(",", "")
    elif cleaned.count(",") >= 1 and cleaned.count(".") == 1:
        cleaned = cleaned.replace(",", "")
    elif cleaned.count(".") > 1 or cleaned.count(",") > 1:
        parts = re.split(r"[,.]", cleaned)
        if len(parts[-1]) == 3:
            cleaned = "".join(parts)
        else:
            cleaned = "".join(parts[:-1]) + "." + parts[-1]
    try:
        return float(cleaned)
    except ValueError:
        return None


def white_text_mask(strip: np.ndarray, min_value: int = 150, max_saturation: int = 70) -> np.ndarray:
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    value = (hsv[:, :, 2] >= min_value).astype(np.uint8) * 255
    low_sat = (hsv[:, :, 1] <= max_saturation).astype(np.uint8) * 255
    return cv2.bitwise_and(value, low_sat)


def read_axis_labels(image: np.ndarray, axis_roi, upscale: int = 3) -> list[dict]:
    """OCR the raw axis strip, then keep only labels drawn on a dark background.

    Masking the glyphs first destroys the thin anti-aliased digits, so we OCR
    the strip as-is and use the mean background saturation around each detected
    box to drop coloured order tags (high saturation) from the grid labels.
    """
    h, w = image.shape[:2]
    x, y, rw, rh = axis_roi
    x0, y0 = int(x * w), int(y * h)
    x1, y1 = int((x + rw) * w), int((y + rh) * h)
    strip = image[y0:y1, x0:x1]
    if strip.size == 0:
        return []
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    padding = np.zeros((strip.shape[0], 12, 3), dtype=np.uint8)
    prepared = np.hstack([padding, strip, padding])
    if upscale > 1:
        prepared = cv2.resize(prepared, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    result, _ = get_ocr()(prepared)
    labels = []
    for box, text, conf in (result or []):
        price = parse_price(text)
        if price is None or not (100 <= price <= 1_000_000):
            continue
        xs = [(point[0] / upscale) - 12 for point in box]
        ys = [point[1] / upscale for point in box]
        bx0 = max(0, int(min(xs)) - 1)
        bx1 = min(strip.shape[1] - 1, int(max(xs)) + 1)
        by0 = max(0, int(min(ys)) - 1)
        by1 = min(strip.shape[0] - 1, int(max(ys)) + 1)
        if bx1 <= bx0 or by1 <= by0:
            continue
        patch = hsv[by0:by1 + 1, bx0:bx1 + 1]
        saturation = float(patch[:, :, 1].mean()) if patch.size else 0.0
        labels.append({
            "price": price,
            "y": y0 + float(sum(ys) / len(ys)),
            "text": text,
            "confidence": float(conf),
            "saturation": round(saturation, 1),
        })
    return labels


def snap_grid(labels: list[dict], tolerance_frac: float = 0.45) -> list[dict]:
    """Axis grid labels are round numbers; snap each to its nearest clean step."""
    if len(labels) < 2:
        return labels
    prices = sorted(label["price"] for label in labels)
    diffs = [b - a for a, b in zip(prices, prices[1:]) if b > a]
    if not diffs:
        return labels
    diffs.sort()
    step = diffs[len(diffs) // 2]
    if step <= 0:
        return labels
    snapped = []
    for label in labels:
        nearest = round(label["price"] / step) * step
        if abs(nearest - label["price"]) <= step * tolerance_frac:
            label = {**label, "price": round(nearest, 4)}
        snapped.append(label)
    return snapped


def fit_scale(labels: list[dict]) -> dict | None:
    if len(labels) < 2:
        return None
    ys = np.array([label["y"] for label in labels], dtype=float)
    ps = np.array([label["price"] for label in labels], dtype=float)
    if np.ptp(ys) < 20:
        return None
    slope, intercept = np.polyfit(ys, ps, 1)
    if slope >= 0:
        return None
    predicted = slope * ys + intercept
    residual = float(np.max(np.abs(predicted - ps)))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "labels": len(labels),
        "max_residual": residual,
        "y_min": float(ys.min()),
        "y_max": float(ys.max()),
        "price_range": [float(ps.min()), float(ps.max())],
    }


def calibrate(labels: list[dict], max_saturation: float = 90.0,
              max_residual: float = 3.0) -> dict | None:
    """Keep grid labels and fit the strongest consistent y->price line.

    Order tags sit on saturated backgrounds and either fail the saturation cut
    or fall off the grid line, so a consensus fit removes them without having
    to identify them individually.
    """
    candidates = [
        label for label in labels
        if float(label.get("saturation", 0)) <= max_saturation
        and float(label.get("confidence", 0)) >= 0.5
    ]
    if len(candidates) < 2:
        return None
    best_inliers: list[dict] = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            dy = b["y"] - a["y"]
            if abs(dy) < 40:
                continue
            slope = (b["price"] - a["price"]) / dy
            if slope >= 0:
                continue
            intercept = a["price"] - slope * a["y"]
            inliers = [
                c for c in candidates
                if abs(slope * c["y"] + intercept - c["price"]) <= max_residual
            ]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
    if len(best_inliers) < 2:
        return None
    ys = np.array([label["y"] for label in best_inliers], dtype=float)
    ps = np.array([label["price"] for label in best_inliers], dtype=float)
    slope, intercept = np.polyfit(ys, ps, 1)
    predicted = slope * ys + intercept
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "labels": len(best_inliers),
        "max_residual": float(np.max(np.abs(predicted - ps))),
        "y_min": float(ys.min()),
        "y_max": float(ys.max()),
        "price_range": [float(ps.min()), float(ps.max())],
        "axis_labels": [label["text"] for label in best_inliers],
    }


def price_at(scale: dict, y: float) -> float:
    return scale["slope"] * y + scale["intercept"]


def min_tick(scale: dict) -> float:
    return abs(scale["slope"]) * 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", type=Path)
    ap.add_argument("--axis-roi", required=True, help="normalized x,y,w,h of the price-axis strip")
    ap.add_argument("--extra-y", default=None,
                    help="comma-separated rows to price with the fitted scale")
    ap.add_argument("--upscale", type=int, default=3)
    args = ap.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"cannot read {args.image}")
    roi = tuple(float(v) for v in args.axis_roi.split(","))
    labels = snap_grid(read_axis_labels(image, roi, args.upscale))
    scale = fit_scale(labels)
    for label in labels:
        print(f"label y={label['y']:.0f} price={label['price']:.2f} conf={label['confidence']:.2f} text={label['text']!r}")
    if scale is None:
        print("scale: not enough consistent labels")
        return 1
    print(f"scale slope={scale['slope']:.4f} intercept={scale['intercept']:.2f} "
          f"labels={scale['labels']} max_residual={scale['max_residual']:.2f}")
    if args.extra_y:
        for raw in args.extra_y.split(","):
            y = float(raw)
            print(f"row {y:.0f} -> {price_at(scale, y):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
