"""Read a trading chart into a structured trade state.

Geometry gives the reliable facts:
  * a solid coloured line with an attached pill  -> entry
  * dotted lines with an attached pill           -> stop / target
  * the axis grid labels calibrate row -> price
OCR only fills in the small pill text (quantity / P&L) and the header.

Direction is derived from which side the stop sits on:
  stop above entry -> short, stop below entry -> long.
For a short the take-profit is a buy limit below; for a long it is a sell
limit above. That matches how these platforms place brackets.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

from chart_lines import COLOR_RANGES, LOOSE_RANGES, color_mask, color_mask_range, crop_roi, suppress_pure_yellow
from price_axis import calibrate, get_ocr, parse_price, price_at, read_axis_labels

_OCR = None


def ocr():
    global _OCR
    if _OCR is None:
        _OCR = get_ocr()
    return _OCR


@dataclass
class Marker:
    color: str
    y: int
    x0: int
    x1: int
    pill: tuple[int, int, int, int] | None = None
    price: float | None = None
    qty: int | None = None
    pnl: float | None = None
    tag_color: str | None = None
    solid: bool = True


@dataclass
class TradeState:
    in_trade: bool = False
    instrument: str | None = None
    direction: str | None = None
    quantity: int | None = None
    entry: float | None = None
    stop: float | None = None
    targets: list[float] = field(default_factory=list)
    markers: list[dict] = field(default_factory=list)
    scale: dict | None = None
    confidence: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def find_pill(mask: np.ndarray, y: int, chart_width: int, axis_width: int) -> tuple[int, int, int, int] | None:
    """Largest solid coloured block in the band, left of the price axis."""
    limit = max(0, chart_width - axis_width)
    y0, y1 = max(0, y - 12), min(mask.shape[0], y + 13)
    band = mask[y0:y1, :limit]
    if band.size == 0:
        return None
    col_hits = (band > 0).sum(axis=0)
    threshold = max(6, int((y1 - y0) * 0.35))
    solid = (col_hits >= threshold).astype(np.uint8)
    solid = cv2.dilate(solid.reshape(1, -1), np.ones((1, 27), np.uint8)).ravel() > 0
    runs = [(a, b) for a, b in _runs(solid) if (b - a + 1) >= 60]
    if not runs:
        return None
    best = max(runs, key=lambda r: r[1] - r[0])
    x0, x1 = best
    rows = np.where((band[:, x0:x1 + 1] > 0).any(axis=1))[0]
    if rows.size == 0:
        return None
    return (x0, y0 + int(rows.min()), x1, y0 + int(rows.max()))


def tag_color_at(image: np.ndarray, y: int, tag_rois) -> str | None:
    """Sample several candidate tag regions and keep the most saturated colour."""
    best, best_count = None, 0
    for roi in tag_rois:
        if not roi:
            continue
        h, w = image.shape[:2]
        x, ry, rw, rh = roi
        x0, y0 = int(x * w), int(ry * h)
        x1, y1 = int((x + rw) * w), int((ry + rh) * h)
        strip = image[max(y0, y - 10): min(y1, y + 11), x0:x1]
        if strip.size == 0:
            continue
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        for color in ("red", "green", "amber", "blue"):
            count = int((color_mask(hsv, color) > 0).sum())
            if count > best_count:
                best, best_count = color, count
    return best if best_count >= 40 else None


def dottedness(mask: np.ndarray, y: int, x0: int, x1: int) -> float:
    """Fraction of the span that is blank: solid lines ~0, dotted lines higher."""
    band = mask[max(0, y - 3): y + 4, x0:x1]
    if band.size == 0:
        return 0.0
    columns = (band > 0).any(axis=0).astype(np.uint8)
    columns = cv2.dilate(columns.reshape(1, -1), np.ones((1, 5), np.uint8)).ravel() > 0
    return 1.0 - float(columns.mean())


def ocr_pill(image: np.ndarray, pill: tuple[int, int, int, int]) -> str:
    x0, y0, x1, y1 = pill
    pad = 2
    crop = image[max(0, y0 - pad): y1 + pad, max(0, x0 - pad): x1 + pad]
    if crop.size == 0:
        return ""
    big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    result, _ = ocr()(big)
    return " ".join(text for _, text, _ in (result or []))


def parse_pill_text(text: str) -> tuple[int | None, float | None]:
    qty = None
    pnl = None
    for token in re.findall(r"[-+]?\d[\d,]*\.?\d*", text):
        cleaned = token.replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        if "." in token:
            if pnl is None or abs(value) > abs(pnl):
                pnl = value
        else:
            if 0 < value < 10000 and (qty is None or abs(value) < qty + 1):
                qty = int(value)
    return qty, pnl


def read_trade(image: np.ndarray, chart_roi, axis_roi, ignore_yellow: bool = False,
               min_line_frac: float = 0.25, tag_roi=None, scale: dict | None = None) -> TradeState:
    state = TradeState()
    chart, (ox, oy) = crop_roi(image, chart_roi)
    height, width = chart.shape[:2]
    hsv = cv2.cvtColor(chart, cv2.COLOR_BGR2HSV)
    axis_width = max(30, int(width * 0.045))
    if tag_roi is None:
        tag_roi = (chart_roi[0] + chart_roi[2] * 0.80, chart_roi[1], chart_roi[2] * 0.20, chart_roi[3])

    if scale is None:
        labels = read_axis_labels(image, axis_roi)
        scale = calibrate(labels)
    if scale:
        state.scale = scale
    else:
        state.notes.append("axis calibration failed")

    markers: list[Marker] = []
    tag_rois = []
    if tag_roi is not None:
        tag_rois.append(tag_roi)
    tag_rois.append(axis_roi)
    for color in COLOR_RANGES:
        mask = color_mask(hsv, color)
        if ignore_yellow:
            mask = suppress_pure_yellow(mask, hsv)
        bridged = cv2.dilate(mask, np.ones((1, 41), np.uint8))
        row_counts = (bridged > 0).sum(axis=1)
        threshold = max(20, int(width * 0.18))
        rows = np.where(row_counts >= threshold)[0]
        groups: list[list[int]] = []
        for row in rows:
            if groups and row - groups[-1][-1] <= 3:
                groups[-1].append(int(row))
            else:
                groups.append([int(row)])
        for group in groups:
            if group[0] <= 3 or group[-1] >= height - 3:
                continue
            if group[0] < height * 0.06 or group[-1] > height * 0.94:
                continue
            y = int(round(sum(group) / len(group)))
            pills = find_pill(mask, y, width, axis_width)
            if pills is None:
                continue
            marker = Marker(color=color, y=y + oy, x0=pills[0] + ox, x1=pills[2] + ox,
                            pill=(pills[0] + ox, pills[1] + oy, pills[2] + ox, pills[3] + oy))
            gap = dottedness(mask, y, pills[0], width - axis_width)
            marker.solid = True if color == "blue" else gap < 0.45
            if scale:
                if not (scale["y_min"] - 6 <= marker.y <= scale["y_max"] + 6):
                    continue
                marker.price = round(price_at(scale, marker.y), 2)
            text = ocr_pill(image, marker.pill)
            marker.qty, marker.pnl = parse_pill_text(text)
            marker.tag_color = tag_color_at(image, marker.y, tag_rois)
            markers.append(marker)

    markers.sort(key=lambda m: m.y)
    state.markers = [asdict(m) for m in markers]
    _assemble(state, markers)
    return state


def _assemble(state: TradeState, markers: list[Marker]) -> None:
    solid = [m for m in markers if m.solid and m.price]
    dotted = [m for m in markers if not m.solid and m.price]
    if not solid:
        state.in_trade = False
        return
    entry_marker = max(solid, key=lambda m: m.price or 0)
    entry = entry_marker.price
    state.entry = entry
    state.quantity = entry_marker.qty
    stop = None
    targets: list[float] = []
    for marker in dotted:
        if marker.price is None:
            continue
        if marker.tag_color == "red" or (stop is None and marker.price > entry):
            stop = marker.price
        else:
            targets.append(marker.price)
    if stop is None and dotted:
        stop = max((m.price for m in dotted if m.price), default=None)
    state.stop = stop
    state.targets = sorted(targets, reverse=True)
    if entry and stop:
        state.direction = "short" if stop > entry else "long"
    state.in_trade = bool(entry and (stop or targets))
    state.confidence = {
        "entry_from_axis": bool(entry),
        "stop_present": stop is not None,
        "target_count": len(state.targets),
        "qty_read": state.quantity is not None,
        "pill_count": len(markers),
    }
    if state.quantity is None:
        state.notes.append("quantity not read from pill")
    if stop is None:
        state.notes.append("no stop marker")
    if not state.targets:
        state.notes.append("no target marker")


_PILL_TEXT_CACHE: dict[tuple, str] = {}


def ocr_pill_text(image: np.ndarray, box: tuple[int, int, int, int]) -> str:
    key = box
    if key in _PILL_TEXT_CACHE:
        return _PILL_TEXT_CACHE[key]
    x0, y0, x1, y1 = box
    pad = 7
    crop = image[max(0, y0 - pad): y1 + pad, max(0, x0 - pad): x1 + pad]
    text = ""
    if crop.size:
        big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        try:
            result, _ = ocr()(big)
            text = " ".join(t for _, t, _ in (result or []))
        except Exception:
            text = ""
    _PILL_TEXT_CACHE[key] = text
    if len(_PILL_TEXT_CACHE) > 400:
        _PILL_TEXT_CACHE.clear()
    return text


def classify_tag_text(text: str) -> dict:
    low = (text or "").lower()
    low = re.sub(r"\s*([.,])\s*", r"\1", low)
    kind = None
    if re.search(r"imit|lmit|limt", low):
        kind = "target"
    elif re.search(r"st[o0]p", low):
        kind = "stop"
    side = None
    if re.search(r"se[l1]l|sell", low):
        side = "long"      # a sell bracket closes a long
    elif "buy" in low:
        side = "short"     # a buy bracket closes a short
    qty = None
    for token in re.findall(r"(?<![\d.])(\d{1,3})(?![\d.])", low):
        value = int(token)
        if 0 < value <= 500:
            qty = value if qty is None else min(qty, value)
    return {"kind": kind, "side": side, "qty": qty}


def find_chart_pills(image: np.ndarray, layout: dict, ocr_text: bool = False) -> list[dict]:
    """Find order pills in the chart area on the correct side of the axis."""
    h, w = image.shape[:2]
    ax, ay, aw, ah = layout["axis_roi"]
    axis_x0 = int(ax * w)
    axis_x1 = int((ax + aw) * w)
    scale = layout["scale"]
    y_low = scale["y_min"] - 10
    y_high = scale["y_max"] + 10
    left_axis = layout["axis_side"] == "left"
    span_x0 = axis_x1 + 2 if left_axis else int(w * 0.03)
    span_x1 = int(w * 0.97) if left_axis else axis_x0 - 2

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pills = []
    for color in ("red", "green", "blue", "amber"):
        mask = color_mask_range(hsv, LOOSE_RANGES, color)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 13), np.uint8))
        strict = color_mask(hsv, color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, pw, ph = cv2.boundingRect(contour)
            if not (40 <= pw <= (340 if ocr_text else 260) and 10 <= ph <= 30):
                continue
            if pw / max(1, ph) < 2.2:
                continue
            if left_axis:
                if x < span_x0:
                    continue
            else:
                if x + pw > span_x1:
                    continue
            if not (y_low <= y + ph / 2 <= y_high):
                continue
            fill = float((mask[y:y + ph, x:x + pw] > 0).mean())
            if fill < (0.30 if ocr_text else 0.55):
                continue
            center_y = y + ph / 2
            gap = dottedness(strict, int(round(center_y)), span_x0, span_x1)
            entry = {
                "color": color,
                "x0": x, "y0": y, "x1": x + pw, "y1": y + ph,
                "y": center_y,
                "price": round(price_at(scale, center_y), 2),
                "fill": round(fill, 2),
                "solid": gap < 0.45,
            }
            if ocr_text:
                text = ocr_pill_text(image, (x, y, x + pw, y + ph))
                entry["text"] = text
                entry.update(classify_tag_text(text))
            pills.append(entry)
    pills.sort(key=lambda p: (-p["price"], p["color"]))
    return pills


def assemble_from_pills(state: TradeState, pills: list[dict]) -> None:
    state.markers = pills
    text_stops = [p for p in pills if p.get("kind") == "stop"]
    text_targets = [p for p in pills if p.get("kind") == "target"]
    side_votes = [p.get("side") for p in pills if p.get("side")]
    blues = [p for p in pills if p["color"] == "blue"]
    stop_pills = [p for p in pills if p["color"] == "red" and not p["solid"]]
    target_pills = [p for p in pills if p["color"] == "green" and not p["solid"]]
    entry_pills = [p for p in pills if p["solid"] and p["color"] in ("green", "blue")]
    if not entry_pills and blues:
        entry_pills = blues
    if not (text_stops or text_targets or stop_pills or target_pills or entry_pills):
        state.in_trade = False
        state.notes.append("no order pills found")
        return

    entry = None
    if entry_pills:
        entry = max(entry_pills, key=lambda p: p["fill"])["price"]
    elif blues:
        entry = blues[0]["price"]

    stop = max(p["price"] for p in text_stops) if text_stops else None
    targets = sorted((p["price"] for p in text_targets), reverse=True)
    if stop is None and stop_pills:
        stop = max(p["price"] for p in stop_pills)
    if not targets and target_pills:
        targets = sorted((p["price"] for p in target_pills), reverse=True)

    quantities = [p["qty"] for p in pills if p.get("qty")]
    if quantities:
        state.quantity = min(quantities)

    if side_votes:
        state.direction = max(set(side_votes), key=side_votes.count)
    elif stop is not None and targets:
        state.direction = "long" if stop < max(targets) else "short"
    elif stop is not None and entry is not None:
        state.direction = "short" if stop > entry else "long"
    elif targets and entry is not None:
        state.direction = "long" if min(targets) > entry else "short"

    state.entry = entry
    state.stop = stop
    state.targets = targets
    state.in_trade = bool(stop is not None or targets)
    state.confidence = {
        "pills": len(pills),
        "entry": entry is not None,
        "stop": stop is not None,
        "targets": len(targets),
    }
    if entry is None:
        state.notes.append("no entry pill/line found")


def read_with_layout(image: np.ndarray, layout: dict, ocr_text: bool = False) -> TradeState:
    """Read a trade using a previously detected layout (cheap, per-frame)."""
    state = TradeState()
    state.scale = layout["scale"]
    pills = find_chart_pills(image, layout, ocr_text=ocr_text)
    assemble_from_pills(state, pills)
    return state


TAG_WORD_RE = re.compile(r"(?:imit|lmit|limt|st[o0]p)", re.IGNORECASE)
TAG_PAIR_RE = re.compile(
    r"(?:(?:buy|sell|se[l1]l)\s*\S*(?:imit|lmit|limt|st[o0]p))"
    r"|(?:\d{1,3}\s*\S*(?:imit|lmit|limt|st[o0]p))",
    re.IGNORECASE,
)


def _line_masks(hsv: np.ndarray) -> dict:
    return {color: color_mask(hsv, color) for color in COLOR_RANGES}


def has_attached_line(masks: dict, y: int, x0: int, x1: int, min_frac: float = 0.25) -> bool:
    """True when a coloured line spans much of the plot at this row.

    This is the rule that separates a real order from an axis badge or UI text:
    only orders draw a line across the chart with a tag attached.
    """
    width = max(1, x1 - x0)
    need = int(width * min_frac)
    for mask in masks.values():
        band = mask[max(0, y - 2): y + 3, x0:x1]
        if band.size == 0:
            continue
        columns = (band > 0).any(axis=0).astype(np.uint8)
        bridged = cv2.dilate(columns.reshape(1, -1), np.ones((1, 15), np.uint8)).ravel()
        run = 0
        best = 0
        for value in bridged:
            run = run + 1 if value else 0
            best = max(best, run)
        if best >= need:
            return True
    return False


def ocr_chart_text(image: np.ndarray, layout: dict, band_height: int = 380,
                   upscale: float = 1.6) -> list[dict]:
    """OCR the chart in horizontal bands at native-ish resolution.

    One OCR pass over a tall region gets downscaled internally and loses the
    small order tags, so scan in bands where the glyphs stay sharp.
    """
    h, w = image.shape[:2]
    ax, ay, aw, ah = layout["axis_roi"]
    axis_x0 = int(ax * w)
    axis_x1 = int((ax + aw) * w)
    y0 = max(0, int(ay * h))
    y1 = min(h, int((ay + ah) * h))
    if layout["axis_side"] == "left":
        x0, x1 = axis_x1, w
    else:
        x0, x1 = 0, axis_x0
    scale = layout["scale"]
    items = []
    hsv_full = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masks = _line_masks(hsv_full)
    plot_x0 = axis_x1 if layout["axis_side"] == "left" else 0
    plot_x1 = w if layout["axis_side"] == "left" else axis_x0
    for band_y in range(y0, y1, max(80, band_height - 60)):
        band_end = min(y1, band_y + band_height)
        crop = image[band_y:band_end, x0:x1]
        if crop.size == 0:
            continue
        big = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        try:
            result, _ = ocr()(big)
        except Exception:
            continue
        for box, text, conf in (result or []):
            if not text or float(conf) < 0.4:
                continue
            cy = band_y + (sum(p[1] for p in box) / 4) / upscale
            cx = x0 + (sum(p[0] for p in box) / 4) / upscale
            candidate = TAG_PAIR_RE.search(text) or ENTRY_RE.search(text)
            if not candidate:
                continue
            if not has_attached_line(masks, int(round(cy)), plot_x0, plot_x1):
                continue
            items.append({
                "text": text,
                "confidence": round(float(conf), 2),
                "y": round(cy, 1),
                "x": round(cx, 1),
                "price": round(price_at(scale, cy), 2),
            })
    return items


ENTRY_RE = re.compile(r"[+-]?\d[\d,]*\.\d+\s*(?:usd)(?![a-z])", re.IGNORECASE)
BARE_NUMBER_RE = re.compile(r"^[-+]?\d{1,3}$")


def attach_quantities(tags: list[dict], items: list[dict]) -> None:
    """Pair a bare size box with its tag only when the tag has no digits.

    Embedded sizes ("2Sell stop") are the reliable case; a free-floating number
    nearby is often a price fragment, so keep this conservative.
    """
    numbers = [
        item for item in items
        if BARE_NUMBER_RE.match((item.get("text") or "").strip())
        and 0 < int((item.get("text") or "0").strip().lstrip("+") or 0) <= 50
    ]
    for tag in tags:
        if tag.get("qty") or re.search(r"\d", tag.get("text") or ""):
            continue
        best = None
        for item in numbers:
            if abs(item["y"] - tag["y"]) > 10:
                continue
            if not (0 < tag["x"] - item["x"] < 70):
                continue
            if best is None or item["x"] > best["x"]:
                best = item
        if best is not None:
            try:
                tag["qty"] = int(best["text"].strip().lstrip("+"))
            except ValueError:
                pass


def assemble_from_ocr(state: TradeState, items: list[dict]) -> None:
    """Turn OCR lines into orders: stop, limit, and the position badge."""
    tags = []
    for item in items:
        text = item["text"]
        if TAG_PAIR_RE.search(text):
            info = classify_tag_text(text)
            tags.append({**item, **info})
        elif ENTRY_RE.search(text):
            side = None
            tags.append({**item, "kind": "entry", "side": side, "qty": None})
    attach_quantities(tags, items)
    state.markers = tags
    stops = [t for t in tags if t.get("kind") == "stop"]
    limits = [t for t in tags if t.get("kind") == "target"]
    entries = [t for t in tags if t.get("kind") == "entry"]
    sides = [t["side"] for t in tags if t.get("side")]
    qtys = [t["qty"] for t in tags if t.get("qty")]

    state.stop = max(t["price"] for t in stops) if stops else None
    state.targets = sorted((t["price"] for t in limits), reverse=True)
    state.entry = max((t["price"] for t in entries), default=None)
    if sides:
        state.direction = max(set(sides), key=sides.count)
    if qtys:
        state.quantity = min(qtys)
    state.in_trade = bool(state.stop is not None or state.targets or state.entry is not None)
    state.confidence = {
        "ocr_lines": len(items),
        "stop": state.stop is not None,
        "targets": len(state.targets),
        "entry": state.entry is not None,
        "qty": state.quantity is not None,
    }
    if not state.in_trade:
        state.notes.append("no order lines recognised")


def read_trade_profile(image: np.ndarray, profile: dict) -> TradeState:
    """Read using fixed ROIs for a known streamer layout (fast, no auto-scan)."""
    axis_roi = tuple(profile["axis_roi"])
    chart_roi = tuple(profile["chart_roi"])
    tag_roi = tuple(profile.get("tag_roi") or ())
    state = TradeState()
    labels = read_axis_labels(image, axis_roi)
    scale = calibrate(labels)
    if not scale:
        state.notes.append("axis calibration failed for profile")
        return state
    h, w = image.shape[:2]
    ax, ay, aw, ah = axis_roi
    axis_x0 = int(ax * w)
    axis_x1 = int((ax + aw) * w)
    side = "left" if axis_x0 < w / 2 else "right"
    layout = {
        "axis_roi": axis_roi,
        "chart_roi": chart_roi,
        "tag_roi": tag_roi,
        "scale": scale,
        "axis_side": side,
    }
    state.scale = scale
    items = ocr_chart_text(image, layout)
    if items:
        assemble_from_ocr(state, items)
        if state.in_trade:
            return state
    pills = find_chart_pills(image, layout, ocr_text=True)
    assemble_from_pills(state, pills)
    return state


def read_trade_auto(image: np.ndarray, ignore_yellow: bool = False) -> TradeState:
    """Locate the axis, then prefer tag text; fall back to coloured pills."""
    from auto_layout import find_axis

    layout = find_axis(image)
    state = TradeState()
    if layout is None:
        state.notes.append("could not locate a price axis")
        return state
    state.scale = layout["scale"]
    state.notes.append(f"auto axis ({layout['axis_side']})")

    tags = ocr_chart_text(image, layout)
    if tags:
        assemble_from_ocr(state, tags)
        if state.in_trade:
            return state

    pills = find_chart_pills(image, layout, ocr_text=True)
    assemble_from_pills(state, pills)
    return state


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", type=Path)
    ap.add_argument("--chart-roi", required=True, help="normalized x,y,w,h of the chart area")
    ap.add_argument("--axis-roi", required=True, help="normalized x,y,w,h of the price-axis strip")
    ap.add_argument("--tag-roi", default=None, help="normalized x,y,w,h of the order-tag column (default: right fifth of chart)")
    ap.add_argument("--ignore-yellow", action="store_true")
    args = ap.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"cannot read {args.image}")
    state = read_trade(
        image,
        tuple(float(v) for v in args.chart_roi.split(",")),
        tuple(float(v) for v in args.axis_roi.split(",")),
        args.ignore_yellow,
        tag_roi=tuple(float(v) for v in args.tag_roi.split(",")) if args.tag_roi else None,
    )
    print(json.dumps(asdict(state), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
