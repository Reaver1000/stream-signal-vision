"""Read trade details from a chart frame via the local VLM (transition-moment enrichment).

The fast visual classifier detects state transitions; this reader runs once at
the transition moment to extract confluence details from the red/green
position/order box drawn on the chart:

- green region on TOP of the pair -> long (buy); on the BOTTOM -> short (sell)
- red region's far edge -> usual stop-loss level
- green region's far edge -> usual take-profit level

This is confluence, not gospel truth. Numbers are reported only when visibly
readable - the prompt forbids inventing values - and a green/red box alone can
be a hypothetical plan; only details captured alongside position-open evidence
become signaled trades. The VLM never runs on every frame; only at transitions.

Usage:
    python scripts/trade_details.py <frame.png>
"""

from __future__ import annotations

import argparse
import base64
import json
import urllib.request
from typing import Any

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5vl:7b"

SYSTEM_PROMPT = (
    "You read futures trading charts. Find the position/order box drawn on the "
    "chart: it has a GREEN region and a RED region. Answer ONLY with a JSON object "
    "with keys: green_on ('top' or 'bottom' or null - which color region sits above "
    "the other), green_box_price (string or null - price printed on/near the green "
    "region), red_box_price (string or null), stop_price (string or null - price at "
    "the stop-loss line/edge), target_price (string or null - price at the "
    "take-profit line/edge), entry_price (string or null), quantity (string or null "
    "- contracts shown on the box or position panel), instrument (string or null - "
    "e.g. NQ, MNQ, ES), chart_symbol (string or null - symbol/ticker on the chart), "
    "description (one short sentence). Read only numbers that are actually visible. "
    "Never invent numbers. If you cannot see a green+red position/order box, set "
    "the color and price fields to null and say so in description."
)

FIELDS = (
    "green_on", "green_box_price", "red_box_price", "stop_price",
    "target_price", "entry_price", "quantity", "instrument",
    "chart_symbol", "description",
)


def _price_value(text: Any) -> float | None:
    if text is None:
        return None
    try:
        return float(str(text).replace(",", "").strip())
    except ValueError:
        return None


def postprocess(details: dict[str, Any]) -> dict[str, Any]:
    """Derive direction from price geometry first; color arrangement is only fallback.

    The user's rule: green on top -> buy, green on bottom -> sell; bottom of red
    is usually the SL and top of green the TP. In practice the printed prices are
    more reliable than the VLM's spatial judgment, so direction comes from
    TP/entry geometry when readable, with contradictions flagged.
    """
    entry = _price_value(details.get("entry_price"))
    stop = _price_value(details.get("stop_price"))
    target = _price_value(details.get("target_price"))

    geometric = None
    if entry is not None and target is not None and target != entry:
        geometric = "long" if target > entry else "short"
    color = {"top": "long", "bottom": "short"}.get(
        str(details.get("green_on") or "").lower() or "", None
    )

    direction = geometric or color
    notes = []
    if geometric and color and geometric != color:
        notes.append(f"color/geometry conflict (using geometry: {geometric})")
    if geometric is None and color:
        notes.append("direction from color arrangement only")
    if entry is not None and stop is not None and target is not None:
        if direction == "long" and not (stop < entry < target):
            notes.append("stop/target sides inconsistent with long geometry")
        if direction == "short" and not (target < entry < stop):
            notes.append("stop/target sides inconsistent with short geometry")
    if entry is not None and target is not None and entry == target:
        notes.append("entry equals target; entry likely misread")

    details["direction"] = direction
    details["direction_source"] = "geometry" if geometric else ("color" if color else None)
    details["notes"] = notes
    return details


def read_trade_details(
    image_bgr: Any,
    url: str = OLLAMA_URL,
    model: str = MODEL,
    timeout: int = 180,
) -> dict[str, Any]:
    """Return trade-detail confluence read from a single chart frame."""
    import cv2

    height, width = image_bgr.shape[:2]
    scale = 1280 / max(height, width)
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, (int(width * scale), int(height * scale)))
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    b64 = base64.b64encode(buf.tobytes()).decode()

    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Read this frame.", "images": [b64]},
        ],
        "format": "json",
        "options": {"temperature": 0},
        "stream": False,
    }).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = json.loads(response.read())["message"]["content"]

    if isinstance(raw, str):
        raw = json.loads(raw)
    details = {field: raw.get(field) for field in FIELDS}
    return postprocess(details)


def format_details(details: dict[str, Any]) -> str:
    """Compact single-line summary for Discord-style messages."""
    parts = []
    if details.get("direction"):
        parts.append("SELL" if details["direction"] == "short" else "BUY")
    for key, label in (
        ("instrument", None),
        ("quantity", "qty"),
        ("entry_price", "entry"),
        ("stop_price", "SL"),
        ("target_price", "TP"),
    ):
        value = details.get(key)
        if value:
            parts.append(f"{label} {value}" if label else str(value))
    if not parts:
        note = str(details.get("description") or "no trade box readable")[:80]
        return f"details: {note}"
    summary = "details: " + " | ".join(str(p) for p in parts)
    for note in details.get("notes", []):
        summary += f" [{note}]"
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    args = parser.parse_args()

    import cv2

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"could not read image: {args.image}")
    details = read_trade_details(image)
    print(json.dumps(details, indent=2, ensure_ascii=False))
    print(format_details(details))
    return 0


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.exit(main())
