"""Vision-model trade reader: read the order tags with a local VLM.

The chart is read the way a person reads it: look at the tags beside the lines,
note quantity and type, and take the price printed on the axis at that height.
That is layout-agnostic, which is why it works across TradingView, TopstepX and
Tradovate without per-streamer coordinates.

Latency comes from model load, not inference. The engine is preloaded once with
a keep-alive so a steady-state read is ~1.5 s on an RTX 3070.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.request

import cv2

from price_axis import calibrate, read_axis_labels

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

PROMPT = (
    "You are reading a futures trading chart. Return ONLY JSON: "
    '{"instrument": string|null, "position_qty": number|null, '
    '"orders": [{"type": "limit"|"stop", "side": "buy"|"sell", '
    '"qty": number|null, "price": number|null}]}. '
    "Order tags are attached to horizontal lines across the chart, for example "
    "'2 Sell stop' or '1 Sell limit'. The leading number is the quantity; LIMIT is a "
    "take-profit and STOP is a stop-loss. A badge showing a +/- USD value is the open "
    "position. Prices are printed on the price axis at the same height as the tag; write "
    "them as plain numbers without commas. Ignore the Buy/Sell buttons and the header. "
    "If there are no tags, return an empty list."
)


class OllamaVision:
    """Thin client that keeps the vision model resident between reads."""

    def __init__(self, model: str = "qwen2.5vl:7b", keep_alive: str = "30m",
                 num_predict: int = 220, timeout: float = 60.0):
        self.model = model
        self.keep_alive = keep_alive
        self.num_predict = num_predict
        self.timeout = timeout
        self.last_latency = 0.0

    def _post(self, content: str, image_b64: str | None) -> dict:
        message = {"role": "user", "content": content}
        if image_b64:
            message["images"] = [image_b64]
        body = json.dumps({
            "model": self.model,
            "messages": [message],
            "format": "json",
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0,
                "num_gpu": -1,
                "num_ctx": 4096,
                "num_predict": self.num_predict,
            },
        }).encode()
        request = urllib.request.Request(
            OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read())
        return json.loads(payload["message"]["content"])

    def warm(self) -> float:
        """Load the model and vision projector so the first real read is fast."""
        blank = cv2.imencode(".jpg", cv2.UMat(64, 64, cv2.CV_8UC3).get())[1]
        b64 = base64.b64encode(blank.tobytes()).decode()
        start = time.time()
        try:
            self._post("Return {\"ok\": true}.", b64)
        except Exception:
            return 0.0
        return time.time() - start

    def read(self, crop) -> dict:
        ok, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            return {}
        b64 = base64.b64encode(buffer.tobytes()).decode()
        start = time.time()
        try:
            data = self._post(PROMPT, b64)
        except Exception as exc:
            self.last_latency = time.time() - start
            return {"error": f"{type(exc).__name__}: {exc}"}
        self.last_latency = time.time() - start
        return data


def tag_band_crop(image, layout, max_width: int = 1200):
    """Crop the half of the chart that holds the tags, including the axis text.

    Trims the top header strip (ticker, timeframes, Buy/Sell buttons) and the
    bottom time axis so the model cannot mistake UI buttons for order tags.
    """
    h, w = image.shape[:2]
    ax, ay, aw, ah = layout["axis_roi"]
    axis_x0 = int(ax * w)
    axis_x1 = int((ax + aw) * w)
    y0 = max(0, int(ay * h))
    y1 = min(h, int((ay + ah) * h))
    pad = int((y1 - y0) * 0.10)
    y0 = min(y0 + pad, y1 - 20)
    y1 = max(y1 - int((y1 - y0) * 0.04), y0 + 20)
    if layout["axis_side"] == "left":
        x0 = axis_x0
        x1 = min(w, axis_x1 + int(w * 0.55))
    else:
        x0 = max(0, axis_x0 - int(w * 0.55))
        x1 = axis_x1
    crop = image[y0:y1, x0:x1]
    if crop.size and crop.shape[1] > max_width:
        scale = max_width / crop.shape[1]
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return crop


def _num(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("$", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def assemble(orders: list[dict], position_qty, scale: dict | None) -> dict:
    """Turn model output into stop / targets / direction / quantity."""
    stops: list[float] = []
    targets: list[float] = []
    sides: list[str] = []
    qtys: list[int] = []
    low = high = None
    if scale:
        low, high = scale["price_range"]
        span = abs(high - low) or 1.0
    for order in orders or []:
        price = _num(order.get("price"))
        if price is None or price <= 0:
            continue
        if low is not None and not (low - span * 0.05 <= price <= high + span * 0.05):
            continue
        kind = str(order.get("type") or "").lower()
        side = str(order.get("side") or "").lower()
        qty = _num(order.get("qty"))
        if qty and 0 < qty <= 500:
            qtys.append(int(qty))
        if side in ("buy", "sell"):
            sides.append(side)
        if "limit" in kind:
            targets.append(price)
        elif "stop" in kind:
            stops.append(price)
    direction = None
    if sides:
        # A sell bracket closes a long, a buy bracket closes a short.
        direction = "long" if sides.count("sell") >= sides.count("buy") else "short"
    quantity = int(_num(position_qty) or 0) or (max(qtys) if qtys else None)
    return {
        "stop": max(stops) if stops else None,
        "targets": sorted(targets, reverse=True),
        "direction": direction,
        "quantity": quantity,
    }


def read_with_vlm(image, engine: OllamaVision, layout_cache: dict) -> dict:
    """Read a frame, reusing the cached layout unless it stops working."""
    from auto_layout import find_axis

    layout = layout_cache.get("layout")
    if layout is None:
        layout = find_axis(image)
        layout_cache["layout"] = layout
    if layout is None:
        return {"in_trade": False, "notes": ["no price axis found"]}

    crop = tag_band_crop(image, layout)
    data = engine.read(crop)
    if "error" in data:
        return {"in_trade": False, "notes": [data["error"]], "error": True}

    assembled = assemble(data.get("orders") or [], data.get("position_qty"), layout["scale"])
    in_trade = bool(assembled["stop"] or assembled["targets"] or assembled["quantity"])
    if not in_trade:
        layout_cache["empty"] = layout_cache.get("empty", 0) + 1
        if layout_cache["empty"] >= 3:
            layout_cache["layout"] = None
            layout_cache["empty"] = 0
    else:
        layout_cache["empty"] = 0
    return {
        "in_trade": in_trade,
        "direction": assembled["direction"],
        "quantity": assembled["quantity"],
        "stop": assembled["stop"],
        "targets": assembled["targets"],
        "instrument": data.get("instrument"),
        "latency": round(engine.last_latency, 2),
    }
