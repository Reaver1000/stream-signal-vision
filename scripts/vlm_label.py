"""VLM-assisted frame labeler for stream-signal-vision.

Sends frames to a local Ollama vision model (qwen2.5vl) and drafts
annotations in the JSONL schema consumed by prepare_yolo.py:
  {"image_path", "platform_id", "session_id", "elements": [...],
   "event": {...}, "needs_review": [...], "confidence": {...}}

Frames are first screened by a cheap grayscale diff so the VLM only sees
keyframes (transitions + quiet anchors), not every near-identical frame.

Usage:
  python scripts/vlm_label.py data/raw/tradingview/channel-a-tv-02/frames \
      --out data/raw/tradingview/tv-02-annotations.jsonl --start 2400 --end 2480
  python scripts/vlm_label.py <frames_dir> --out <jsonl>              # full pass
  python scripts/vlm_label.py <frames_dir> --out <jsonl> --no-diff    # label all
"""
import argparse
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

import cv2

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = os.environ.get("SSV_VLM_MODEL", "qwen2.5vl:7b")
DIFF_THRESHOLD = 14.0   # mean absdiff (0-255) worth sending to the VLM
MAX_IMAGE_EDGE = 1280   # downscale before base64 (VLM time + tokens)

PLATFORMS = ["tradingview", "topstepx", "tradovate", "wealthcharts", "mt5", "other"]
KINDS = ["entry_marker", "stop_line", "target_line", "quantity_badge", "position_panel", "order_ticket"]

SYSTEM_PROMPT = (
    "You label screenshots from live futures-trading livestreams. The screen may "
    "show any platform (TradingView, TopstepX ProjectX, Tradovate, WealthCharts, "
    "MetaTrader5) or an unrelated screen (chat, desktop, browser). "
    "Return ONLY a JSON object with keys: "
    "platform_id (one of " + ", ".join(PLATFORMS) + "; use 'other' if no trading platform chart is visible), "
    "elements (array; each item: {kind, bbox:[x1,y1,x2,y2], text?}); kind in "
    "[" + ", ".join(KINDS) + "]. bbox is [x1, y1, x2, y2] in PIXEL coordinates of this exact image "
    "(top-left origin). Box the entry position marker, the stop-loss line/label, "
    "each take-profit line/label, the visible order quantity, the filled-position badge/panel, "
    "and any order-entry ticket/dialog. Label a ticket as order_ticket, not position_panel. "
    "If none are visible, use []. "
    "event (object with trade_state: open|pending|flat|unknown, action: entry|modify|exit|none, "
    "instrument, direction: long|short|none, "
    "order_type: market|limit, entry_price, stop_price, target_prices (array), "
    "displayed_quantity: int; use null/none values when absent; if the trader filled at "
    "market, order_type=market and entry_price is the fill/ask-bid-ish visible price. "
    "CRITICAL: a visible order ticket, limit line, drawing, or planned setup is NOT a trade. "
    "Use trade_state=pending for an unfilled working order, flat when no position is open, "
    "open only when the screen shows evidence of a filled/current position (position panel, "
    "filled marker or explicit position quantity). Set action=entry only for a transition "
    "into an evidenced open position; never infer a fill from an order being present. "
    "A single frame usually cannot prove a transition: set action=none unless a visible "
    "platform toast/log explicitly reports a fill, modification or exit. Sequence logic derives "
    "entry/modify/exit transitions from consecutive trade_state and level observations. "
    "Set action=modify only for an explicit modification toast/log). "
    "confidence (object mapping each filled field name to 0.0-1.0). "
    "needs_review (array of short strings listing anything you are unsure about). "
    "Be conservative: wrong prices are worse than no answer. Empty chart -> empty elements, action none."
)


def norm_mean_absdiff(a, b):
    return cv2.absdiff(a, b).mean()


def pick_keyframes(frames, use_diff):
    keep, prev_gray, chosen = [], None, []
    for i, p in enumerate(frames):
        if not use_diff:
            keep.append(i)
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (960, 540))
        if prev_gray is None or norm_mean_absdiff(img, prev_gray) > DIFF_THRESHOLD:
            keep.extend(range(max(0, i - 2), min(len(frames), i + 3)))
            chosen.append(i)
        prev_gray = img
    return sorted(set(keep))


def sample_evenly(items, limit):
    if limit is None or limit <= 0 or len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[(len(items) - 1) // 2]]
    indices = [round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)]
    return [items[i] for i in indices]


def downscale_b64(path):
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    scale = MAX_IMAGE_EDGE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode(), (img.shape[1], img.shape[0])


def normalize_elements(draft, w, h):
    for el in draft.get("elements") or []:
        bx = el.get("bbox")
        if not (isinstance(bx, list) and len(bx) == 4):
            el["bbox"] = None
            continue
        try:
            b = [float(v) for v in bx]
        except (TypeError, ValueError):
            el["bbox"] = None
            continue
        if any(v > 1.0 for v in b):
            x1, y1, x2, y2 = b
            if x2 < x1 or y2 < y1:
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
            el["bbox"] = [round(x1 / w, 5), round(y1 / h, 5),
                          round((x2 - x1) / w, 5), round((y2 - y1) / h, 5)]
        else:
            x, y, bw, bh = b
            el["bbox"] = [round(x, 5), round(y, 5), round(bw, 5), round(bh, 5)]
    return draft


def ask_vlm(b64, num_gpu=0, model=None):
    body = json.dumps({
        "model": model or DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Label this frame.",
             "images": [b64]},
        ],
        "format": "json",
        "options": {"temperature": 0, "num_ctx": 3072, "num_gpu": num_gpu},
        "stream": False,
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["message"]["content"]


def sanity(record):
    issues = list(record.get("needs_review") or [])
    for el in record.get("elements", []):
        bx = el.get("bbox")
        kinds = el.get("kind") in KINDS
        flat = bx and len(bx) == 4 and all(0.0 <= float(v) <= 1.0 for v in bx)
        if not kinds or not flat:
            issues.append(f"bad_element:{el.get('kind')}: {bx}")
    ev = record.get("event") or {}
    if ev.get("action") == "entry" and ev.get("trade_state") != "open":
        issues.append("entry_without_confirmed_open_position")
    if ev.get("action") == "modify" and ev.get("trade_state") != "open":
        issues.append("modify_without_open_position")
    sp, ep = ev.get("stop_price"), ev.get("entry_price")
    if sp and ep and ev.get("direction"):
        long_i = ev["direction"] == "long"
        breach = (float(sp) >= float(ep)) if long_i else (float(sp) <= float(ep))
        if breach:
            issues.append("stop_on_wrong_side")
    conf = record.get("confidence") or {}
    low = [k for k, v in conf.items() if isinstance(v, (int, float)) and v < 0.6
           and k not in ("platform_id",)]
    if low:
        issues.append("low_conf:" + ",".join(low))
    if record.get("platform_id") == "other" and (ev.get("action") or "none") != "none":
        issues.append("event_without_platform")
    record["needs_review"] = sorted(set(issues))
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--session-id")
    ap.add_argument("--start", type=int, default=0, help="first frame index to consider")
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--no-diff", action="store_true", help="label every frame")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Ollama vision model name (env SSV_VLM_MODEL)")
    ap.add_argument("--num-gpu", type=int, default=int(os.environ.get("SSV_VLM_NUM_GPU", "-1")),
                    help="Ollama GPU layers; -1 uses all layers, 0 forces CPU")
    a = ap.parse_args()

    frames = sorted(p for p in a.frames.glob("*.png"))
    if not frames:
        frames = sorted(p for p in a.frames.glob("*.jpg"))
    if not frames:
        sys.exit(f"no images under {a.frames}")
    end = a.end if a.end is not None else len(frames)
    frames = frames[a.start:end]
    session = a.session_id or a.frames.parent.name

    keep = pick_keyframes(frames, use_diff=not a.no_diff)
    if a.limit:
        keep = sample_evenly(keep, a.limit)
    print(f"{len(frames)} frames in window, {len(keep)} keyframes to label")

    n_done, n_review = 0, 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("a", encoding="utf-8") as f:
        for i in keep:
            path = frames[i]
            try:
                b64, dims = downscale_b64(path)
                draft = json.loads(ask_vlm(b64, num_gpu=a.num_gpu, model=a.model))
            except Exception as e:
                print(f"[{path.name}] FAILED: {e}")
                continue
            if isinstance(draft, str):
                draft = json.loads(draft)
            draft = normalize_elements(draft, *dims)
            draft["image_path"] = path.name
            draft["platform_id"] = draft.get("platform_id", "other")
            draft["session_id"] = session
            draft["frame_index"] = i
            draft["source_size"] = list(dims)
            draft = sanity(draft)
            f.write(json.dumps(draft, ensure_ascii=False) + "\n")
            n_done += 1
            if draft["needs_review"]:
                n_review += 1
                print(f"[{path.name}] REVIEW: {draft['needs_review']}")
            else:
                act = (draft.get("event") or {}).get("action", "?")
                n_el = len(draft.get("elements", []))
                print(f"[{path.name}] {draft['platform_id']} act={act} els={n_el}")
    print(f"done: {n_done} labeled, {n_review} flagged for review -> {a.out}")


if __name__ == "__main__":
    main()
