"""Watch a live stream (or a replay) and push trade details to Discord.

Monkey-see-monkey-do: decode the stream, read the chart every frame, and when a
position appears or its details change, post them. No model training needed.

Usage:
    # live stream, NSO layout
    python scripts/watch_trades.py --url "https://www.youtube.com/watch?v=..." `
        --profile nso --source-id nso-live --routes-config <private>/webhook_routes.json

    # replay a frames directory
    python scripts/watch_trades.py --frames-dir data/raw/<session>/frames --profile rips

    # single still (debug)
    python scripts/watch_trades.py --image frame.png --profile topstepx --dry-run

Profiles live in config/profiles.json: chart_roi, axis_roi, tag_roi and an
optional instrument label. All ROIs are normalized x,y,w,h.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import json
import sys
import time
from pathlib import Path

import cv2

SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from trade_reader import TradeState, read_trade_auto, read_trade_profile, read_with_layout  # noqa: E402
from vlm_reader import OllamaVision, read_with_vlm  # noqa: E402

PROFILES_PATH = REPO_ROOT / "config" / "profiles.json"
DEFAULT_PROFILES = {
    "topstepx": {
        "chart_roi": [0.024, 0.076, 0.766, 0.50],
        "axis_roi": [0.762, 0.076, 0.026, 0.50],
        "tag_roi": [0.55, 0.076, 0.22, 0.50],
        "instrument": None,
    },
    "keyflow": {
        "chart_roi": [0.138, 0.049, 0.651, 0.549],
        "axis_roi": [0.135, 0.05, 0.040, 0.55],
        "tag_roi": [0.680, 0.049, 0.115, 0.549],
        "instrument": None,
    },
    "rips": {
        "chart_roi": [0.123, 0.092, 0.697, 0.397],
        "axis_roi": [0.753, 0.092, 0.051, 0.409],
        "tag_roi": [0.60, 0.092, 0.20, 0.397],
        "instrument": None,
    },
}


def load_profiles() -> dict:
    profiles = dict(DEFAULT_PROFILES)
    if PROFILES_PATH.is_file():
        try:
            profiles.update(json.loads(PROFILES_PATH.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    return profiles


def resolve_webhook(direct: str | None, routes_config: str | None, source_id: str) -> str | None:
    if direct:
        return direct
    if not routes_config:
        return None
    path = Path(routes_config)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    routes = data.get("routes", data) if isinstance(data, dict) else {}
    if source_id in routes:
        return routes[source_id]
    prefix = source_id.split("-")[0]
    return routes.get(prefix)


def money(value: float | None) -> str:
    if value is None:
        return "?"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def signature(state) -> tuple:
    return (
        state.in_trade,
        state.direction,
        state.quantity,
        round(state.entry, 2) if state.entry else None,
        round(state.stop, 2) if state.stop else None,
        tuple(round(t, 2) for t in state.targets),
    )


def format_message(state, previous, source_id: str, instrument: str | None, stop_step: float = 2.0) -> str | None:
    label = instrument or state.instrument or (previous.instrument if previous else None) or "?"
    long_side = (state.direction or (previous.direction if previous else None)) == "long"
    entry_word = "BUY" if long_side else "SELL"

    def levels(s) -> str:
        parts = []
        if s.stop:
            parts.append(f"STOP {money(s.stop)}")
        if s.targets:
            parts.append("LIMIT " + " / ".join(money(t) for t in s.targets))
        return " | ".join(parts)

    if state.in_trade and not (previous and previous.in_trade):
        qty = f" {state.quantity}" if state.quantity else ""
        message = f"[{source_id}] **{label} {entry_word}{qty} at market**"
        tail = levels(state)
        return message + (f" | {tail}" if tail else "")

    if state.in_trade and previous and previous.in_trade:
        changes = []
        if state.quantity and previous.quantity and state.quantity != previous.quantity:
            delta = abs(state.quantity - previous.quantity)
            if state.quantity > previous.quantity:
                changes.append(f"scale in: {entry_word} {delta} at market (now {state.quantity})")
            else:
                closes = "SELL" if long_side else "BUY"
                changes.append(f"scale out: {closes} {delta} at market (now {state.quantity})")
        if state.stop and previous.stop and abs(state.stop - previous.stop) >= stop_step:
            changes.append(f"STOP {money(previous.stop)} -> {money(state.stop)}")
        elif state.stop and not previous.stop:
            changes.append(f"STOP {money(state.stop)}")
        if state.targets and state.targets != previous.targets:
            changes.append("LIMIT -> " + " / ".join(money(t) for t in state.targets))
        if changes:
            return f"[{source_id}] **{label}** " + "; ".join(changes)
        return None

    if not state.in_trade and previous and previous.in_trade:
        closes = "SELL" if long_side else "BUY"
        qty = f" {previous.quantity}" if previous.quantity else ""
        return f"[{source_id}] **{label} {closes}{qty} at market** (flat)"
    return None


def evaluate(image, profile, cache=None, engine=None):
    """OCR first (fast and exact); vision model only when OCR finds nothing."""
    if profile is not None:
        state = read_trade_profile(image, profile)
    elif cache is not None:
        layout = cache.get("layout")
        if layout is None:
            from auto_layout import find_axis
            layout = find_axis(image)
            cache["layout"] = layout
        state = read_with_layout(image, layout) if layout else read_trade_auto(image)
    else:
        state = read_trade_auto(image)

    if engine is not None and not state.in_trade:
        data = read_with_vlm(image, engine, cache if cache is not None else {})
        if data.get("in_trade"):
            state = TradeState()
            state.in_trade = True
            state.direction = data.get("direction")
            state.quantity = data.get("quantity")
            state.stop = data.get("stop")
            state.targets = list(data.get("targets") or [])
            state.instrument = data.get("instrument")
            state.notes.append("vlm fallback")
    return state


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="live stream URL")
    source.add_argument("--frames-dir", type=Path, help="replay a frames directory")
    source.add_argument("--image", type=Path, help="single still image")
    ap.add_argument("--profile", default="auto",
                    help="auto (find the axis itself) or a name from config/profiles.json")
    ap.add_argument("--reader", choices=("vlm", "ocr"), default="vlm",
                    help="vlm reads the tags with a local vision model; ocr uses text scanning")
    ap.add_argument("--vlm-model", default="qwen2.5vl:7b")
    ap.add_argument("--vlm-keep-alive", default="30m",
                    help="how long Ollama keeps the model resident between reads")
    ap.add_argument("--source-id", default="live-stream")
    ap.add_argument("--discord-webhook", default=None)
    ap.add_argument("--routes-config", default=None)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--confirm-frames", type=int, default=2,
                    help="identical reads required before emitting")
    ap.add_argument("--vote-window", type=int, default=1, help="frames voted per decision")
    ap.add_argument("--flat-confirm", type=int, default=2, help="flat reads required to close a trade")
    ap.add_argument("--stop-step", type=float, default=2.0,
                    help="minimum stop change to report (filters axis interpolation jitter)")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--max-frames", type=int, default=None, help="stop after N frames (replay)")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame (replay)")
    ap.add_argument("--change-threshold", type=float, default=3.0,
                    help="mean frame difference below which the last read is reused")
    ap.add_argument("--instrument", default=None,
                    help="label to use in messages when the header is not read (e.g. MNQ)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    profiles = load_profiles()
    if args.profile == "auto":
        profile = None
    elif args.profile not in profiles:
        raise SystemExit(f"unknown profile {args.profile!r}; have {sorted(profiles)}")
    else:
        profile = profiles[args.profile]
    webhook = resolve_webhook(args.discord_webhook, args.routes_config, args.source_id)

    from watch_live import live_frames, post_discord, replay_frames

    engine = None
    if args.reader == "vlm":
        engine = OllamaVision(args.vlm_model, args.vlm_keep_alive)
        load = engine.warm()
        print(f"[vlm] {args.vlm_model} resident (load {load:.1f}s, keep_alive {args.vlm_keep_alive})", flush=True)

    if args.image is not None:
        image = cv2.imread(str(args.image))
        if image is None:
            raise SystemExit(f"cannot read {args.image}")
        state = evaluate(image, profile, engine=engine)
        message = format_message(state, None, args.source_id, args.instrument or (profile or {}).get("instrument"), args.stop_step)
        print(message or "[no position detected]")
        return 0

    if args.frames_dir is not None:
        frames = replay_frames(args.frames_dir, None, 0.0, None)
    else:
        frames = live_frames(args.url, args.fps, args.width, args.height, None, 5.0, None)

    previous = None
    started = time.time()
    seen = 0
    layout_cache: dict = {}
    last_gray = None
    last_state = None
    votes: deque = deque(maxlen=max(1, args.vote_window))
    flat_streak = 0
    held = None

    def merge(state, reference):
        """Keep known levels when a single read drops them."""
        if reference is None or not reference.in_trade or not state.in_trade:
            return state
        if state.direction is None:
            state.direction = reference.direction
        if state.quantity is None:
            state.quantity = reference.quantity
        if state.stop is None:
            state.stop = reference.stop
        if not state.targets:
            state.targets = list(reference.targets)
        return state

    try:
        for image, _, stamp, _ in frames:
            seen += 1
            if args.stride > 1 and (seen - 1) % args.stride:
                continue
            if args.max_frames is not None and seen > args.max_frames:
                break
            small = cv2.cvtColor(cv2.resize(image, (320, 180)), cv2.COLOR_BGR2GRAY)
            changed = last_gray is None or float(cv2.absdiff(small, last_gray).mean()) >= args.change_threshold
            if changed or last_state is None:
                state = evaluate(image, profile, cache=layout_cache, engine=engine)
            else:
                state = last_state
            last_gray = small
            last_state = state

            votes.append((signature(state), state))
            counts = Counter(sig for sig, _ in votes)
            best_sig, best_count = counts.most_common(1)[0]
            needed = min(len(votes), max(2, (args.vote_window + 1) // 2))
            if best_count < needed:
                continue
            state = next(s for sig, s in votes if sig == best_sig)

            if state.in_trade:
                flat_streak = 0
                state = merge(state, held)
                held = state
            else:
                flat_streak += 1
                if held is not None and flat_streak < args.flat_confirm:
                    state = held
                else:
                    held = None

            message = format_message(state, previous, args.source_id, args.instrument or (profile or {}).get("instrument"), args.stop_step)
            if message:
                print(f"{stamp} {message}", flush=True)
                if webhook and not args.dry_run:
                    try:
                        post_discord(webhook, message)
                    except Exception as exc:
                        print(f"  discord error: {exc}", flush=True)
                previous = state
            if args.max_seconds and time.time() - started > args.max_seconds:
                break
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
