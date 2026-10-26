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
import json
import sys
import time
from pathlib import Path

import cv2

SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from trade_reader import read_trade, read_trade_auto, read_with_layout  # noqa: E402

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


def format_message(state, previous, source_id: str, instrument: str | None) -> str | None:
    label = instrument or state.instrument or (previous.instrument if previous else None) or "?"
    if state.in_trade and not (previous and previous.in_trade):
        parts = [f"**{label} {state.direction.upper()}**"]
        if state.quantity:
            parts.append(f"qty {state.quantity}")
        parts.append(f"entry {money(state.entry)}")
        parts.append(f"SL {money(state.stop)}")
        if state.targets:
            parts.append("TP " + " / ".join(money(t) for t in state.targets))
        return f"[{source_id}] " + " | ".join(parts)
    if state.in_trade and previous and previous.in_trade:
        changes = []
        if state.quantity and previous.quantity and state.quantity != previous.quantity:
            delta = state.quantity - previous.quantity
            changes.append(f"size {previous.quantity} -> {state.quantity} ({delta:+d})")
        if state.stop and previous.stop and abs(state.stop - previous.stop) >= 0.25:
            changes.append(f"SL {money(previous.stop)} -> {money(state.stop)}")
        if state.targets and previous.targets and state.targets != previous.targets:
            changes.append("TP -> " + " / ".join(money(t) for t in state.targets))
        if changes:
            return f"[{source_id}] **{label} {state.direction.upper()}** update: " + "; ".join(changes)
        return None
    if not state.in_trade and previous and previous.in_trade:
        return f"[{source_id}] **{label} {previous.direction.upper()}** closed"
    return None


def evaluate(image, profile, cache=None, layout_ttl: float = 10.0):
    if profile is not None:
        return read_trade(
            image,
            tuple(profile["chart_roi"]),
            tuple(profile["axis_roi"]),
            tag_roi=tuple(profile.get("tag_roi") or ()),
        )
    if cache is None:
        return read_trade_auto(image)
    now = time.time()
    layout = cache.get("layout")
    if layout is None or now - cache.get("at", 0.0) > layout_ttl:
        from auto_layout import find_axis
        layout = find_axis(image)
        cache["layout"] = layout
        cache["at"] = now
    if layout is None:
        return read_trade_auto(image)
    return read_with_layout(image, layout)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="live stream URL")
    source.add_argument("--frames-dir", type=Path, help="replay a frames directory")
    source.add_argument("--image", type=Path, help="single still image")
    ap.add_argument("--profile", default="auto",
                    help="auto (find the axis itself) or a name from config/profiles.json")
    ap.add_argument("--source-id", default="live-stream")
    ap.add_argument("--discord-webhook", default=None)
    ap.add_argument("--routes-config", default=None)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--confirm-frames", type=int, default=2,
                    help="identical reads required before emitting")
    ap.add_argument("--max-seconds", type=float, default=None)
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

    if args.image is not None:
        image = cv2.imread(str(args.image))
        if image is None:
            raise SystemExit(f"cannot read {args.image}")
        state = evaluate(image, profile)
        message = format_message(state, None, args.source_id, args.instrument or (profile or {}).get("instrument"))
        print(message or "[no position detected]")
        return 0

    if args.frames_dir is not None:
        frames = replay_frames(args.frames_dir, None, 0.0, None)
    else:
        frames = live_frames(args.url, args.fps, args.width, args.height, None, 5.0, None)

    previous = None
    pending = None
    pending_count = 0
    started = time.time()
    try:
        for image, _, stamp, _ in frames:
            state = evaluate(image, profile)
            sig = signature(state)
            if sig == pending:
                pending_count += 1
            else:
                pending = sig
                pending_count = 1
            if pending_count < args.confirm_frames:
                continue
            message = format_message(state, previous, args.source_id, args.instrument or (profile or {}).get("instrument"))
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
