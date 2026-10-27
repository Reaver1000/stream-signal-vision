"""Mirror a position shown anywhere on this PC's screen to NinjaTrader.

Works on a Discord window, a browser, a chart - anything visible. You pick the
scan box once with ``scripts/select_region.py``; this watches that box and reads
the POSITION value:

    "5@29539.25"    green ink -> long 5
    "-1@29551.75"   red ink   -> short 1
    "0"                        -> flat

Each change is translated into the NT8 relay protocol and sent straight to the
local executor:

    OPEN|MNQ|long|5|0|     CLOSE|2     FLATTEN

Usage:
    python scripts/watch_screen.py --dry-run              # read only
    python scripts/watch_screen.py                        # send to NT8
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from price_axis import get_ocr  # noqa: E402
from watch_position import NT8Bridge, format_change, format_commands  # noqa: E402

ROI_PATH = REPO_ROOT / "config" / "screen_roi.json"
CONTRACT_RE = re.compile(r"\s*(-?)\s*(\d{1,3})\s*@")


def read_contracts(crop: np.ndarray) -> int | None:
    """Read the contract count from "<n>@<price>" or "0".

    Glyph scanning is used first: read digits from the left and stop at the wide
    '@' glyph, so the price after it is never mistaken for contracts. OCR on the
    whole line is only a fallback.
    """
    value = read_with_digits(crop)
    if value is not None:
        return value
    if is_covered(crop):
        return None

    big = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    try:
        result, _ = get_ocr()(big)
        text = " ".join(t for _, t, _ in (result or []))
    except Exception:
        return None
    match = CONTRACT_RE.search(text)
    if match:
        value = int(match.group(2))
        return -value if match.group(1) == "-" else value
    digits = re.search(r"(-?\d{1,3})", text)
    return int(digits.group(1)) if digits else None


def is_covered(crop: np.ndarray) -> bool:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float((gray > 200).mean()) > 0.75


def read_with_digits(crop: np.ndarray) -> int | None:
    """Read contracts from a tight crop of "<n>@<price>" or "<n>".

    Upscale first so glyphs separate, then take the digits before the '@' (the
    only glyph that is about as wide as it is tall).
    """
    from watch_position import load_model

    global _MODEL
    try:
        _MODEL
    except NameError:
        _MODEL = load_model()

    big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    height, width = big.shape[:2]
    hsv = cv2.cvtColor(big, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    mask = (((value >= 70) & (saturation >= 70)) | (value >= 140)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < 20:
            continue
        if w * h > 0.5 * width * height:
            continue
        blobs.append((x, y, w, h))
    if not blobs:
        return None
    blobs.sort(key=lambda b: b[0])

    at_index = None
    for index, (x, y, w, h) in enumerate(blobs):
        if 0.95 <= w / max(1, h) <= 1.35:
            at_index = index
            break
    contract_blobs = blobs[:at_index] if at_index else blobs

    negative = False
    digits: list[int] = []
    for x, y, w, h in contract_blobs:
        if w / max(1, h) > 1.5:
            negative = True
            continue
        glyph = mask[y:y + h, x:x + w]
        side = max(w, h)
        square = np.zeros((side, side), np.uint8)
        square[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = glyph
        digit_img = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
        import torch

        tensor = torch.from_numpy(digit_img).float().unsqueeze(0).unsqueeze(0) / 255.0
        tensor = (tensor - 0.1307) / 0.3081
        with torch.no_grad():
            digits.append(int(_MODEL(tensor).argmax(dim=1).item()))
        if len(digits) >= 3:
            break
    if not digits:
        return None
    result = 0
    for digit in digits:
        result = result * 10 + digit
    return -result if negative else result


def ink_direction(crop: np.ndarray) -> str | None:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].reshape(-1)
    value = hsv[:, :, 2].reshape(-1)
    hue = hsv[:, :, 0].reshape(-1)
    keep = (saturation > 90) & (value > 90)
    if keep.sum() < 20:
        return None
    mean_hue = float(hue[keep].mean())
    if mean_hue < 15 or mean_hue > 165:
        return "short"
    if 35 <= mean_hue <= 100:
        return "long"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi", type=Path, default=ROI_PATH)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--instrument", default="MNQ")
    parser.add_argument("--nt-instrument", default="MNQ")
    parser.add_argument("--nt-host", default="127.0.0.1")
    parser.add_argument("--nt-mode", choices=("absolute", "delta"), default="absolute",
                        help="absolute sends POS|<inst>|<signed qty> (position mirror); delta sends OPEN/CLOSE/FLATTEN")
    parser.add_argument("--nt-port", type=int, default=7778)
    parser.add_argument("--no-nt", action="store_true")
    parser.add_argument("--nt-dry-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print only")
    parser.add_argument("--confirm", type=int, default=2)
    parser.add_argument("--max-contracts", type=int, default=50,
                        help="reads above this are ignored as misreads")
    parser.add_argument("--mirror-initial", action="store_true",
                        help="also send the position found at startup (default: baseline only)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--reuse", action="store_true",
                        help="skip the selection screen and reuse config/screen_roi.json")
    parser.add_argument("--check", action="store_true",
                        help="read the box once, report the value and NT8 reachability, then exit")
    args = parser.parse_args()

    region = None
    if not args.reuse:
        from select_region import choose_region

        print("[screen] drag the box around the POSITION value, then press Enter", flush=True)
        region = choose_region()
        if region is not None:
            args.roi.parent.mkdir(parents=True, exist_ok=True)
            args.roi.write_text(json.dumps(region, indent=2) + "\n", encoding="utf-8")
            print(f"[screen] saved box {region}", flush=True)

    if region is None:
        if not args.roi.is_file():
            raise SystemExit("no box selected and no saved box to reuse")
        roi = json.loads(args.roi.read_text(encoding="utf-8"))
        region = {
            "left": roi["left"], "top": roi["top"],
            "width": roi["width"], "height": roi["height"],
        }
        print(f"[screen] reusing saved box {region}", flush=True)

    import mss

    bridge = NT8Bridge(args.nt_host, args.nt_port)

    if args.check:
        import socket

        with mss.mss() as sct:
            shot = sct.grab(region)
        frame = np.frombuffer(shot.bgra, dtype=np.uint8).reshape((shot.height, shot.width, 4))
        crop = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        value = read_contracts(crop)
        direction = ink_direction(crop)
        if value is not None and direction == "short" and value > 0:
            value = -value
        elif value is not None and direction == "long" and value < 0:
            value = -value
        print(f"[check] box {region}")
        print(f"[check] read position: {value} (ink {direction or 'none'})")
        if value:
            if args.nt_mode == "absolute":
                print(f"[check] would send: POS|{args.nt_instrument}|{value}")
            else:
                print(f"[check] would send on flat->this: {format_commands(0, value, args.nt_instrument)}")
        try:
            with socket.create_connection((args.nt_host, args.nt_port), timeout=2):
                print(f"[check] NT8 relay reachable at {args.nt_host}:{args.nt_port}")
        except Exception as exc:
            print(f"[check] NT8 relay NOT reachable at {args.nt_host}:{args.nt_port} ({type(exc).__name__})")
        return 0

    print(f"[screen] scanning {region} at {args.fps}fps -> NT8 {args.nt_host}:{args.nt_port}"
          f"{' (dry run)' if args.nt_dry_run or args.no_nt else ''}", flush=True)
    if region["height"] < 20 or region["width"] < 40:
        print(f"[screen] WARNING: box is only {region['width']}x{region['height']}px. "
              f"Redraw it around the whole POSITION number (needs ~25px height).", flush=True)

    previous = 0
    baseline_set = False
    candidate = None
    candidate_count = 0
    last_gray = None
    last_change = 0.0
    rejected = None
    period = 1.0 / max(1.0, args.fps)

    with mss.mss() as sct:
        while True:
            tick = time.time()
            shot = sct.grab(region)
            frame = np.frombuffer(shot.bgra, dtype=np.uint8).reshape((shot.height, shot.width, 4))
            crop = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            changed = last_gray is None or float(cv2.absdiff(gray, last_gray).mean()) > 1.5
            last_gray = gray
            if changed:
                last_change = time.time()
                value = read_contracts(crop)
                if value is not None:
                    direction = ink_direction(crop)
                    if direction == "short" and value > 0:
                        value = -value
                    elif direction == "long" and value < 0:
                        value = -value
                    if abs(value) > args.max_contracts:
                        if value != rejected:
                            print(f"[screen] ignoring implausible read {value:+d} "
                                  f"(> {args.max_contracts}); redraw the box tighter", flush=True)
                            rejected = value
                        value = None
                    elif args.verbose:
                        print(f"read {value:+d} ({direction or 'sign'})", flush=True)
                if value is not None:
                    if value == candidate:
                        candidate_count += 1
                    else:
                        candidate = value
                        candidate_count = 1
                    if candidate_count >= args.confirm and not baseline_set:
                        baseline_set = True
                        previous = 0
                        if args.mirror_initial:
                            print(f"[screen] baseline {value:+d} (sending initial position)", flush=True)
                        else:
                            print(f"[screen] baseline {value:+d} (no order sent)", flush=True)
                            continue
                    if candidate_count >= args.confirm and value != previous:
                        message = format_change(previous, value, args.instrument)
                        if args.nt_mode == "absolute":
                            commands = [f"POS|{args.nt_instrument}|{value}"]
                        else:
                            commands = format_commands(previous, value, args.nt_instrument)
                        previous = value
                        if message:
                            print(f"{message}", flush=True)
                        for command in commands:
                            if args.nt_dry_run or args.no_nt or args.dry_run:
                                print(f"  nt> {command}", flush=True)
                                continue
                            try:
                                reply = bridge.send(command)
                                print(f"  nt> {command}  <- {reply}", flush=True)
                            except Exception as exc:
                                print(f"  nt error: {exc}", flush=True)
            sleep = period - (time.time() - tick)
            if sleep > 0:
                time.sleep(sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
