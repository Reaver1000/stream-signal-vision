"""Capture consented local screen frames for private annotation.

Requires the optional `capture` dependencies (`mss`). Captures are written
under captures/ (gitignored) and are never uploaded or committed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=(
        "topstepx", "tradingview", "tradovate", "wealthcharts", "mt5"))
    parser.add_argument("--session", required=True, help="Private session/VOD identifier")
    parser.add_argument("--duration", type=int, default=60, help="Capture duration in seconds")
    parser.add_argument("--fps", type=float, default=1.0, help="Sampling rate (default: one frame per second)")
    parser.add_argument("--monitor", type=int, default=1, help="mss monitor index (1 = primary display)")
    args = parser.parse_args()

    try:
        import mss
        import mss.tools
    except ImportError:
        parser.error("Install capture support first: pip install -e '.[capture]'")

    safe_session = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.session)
    out_dir = Path("captures") / "raw" / args.platform / safe_session
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "frames.jsonl"
    interval = 1.0 / max(0.1, args.fps)
    stop_at = time.monotonic() + max(1, args.duration)
    index = 0

    print(f"Capturing local display to {out_dir.resolve()} - keep it private.")
    with mss.mss() as capture, manifest.open("a", encoding="utf-8") as out:
        monitors = capture.monitors
        if args.monitor >= len(monitors):
            parser.error(f"monitor {args.monitor} unavailable; found {len(monitors) - 1} display(s)")
        monitor = monitors[args.monitor]
        while time.monotonic() < stop_at:
            started = time.monotonic()
            shot = capture.grab(monitor)
            index += 1
            captured_at = datetime.now(timezone.utc).isoformat()
            filename = f"frame_{index:06d}_{int(time.time() * 1000)}.png"
            image_path = out_dir / filename
            image_path.write_bytes(mss.tools.to_png(shot.rgb, shot.size))
            record = {
                "image_path": str(image_path.resolve()),
                "platform_id": args.platform,
                "session_id": safe_session,
                "captured_at": captured_at,
                "width": shot.width,
                "height": shot.height,
                "elements": [],
                "event": None,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            remaining = interval - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    print(f"Captured {index} frame(s). Annotate privately; do not commit captures/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
