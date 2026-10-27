"""Supervisor: start the trade watcher when a channel goes live, stop when it ends.

Polls each channel's Streams tab. When a new live video appears it launches
``scripts/watch_trades.py`` for that stream; when the stream ends (the watcher
exits or the channel stops reporting live) it shuts the watcher down.

Channels and options live in ``config/streams.json``.

Usage:
    python scripts/auto_watch.py                  # run forever
    python scripts/auto_watch.py --once           # single poll, then exit
    python scripts/auto_watch.py --dry-run        # watchers log, no Discord posts
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "streams.json"
LOG_DIR = REPO_ROOT / "runs" / "live"
SUPERVISOR_LOG = REPO_ROOT / "runs" / "auto_watch.log"


def log(message: str) -> None:
    stamp = dt.datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with SUPERVISOR_LOG.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def probe_channel(streams_url: str, timeout: float = 120.0) -> list[dict]:
    command = [
        sys.executable, "-m", "yt_dlp",
        "--flat-playlist",
        "--print", "%(id)s|%(live_status)s|%(title)s",
        "--playlist-end", "5",
        "--no-warnings",
        streams_url,
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=timeout,
    )
    videos = []
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) >= 2 and parts[0].strip():
            videos.append({
                "id": parts[0].strip(),
                "status": parts[1].strip(),
                "title": "|".join(parts[2:]).strip(),
            })
    return videos


def start_watcher(entry: dict, video_id: str, args) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{entry['source_id']}.log"
    handle = log_path.open("a", encoding="utf-8")
    handle.write(
        f"\n=== {dt.datetime.now().isoformat(timespec='seconds')} "
        f"start {entry['name']} {video_id} ===\n"
    )
    handle.flush()

    command = [
        sys.executable, "-u", "scripts/watch_trades.py",
        "--url", f"https://www.youtube.com/watch?v={video_id}",
        "--instrument", entry.get("instrument", "NQ"),
        "--source-id", entry["source_id"],
        "--reader", args.reader,
    ]
    if args.routes_config:
        command += ["--routes-config", args.routes_config]
    if entry.get("profile"):
        command += ["--profile", entry["profile"]]
    if args.dry_run:
        command.append("--dry-run")

    process = subprocess.Popen(
        command, cwd=str(REPO_ROOT), stdout=handle, stderr=subprocess.STDOUT
    )
    log(f"started {entry['name']} ({video_id}) pid={process.pid} -> {log_path.name}")
    return {
        "process": process,
        "handle": handle,
        "video": video_id,
        "started": time.time(),
        "entry": entry,
        "log": log_path,
    }


def stop_watcher(active: dict, reason: str) -> None:
    process = active["process"]
    try:
        if process.poll() is None:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
            )
    except Exception:
        pass
    try:
        active["handle"].write(
            f"=== {dt.datetime.now().isoformat(timespec='seconds')} stop ({reason}) ===\n"
        )
        active["handle"].close()
    except Exception:
        pass
    log(f"stopped {active['entry']['name']} ({active['video']}): {reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--poll", type=float, default=None, help="seconds between polls")
    parser.add_argument("--reader", default=None, choices=("vlm", "ocr"))
    parser.add_argument("--routes-config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true", help="poll once and exit")
    parser.add_argument("--max-runtime", type=float, default=8 * 3600,
                        help="stop a watcher after this many seconds")
    parser.add_argument("--run-seconds", type=float, default=None,
                        help="exit the supervisor after this many seconds (0 = forever)")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    options = config.get("options", {})
    args.poll = args.poll or float(options.get("poll_seconds", 60))
    args.reader = args.reader or options.get("reader", "vlm")
    args.routes_config = args.routes_config or options.get("routes_config")
    if options.get("dry_run", False):
        args.dry_run = True

    channels = [c for c in config.get("channels", []) if c.get("enabled", True)]
    if not channels:
        parser.error("no enabled channels in config")

    log(f"supervisor starting: {len(channels)} channel(s), poll={args.poll:.0f}s, "
        f"reader={args.reader}, dry_run={args.dry_run}")
    active: dict[str, dict] = {}
    started_at = time.time()

    try:
        while True:
            for entry in channels:
                source_id = entry["source_id"]
                try:
                    videos = probe_channel(entry["streams_url"])
                    probe_ok = bool(videos)
                except Exception as exc:
                    log(f"{entry['name']}: probe failed ({type(exc).__name__})")
                    probe_ok = False
                    videos = []
                if not probe_ok:
                    continue

                current = active.get(source_id)
                live = [v for v in videos if v["status"] == "is_live"]

                if live:
                    video_id = live[0]["id"]
                    if current and current["video"] != video_id:
                        stop_watcher(current, f"replaced by new stream {video_id}")
                        active.pop(source_id, None)
                        current = None
                    if current is None:
                        active[source_id] = start_watcher(entry, video_id, args)
                    elif current["process"].poll() is not None:
                        current["handle"].close()
                        active.pop(source_id, None)
                        log(f"{entry['name']}: watcher exited on its own")
                else:
                    if current is None:
                        continue
                    if current["process"].poll() is not None:
                        current["handle"].close()
                        active.pop(source_id, None)
                        log(f"{entry['name']}: stream ended (watcher exited)")
                    elif probe_ok and any(v["id"] == current["video"] and v["status"] in ("was_live", "post_live") for v in videos):
                        stop_watcher(current, "stream finished")
                        active.pop(source_id, None)
                    elif probe_ok and videos and not live:
                        # Channel is reachable and no live video is listed; give it
                        # one extra poll before tearing the watcher down.
                        current["misses"] = current.get("misses", 0) + 1
                        if current["misses"] >= 2:
                            stop_watcher(current, "channel no longer live")
                            active.pop(source_id, None)
                    else:
                        current["misses"] = 0

            for source_id, current in list(active.items()):
                if time.time() - current["started"] > args.max_runtime:
                    stop_watcher(current, "max runtime reached")
                    active.pop(source_id, None)

            if args.once:
                break
            if args.run_seconds and time.time() - started_at >= args.run_seconds:
                log("run window elapsed")
                break
            time.sleep(args.poll)
    except KeyboardInterrupt:
        log("interrupt received")
    finally:
        for source_id, current in list(active.items()):
            stop_watcher(current, "supervisor shutdown")
        log("supervisor stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
