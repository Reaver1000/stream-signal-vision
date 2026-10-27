"""Live audio trade reader: transcribe the stream and post what the trader says.

Streamers narrate their trades ("I'm long two here, stop 29145, target 29214"),
which is far more reliable than reading tiny pixels. This listens to the live
audio, transcribes rolling windows with faster-whisper, and turns trade speech
into the same Discord messages as the vision path.

Usage:
    python scripts/watch_audio.py --url "<live>" --instrument NQ --source-id nq-audio --dry-run
    python scripts/watch_audio.py --url "<live>" --routes-config <private>/webhook_routes.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

import os

SAMPLE_RATE = 16000
REPO_ROOT = Path(__file__).resolve().parents[1]

PROMPT = (
    "Futures day trading. Terms: NQ, MNQ, ES, long, short, market order, limit, "
    "filled, contracts, stop, stop loss, take profit, target, breakeven, scaled, flat."
)

PRICE_RE = re.compile(r"(?<!\d)(\d{2}[,.]?\d{3}(?:[.,]\d{1,2})?|\d{1,2}[.,]\d{3}(?:[.,]\d{1,2})?)(?!\d)")


def parse_price(token: str) -> float | None:
    cleaned = token.replace(",", "")
    parts = cleaned.split(".")
    if len(parts) == 2 and len(parts[1]) > 2:
        cleaned = parts[0] + "." + parts[1][:2]
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if 100 <= value <= 1_000_000 else None


def prices_in(text: str) -> list[float]:
    found = []
    for match in PRICE_RE.finditer(text):
        price = parse_price(match.group(1))
        if price is not None:
            found.append(price)
    return found


WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
}


def extract_qty(text: str) -> int | None:
    match = re.search(r"\b(\d{1,3})\s*(contracts?|micros?|lots?|cars?|nq|mnq|es|mes)\b", text)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(" + "|".join(WORD_NUMBERS) + r")\s*(contracts?|micros?|lots?)\b", text)
    if match:
        return WORD_NUMBERS[match.group(1)]
    match = re.search(r"\b(?:long|short|bought|sold|buy|sell)\s+(" + "|".join(WORD_NUMBERS) + r")\b", text)
    if match:
        return WORD_NUMBERS[match.group(1)]
    return None


def price_after(text: str, keyword_pattern: str) -> float | None:
    match = re.search(keyword_pattern + r".{0,25}?(\d[\d,.]*)", text)
    if not match:
        return None
    return parse_price(match.group(match.lastindex or 1))


def parse_trade_speech(text: str) -> dict | None:
    """Extract one trade action from a spoken sentence."""
    low = " " + text.lower().replace("’", "'") + " "

    direction = None
    if re.search(r"\b(long|longs|buy|bought)\b", low):
        direction = "long"
    if re.search(r"\b(short|shorts|sell|sold)\b", low):
        direction = "short" if direction is None else direction

    qty = extract_qty(low)
    prices = prices_in(low)
    stop = price_after(low, r"\bstop(?:\s*loss)?\b")
    target = price_after(low, r"\b(?:target|take profit|tp)\b")

    if re.search(r"\b(i'?m|i am|we'?re|we are)\s+(flat|out|done)\b|\bflat now\b|\bi'?m out\b|\bstopped out\b|\btook profit\b", low):
        return {"action": "exit", "prices": prices}

    if re.search(r"\b(breakeven|break even)\b", low) and re.search(r"\bstop\b", low):
        return {"action": "stop_move", "stop": stop, "breakeven": True}

    if re.search(r"\b(move|moving|raise|raising|tighten|tightening|trail|trailing)\b.{0,25}\bstop\b", low):
        return {"action": "stop_move", "stop": stop}

    if re.search(r"\b(add|adding|added|scale in|scaling in|scaled in)\b", low):
        return {"action": "scale_in", "direction": direction, "qty": qty, "price": prices[0] if prices else None}

    if re.search(r"\b(scale out|scaling out|scaled out|trim|trimmed|taking off|took off|partial|partials|covering)\b", low):
        return {"action": "scale_out", "qty": qty, "price": prices[0] if prices else None}

    if direction and (qty or stop or target or re.search(r"\b(entered|entry|filled|in now|i'?m in|taking|took|getting in|placing)\b", low)):
        entry_price = None
        for price in prices:
            if price not in (stop, target):
                entry_price = price
                break
        return {
            "action": "entry",
            "direction": direction,
            "qty": qty,
            "price": entry_price,
            "stop": stop,
            "target": target,
        }

    if target:
        return {"action": "target", "targets": [target]}

    if stop:
        return {"action": "stop", "stop": stop}

    return None


class AudioStream:
    def __init__(self, url: str):
        self.url = url
        self.downloader = None
        self.ffmpeg = None

    def start(self) -> None:
        self.downloader = subprocess.Popen(
            [sys.executable, "-m", "yt_dlp", "-f", "bestaudio/best",
             "--no-playlist", "-o", "-", self.url],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self.ffmpeg = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
             "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
            stdin=self.downloader.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if self.downloader.stdout is not None:
            self.downloader.stdout.close()

    def read(self, seconds: float) -> np.ndarray:
        assert self.ffmpeg is not None and self.ffmpeg.stdout is not None
        need = int(SAMPLE_RATE * seconds) * 2
        data = self.ffmpeg.stdout.read(need)
        if not data or len(data) < need:
            return np.zeros(0, dtype=np.float32)
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    def stop(self) -> None:
        for process in (self.ffmpeg, self.downloader):
            if process is not None:
                try:
                    process.kill()
                except Exception:
                    pass


def post_discord(webhook: str, content: str) -> None:
    body = json.dumps({"content": content}).encode()
    request = urllib.request.Request(
        webhook, data=body,
        headers={"Content-Type": "application/json", "User-Agent": "StreamSignalVision/0.3"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        response.status


def load_transcriber(model_name: str, device: str):
    """Load faster-whisper, adding the CUDA DLL paths torch ships with."""
    try:
        from transcribe_audio import configure_cuda_dll_paths
        configure_cuda_dll_paths()
    except Exception:
        pass
    from faster_whisper import WhisperModel
    if device == "cuda":
        try:
            return WhisperModel(model_name, device="cuda", compute_type="int8_float16")
        except Exception as exc:
            print(f"[audio] cuda unavailable ({type(exc).__name__}); using CPU", flush=True)
    return WhisperModel(model_name, device="cpu", compute_type="int8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--window", type=float, default=12.0, help="seconds of audio per pass")
    parser.add_argument("--hop", type=float, default=6.0, help="seconds between passes")
    parser.add_argument("--instrument", default="NQ")
    parser.add_argument("--source-id", default="live-audio")
    parser.add_argument("--routes-config", default=None)
    parser.add_argument("--discord-webhook", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args()

    webhook = args.discord_webhook
    if webhook is None and args.routes_config:
        path = Path(args.routes_config)
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            routes = data.get("routes", data)
            webhook = routes.get(args.source_id) or routes.get(args.source_id.split("-")[0])

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("install faster-whisper first", file=sys.stderr)
        return 1

    print(f"[audio] loading {args.model} on {args.device}", flush=True)
    model = load_transcriber(args.model, args.device)

    stream = AudioStream(args.url)
    stream.start()
    buffer = np.zeros(0, dtype=np.float32)
    started = time.time()
    last_events: dict[str, float] = {}

    try:
        while True:
            if args.max_seconds and time.time() - started > args.max_seconds:
                break
            chunk = stream.read(args.hop)
            if chunk.size == 0:
                print("[audio] stream ended", flush=True)
                break
            buffer = np.concatenate([buffer, chunk])[-int(SAMPLE_RATE * args.window):]
            segments, _ = model.transcribe(
                buffer, language="en", beam_size=1, vad_filter=True,
                initial_prompt=PROMPT, condition_on_previous_text=False,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
            if not text:
                continue
            print(f"[heard] {text[-220:]}", flush=True)
            event = parse_trade_speech(text)
            if not event:
                continue
            key = json.dumps(event, sort_keys=True)
            now = time.time()
            if now - last_events.get(key, 0) < 45:
                continue
            last_events[key] = now
            message = format_event(event, args.source_id, args.instrument)
            if not message:
                continue
            print(message, flush=True)
            if webhook and not args.dry_run:
                try:
                    post_discord(webhook, message)
                except Exception as exc:
                    print(f"  discord error: {exc}", flush=True)
            if args.max_seconds and now - started > args.max_seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
    return 0


def format_event(event: dict, source_id: str, instrument: str) -> str | None:
    action = event.get("action")
    name = instrument
    if action == "entry":
        direction = (event.get("direction") or "").upper() or "TRADE"
        qty = f" {event['qty']}" if event.get("qty") else ""
        price = f" @ {event['price']:,.2f}" if event.get("price") else " at market"
        message = f"[{source_id}] **{name} {direction}{qty}{price}**"
        level_bits = []
        if event.get("stop"):
            level_bits.append(f"STOP {event['stop']:,.2f}")
        if event.get("target"):
            level_bits.append(f"LIMIT {event['target']:,.2f}")
        if level_bits:
            message += " | " + " | ".join(level_bits)
        return message + " (heard)"
    if action == "exit":
        return f"[{source_id}] **{name} FLAT** (heard)"
    if action == "stop":
        price = event.get("stop")
        return f"[{source_id}] **{name}** STOP {price:,.2f}" if price else None
    if action == "stop_move":
        price = event.get("stop")
        label = "breakeven" if event.get("breakeven") else "stop moved"
        return f"[{source_id}] **{name}** {label}" + (f" -> {price:,.2f}" if price else "")
    if action == "target":
        targets = event.get("targets") or []
        if not targets:
            return None
        return f"[{source_id}] **{name}** LIMIT -> " + " / ".join(f"{t:,.2f}" for t in targets)
    if action == "scale_in":
        qty = f" {event['qty']}" if event.get("qty") else ""
        return f"[{source_id}] **{name}** scale in:{(event.get('direction') or 'add').upper()}{qty} at market (heard)"
    if action == "scale_out":
        qty = f" {event['qty']}" if event.get("qty") else ""
        return f"[{source_id}] **{name}** scale out{qty} at market (heard)"
    return None


if __name__ == "__main__":
    raise SystemExit(main())
