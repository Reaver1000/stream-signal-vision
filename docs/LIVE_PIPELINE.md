# Live Pipeline Design

Target: one or more **live** streamer feeds → when a real order/trade appears,
push the details to Discord within ~2 seconds, then keep tracking changes.

This is the runtime contract. The VOD/training tooling exists only to tune the
readers and the model that this pipeline uses.

## Two independent signals

| Signal | Source | Strength | Weakness |
|---|---|---|---|
| **Vision** | HLS frames (2–4 fps) at native resolution | proves *what is on screen*: side, qty, entry, SL/TP lines | needs layout profiles; small text |
| **Audio** | rolling buffer → faster-whisper (small/base, VAD) | proves *intent and narration*: "I'm long 5", "stopped out", "adding" | speech lags/leads the fill; may be wrong or joking |

Vision is authoritative for a **fill**. Audio never creates a trade on its own;
it confirms, disambiguates, and catches things the visuals miss.

## Fusion rules

1. **Entry**: require a visual position marker (line + attached pill, or
   Positions row). Audio raises confidence; absence of audio does not block.
2. **Exit / flat**: visual disappearance of the position marker *or* an
   explicit audio exit ("out", "flat", "stopped out") — visual wins if they
   disagree for more than one second; then re-check the next frames.
3. **Modification / scale-in**: any change in qty or SL/TP price emits a **new**
   message (e.g. `buy 5 market`, `SL → 29850`). Audio speeds this up.
4. **Disagreement**: hold the event, re-read 2–3 frames, and only emit if the
   visual evidence persists. Never emit on audio alone.

## Latency budget (2 s)

```
frame capture        ~50–100 ms   (HLS segment already buffered)
chart read (CV)      ~80–200 ms   (OpenCV geometry; OCR only on change)
OCR (axis/pill)      ~200–500 ms  (only when a marker appears/changes)
state machine        <5 ms
Discord POST         ~100–300 ms
audio confirm (opt.) ~0.5–2 s     (runs in parallel; never blocks the push)
```

Rule: **do not wait for OCR or audio to emit.** Emit on geometry + axis
calibration, then follow with an edit or a correction only if a later read
contradicts it (rare).

## Multi-stream

- One worker per stream URL, each with its own crop profile.
- Shared OCR engine, serialised behind a semaphore (RapidOCR is not thread-safe).
- Per-streamer profile: chart ROI, axis side (KeyFlow = left), tag region,
  qty region (TopstepX positions table, Tradovate positions row, KeyFlow badge).
- Per-streamer Discord route via `webhook_routes.json`.

## What is already built

- `scripts/chart_lines.py` — coloured line + attached-pill detection
- `scripts/price_axis.py` — axis OCR → robust y→price calibration
- `scripts/trade_reader.py` — entry/stop/target/direction assembly
- `scripts/transcribe_audio.py` — cue classification reused for live chunks
- `scripts/watch_live.py` — HLS intake, hysteresis, per-streamer routing

## Not yet built

1. Per-platform profiles (ROIs, axis side, tag/qty regions)
2. Quantity from position panels/badges
3. State machine emitting on change (including scale-in as a new message)
4. Live audio loop (rolling Whisper buffer) feeding the fusion rules
5. Multi-stream supervisor with shared OCR
