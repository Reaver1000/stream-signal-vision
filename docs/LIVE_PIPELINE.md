# Live Pipeline (final design)

Point the watcher at any live streamer. It reads the chart, and when the
streamer enters, moves stops, scales or exits, it posts the trade to Discord in
trader language.

```
NQ BUY 2 at market | STOP 29,145.75 | LIMIT 29,214.75
NQ scale in: BUY 2 at market (now 4); STOP 29,145.75 -> 29,165
NQ scale out: SELL 2 at market (now 2)
NQ SELL 4 at market (flat)
```

## How it reads a chart

1. **Find the price axis automatically.** Text-like blobs are clustered into
   vertical columns; each candidate is OCR'd and the column whose numbers form a
   consistent linear scale wins. This works whether the axis is on the right
   (TopstepX, Tradovate) or the left (KeyFlow's TradingView layout).
2. **Read the order tags.** OCR the tag band for the words on each tag
   (`Sell stop`, `Sell limit`, `2 Sell stop`). Prices come from the axis at the
   tag's row, never from the tag glyphs.
3. **Fall back to a local vision model** (`qwen2.5vl:7b`) only when OCR finds
   nothing on a changed frame.
4. **Direction** comes from the bracket side: `sell` brackets close a long,
   `buy` brackets close a short. Geometry (stop above/below entry) is the
   secondary check.
5. **State machine:** new position → entry message; stop moved by at least
   `--stop-step` → stop message; size change → scale in/out message; flat for
   `--flat-confirm` reads → close message. A trade is held through single-frame
   miss-reads.

## Latency

| Stage | Cost |
|---|---|
| preload (once, at start) | 7–23 s |
| OCR read (warm) | 0.5–1.0 s |
| Vision fallback (warm) | 1.5–3.0 s |
| Discord POST | 0.1–0.3 s |

The model is preloaded with `keep_alive` so the steady state stays near the 2 s
target. Unchanged frames are skipped by a mean-difference gate.

## Run

```powershell
# live, auto-detect the layout
python scripts/watch_trades.py `
  --url "https://www.youtube.com/live/XXXX" `
  --instrument NQ --source-id keyflow-live `
  --routes-config "C:\Users\david\Documents\Default Project\stream-signal-vision\data\raw\webhook_routes.json"

# dry run (no Discord posts)
python scripts/watch_trades.py --url "..." --instrument NQ --dry-run

# replay a recorded session
python scripts/watch_trades.py --frames-dir <session>/frames `
  --reader ocr --instrument NQ --stride 15 --max-frames 450 --dry-run
```

Flags that matter: `--reader vlm|ocr` (vlm = OCR then vision fallback),
`--vote-window`, `--flat-confirm`, `--stop-step` (jitter filter),
`--change-threshold`, `--vlm-model`, `--vlm-keep-alive`.

## Verified behaviour

Replaying past KeyFlow sessions produced real trade sequences, including
stop-trailing, e.g.:

```
00:42:37 NQ BUY at market | STOP 29,415.75
00:44:42 NQ STOP 29,415.75 -> 29,418.08
00:51:16 NQ STOP 29,418.08 -> 29,420.32
01:30:54 NQ STOP 29,420.32 -> 29,417.5
```

Single-frame accuracy against the owner's annotated KeyFlow frame:
SL `29,145.64` vs `29,145.75`, TP `29,214.73` vs `29,214.75`, qty `2`, long.

## Known limits

- Quantity comes from the tag text; if the streamer's platform shows size only
  in a panel, qty can be missing.
- Breakeven is reported as a stop move; it is not labelled "BE" yet.
- The vision fallback is slower than the OCR path, so heavy OCR failure rates
  push latency toward 3 s.
