# Backlog

## Streamers to add

- **Dhesi Trades** — `https://www.youtube.com/live/YzlfAJBPM5A` (live 2026-10-26).
  Chart layout not yet captured; stream was still on the "Starting Soon" slate.
  Next time he goes live: grab a frame, confirm the platform, then run
  `scripts/watch_trades.py --url <live>` in dry-run and compare with his trades.
  No Discord route configured yet (routes only has `nso`).
- **RipsTrades** — 829 completed VODs, layout partially understood
  (TopstepX/TradingView, tags carry qty + P&L, red = stop, green = target).
- **NSO** — TopstepX, already the best-supported layout.

## Pipeline work

- Quantity for TopstepX/Tradovate from their Positions panels (KeyFlow reads it
  from the tag text).
- Breakeven label: needs entry price, which comes from the position badge line.
- Scale in/out messages need a stable qty read across frames.
- Multi-stream supervisor with a shared OCR engine.
- Live Whisper confirmation on a rolling audio buffer (small/base model + VAD).

## Training

- Detail detector (entry/stop/target/quantity/position/order boxes) rebuilt from
  all annotated sessions; superseded the open/not-open classifier, which
  regressed when KeyFlow layouts were pooled in.
