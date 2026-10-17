# First Platform Training Guide

The repository does **not** ship trained weights. Training needs representative,
permitted captures from the actual platform layout. This guide starts with one
streamer/layout at a time, then repeats the process for TradingView, TopstepX,

## 0. Permission and privacy gate

Capture only your own screen or a stream for which you have explicit
permission to record/process. Do not place screenshots, VODs, usernames,
account numbers, or provider content in the public GitHub repository. The
`.gitignore` excludes `captures/`, `data/raw/`, `data/private/`, model weights,
and run outputs.

## 1. Capture frames

From the repository root, install capture support and sample the local display
while the platform is visible:

```powershell
pip install -e ".[capture]"
python scripts/capture_screen.py --platform tradingview --session tv-layout-a --duration 300 --fps 1
```

Repeat with the exact platform ids `topstepx`, `tradovate`, and `tradingview`.
The first milestone is **one platform, one layout, one consenting streamer**.
The capture tool writes PNGs and a `frames.jsonl` manifest under the ignored
`captures/raw/` directory. It records screen frames only; it does not access
broker APIs or credentials.

The current tool captures the local display. For remote Twitch/YouTube video,
add a separate `FrameSource` adapter (for example, an authorized local OBS
preview or permitted VOD source). Do not assume a local webhook can receive
TradingView cloud alerts; those originate from TradingView's servers.

## 2. Label UI objects and events

Each JSONL frame record uses normalized bounding boxes `[x, y, width, height]`
from 0 to 1. Label these UI objects:

| Class | What to box |
|---|---|
| `entry_marker` | Visible entry arrow/marker, position tool anchor, or filled-entry marker |
| `stop_line` | Stop-loss line/label (include the readable price label when possible) |
| `target_line` | Each take-profit line/label |
| `quantity_badge` | Visible order/position quantity |
| `position_panel` | Panel region with side, quantity, average entry, or status |

Add the structured event to the `event` field for the transition frame(s):
action, instrument, long/short, entry, stop, target list, displayed quantity.
Include negative frames with `elements: []` (no new trade state) — these are
essential for measuring false positives.

Example annotation record:

```json
{"image_path":"captures/raw/tradingview/tv-layout-a/frame_000042.png","platform_id":"tradingview","session_id":"tv-layout-a-session-01","elements":[{"kind":"entry_marker","bbox":[0.61,0.42,0.03,0.04]},{"kind":"stop_line","bbox":[0.12,0.30,0.76,0.01],"text":"20050.00"}],"event":{"action":"entry","instrument":"MNQ","direction":"short","entry_price":"20000.00","stop_price":"20050.00","target_prices":["19975.00","19950.00"],"displayed_quantity":4}}
```

Ground-truth labels should be checked against the platform's order/position
panel or the streamer's own timestamped trade log — not inferred from the
picture alone.

## 3. Build a session-separated dataset

Use at least **three distinct sessions/VODs** per platform so train, validation,
The exporter keeps all frames from one session in a single split:

```powershell
python scripts/prepare_yolo.py captures/raw/tradingview/annotations.jsonl `
  --root . --output data/derived/tradingview
```

The output is a YOLO-format dataset with `dataset.yaml`. `data/derived/` is
ignored by Git. Inspect the generated split counts and sample labels before
training.

## 4. Train a detector (research backend)

The training wrapper uses Ultralytics as an **optional research backend**:

```powershell
pip install -e ".[train-yolo]"
python scripts/train_detector.py data/derived/tradingview/dataset.yaml `
  --model yolo26n.pt --epochs 100 --imgsz 960 --name tradingview-layout-a
```

Ultralytics licensing must be reviewed before bundling it in a closed-source
commercial product. The model's role is to locate UI elements (markers, lines,
badges), **not** predict trades. OCR (PaddleOCR) reads numeric labels. Keep the
recognizer behind an adapter so an Apache-licensed or in-house model can replace
the research backend if needed.

## 5. Evaluate before connecting execution

Report, separately for each platform/layout:

- object precision/recall per UI class;
- event-level precision/recall/F1;
- direction accuracy;
- exact tick-level entry/SL/TP accuracy;
- quantity accuracy;
- duplicate-event rate;
- capture-to-candidate p50/p95 latency;
- false executable events per hour and abstention rate.

Do not forward to NT8 until the session-separated holdout passes the acceptance
then dry-run; then sim accounts. Never train/test split by random frames.
