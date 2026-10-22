# Pooled Stream Training Guide

The repository does **not** ship trained weights. Training uses permitted VODs
from the actual streamers. One universal detector is trained on pooled frames
from all supported interfaces; platform profiles only calibrate regions/theme.

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

The first milestone is **one reliable session from each target streamer**. Add
more VOD sessions to improve the same detector and session-held-out evaluation;
do not create a separate model per platform.
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
| `order_ticket` | Order-entry ticket/dialog; presence alone means pending, not filled |

Add per-frame `trade_state` (`open`, `pending`, `flat`, `unknown`) and the visible
instrument, direction, order type, entry, stop, targets, and quantity. Derive
`entry`/`modify`/`exit` from a sequence transition, never from a static order
popup. Include negative frames with `elements: []` — especially working but
unfilled orders — to measure false positives.

Example annotation record:

```json
{"image_path":"captures/raw/combined/session-a/frame_000042.png","platform_id":"tradingview","session_id":"session-a","elements":[{"kind":"entry_marker","bbox":[0.61,0.42,0.03,0.04]},{"kind":"stop_line","bbox":[0.12,0.30,0.76,0.01],"text":"20050.00"}],"event":{"trade_state":"open","action":"entry","instrument":"MNQ","direction":"short","order_type":"market","entry_price":"20000.00","stop_price":"20050.00","target_prices":["19975.00","19950.00"],"displayed_quantity":4}}
```

Ground-truth labels should be checked against the platform's order/position
panel or the streamer's own timestamped trade log — not inferred from the
picture alone.

## 3. Build one pooled, session-separated dataset

Pool annotations from every platform into one class set and one dataset. Keep
each entire VOD/session in a single train/validation/test split; never randomly
split neighboring frames. The two current VODs are an initial discovery set,
not enough evidence to claim robust generalization:

```powershell
python scripts/prepare_yolo.py data/annotations/combined.jsonl `
  --root . --output data/derived/pooled
```

The output is a YOLO-format dataset with `dataset.yaml`. `data/derived/` is
ignored by Git. Inspect the generated split counts and sample labels before
training.

## 4. First visual-state prototype

The initial visual model classifies chart crops as `open` vs `not_open` without
ASR input. `pending` orders map to `not_open` for this classifier; a future
`order_ticket` object detector plus temporal logic will distinguish pending
from flat. Use owner-reviewed state intervals, not VLM action guesses:

```powershell
python scripts/prepare_state_classifier.py data/private/state_intervals.jsonl `
  --source channel-a=data/raw/channel-a/frames `
  --source channel-a=data/raw/channel-a/segment_50_87/frames `
  --source channel-b=data/raw/channel-b/frames `
  --roi channel-a=0.02,0.11,0.69,0.87 --output data/derived/state-prototype
python scripts/train_state_classifier.py data/derived/state-prototype `
  --model yolo26n-cls.pt --epochs 20 --imgsz 512 --batch 16 --device 0
```

This first experiment uses a temporal holdout from one VOD and a separate
owner-confirmed channel-b no-trade interval. Treat its metrics as prototype checks,
not cross-session accuracy. The trained weights stay in ignored `runs/`.
The latest hard-negative-mined run (`v11`) used 529 training crops, 341
temporal-validation crops, and 150 held-out channel-b negative frames (covering both
the Tradovate screen-switch and dual-chart hypothetical-drawing windows); it
reached 100.0% temporal top-1 (`341/341`) and `0/150` false-open predictions on
the held-out negative slice. Owner review corrected the stop-out to 00:39:34
(00:39:17 was a warning); the state interval is labeled open through 00:39:33
and flat from 00:39:34.

## 5. Train the visual object detector (research backend)

The training wrapper uses Ultralytics as an **optional research backend**:

```powershell
pip install -e ".[train-yolo]"
python scripts/train_detector.py data/derived/pooled/dataset.yaml `
  --model yolo26n.pt --epochs 100 --imgsz 960 --name pooled-stream-ui
```

Ultralytics licensing must be reviewed before bundling it in a closed-source
commercial product. The model's role is to locate UI elements (markers, lines,
badges), **not** predict trades. OCR (PaddleOCR) reads numeric labels. Keep the
recognizer behind an adapter so an Apache-licensed or in-house model can replace
the research backend if needed.

## 6. Evaluate before connecting execution

Report overall and broken down by platform/layout (the model remains one pooled
model, but a weak platform must not be hidden by a strong one):

- object precision/recall per UI class;
- event-level precision/recall/F1;
- direction accuracy;
- exact tick-level entry/SL/TP accuracy;
- quantity accuracy;
- duplicate-event rate;
- capture-to-candidate p50/p95 latency;
- false executable events per hour and abstention rate.

Do not post executable events to the Discord relay until session-separated
holdout checks pass; start with review-only dry-run, then sim accounts. Never
train/test split by random frames.
