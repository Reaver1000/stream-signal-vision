# Co-developer Training Setup

This workflow trains the **visual-only open/not-open classifier** from local,
owner-reviewed video intervals. It is independent of Whisper at inference time.
Whisper can help find candidate windows, but labels must be checked against the
video before training. VODs, extracted frames, interval files, annotations, and
weights stay local; none belong in public GitHub.

## 1. Environment

Python 3.11+ is supported. On a CUDA machine, install the matching CUDA-enabled
PyTorch/torchvision build first from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/),
then install the project extras:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[train-yolo,audio-asr]"
python -c "import torch; print(torch.cuda.is_available())"
```

The audio extra is needed only for Whisper transcription. `yolo26n-cls.pt` and
other base checkpoints are downloaded by Ultralytics on first use; weights are
not stored in the repository.

## 2. Local video and frames

Obtain media only from an authorized source. Put source video and frames under
ignored `data/raw/<session>/` folders. For an existing local VOD, extract a
timestamped 1 fps frame set with ffmpeg, then create its manifest:

```powershell
ffmpeg -hide_banner -loglevel error -ss 00:10:00 -to 00:30:00 `
  -i data/raw/session-a/source.mp4 -vf fps=1 -start_number 1 `
  data/raw/session-a/frames/frame_%05d.png
python scripts/manifest_ffmpeg.py data/raw/session-a --start 00:10:00 `
  --fps 1 --platform tradingview
```

For hour-long VODs, transcribe first and extract only cue windows rather than
writing every frame. This is optional for the visual classifier, but is the
recommended way to build a focused visual review set:

```powershell
python scripts/transcribe_audio.py data/raw/session-a/source.mp4 `
  --out data/raw/session-a/transcript.jsonl --start 00:10:00 --end 00:30:00
python scripts/extract_cue_frames.py data/raw/session-a/source.mp4 `
  data/raw/session-a/transcript.jsonl data/raw/session-a/cue-frames `
  --platform tradingview --session-id session-a --dry-run
python scripts/extract_cue_frames.py data/raw/session-a/source.mp4 `
  data/raw/session-a/transcript.jsonl data/raw/session-a/cue-frames `
  --platform tradingview --session-id session-a
python scripts/label_cue_windows.py data/raw/session-a/cue-frames/frames `
  data/raw/session-a/transcript.jsonl --out data/raw/session-a/cue-visuals.jsonl
```

The frame extractor merges overlapping cues, keeps absolute VOD timestamps in
the manifest, and refuses to mix output with an existing frame directory. Use a
new cue-frame directory for each extraction pass.

## 3. Mark verified state intervals

Copy [`examples/state_intervals.example.jsonl`](../examples/state_intervals.example.jsonl)
to ignored `data/private/state_intervals.jsonl` and replace the synthetic
intervals with source-owner-reviewed times. `source_id` must match a
`--source` argument. Valid states are:

- `open`: visible, filled/current position evidence;
- `pending`: an unfilled order ticket/working order;
- `flat`: no active position;
- `unknown`: ambiguous interval; these frames are deliberately excluded.

`pending` and `flat` map to `not_open` for the current binary classifier. Keep
the underlying interval value so a future multi-state model can separate them.
`split` is `train`, `val`, or `test`; all frames in one session should stay in
one split for a true session-held-out benchmark. The current two-VOD prototype
uses a temporal split and must be described as exploratory, not as independent
session generalization.

## 4. Prepare and train

Supply each session's `frames/` directory. Repeat a source ID for multiple
segments from that same VOD; provide a calibrated chart ROI per layout where
possible. ROI values are normalized `x,y,width,height`:

```powershell
python scripts/prepare_state_classifier.py data/private/state_intervals.jsonl `
  --source session-a=data/raw/session-a/frames `
  --source session-b=data/raw/session-b/frames `
  --source session-c=data/raw/session-c/frames `
  --roi session-a=0.02,0.11,0.69,0.87 `
  --output data/derived/state-prototype

python scripts/train_state_classifier.py data/derived/state-prototype `
  --model yolo26n-cls.pt --epochs 20 --imgsz 512 --batch 16 --device 0
```

The preparer writes `train/`, `val/`, and `external/` (the interval file's
`test` split). The trainer reports temporal validation accuracy and checks the
external negative-only set for false `open` predictions. The final checkpoint
and run logs are in ignored `runs/`.

## 5. Inspect, evaluate, and automate channel onboarding

Score a trained checkpoint across `train`, `val`, or `external` splits and list
any frame-level errors:

```powershell
python scripts/evaluate_state_classifier.py `
  runs/classify/runs/vision/<run>/weights/best.pt `
  data/derived/state-prototype --split val
```

Spot-check specific transition frames:

```powershell
python scripts/predict_trade_state.py runs/classify/runs/vision/<run>/weights/best.pt `
  data/raw/session-a/frames --names frame_00042.png,frame_00043.png `
  --roi 0.02,0.11,0.69,0.87 --device 0
```

To let Python orchestrate back-catalogue VOD discovery, Whisper speech-cue
transcription, sparse cue-frame extraction, local VLM auto-labeling, timeline
generation, and model training/evaluation in one command:

```powershell
# Discover unseen back-catalogue streams from a streamer channel:
python scripts/train_channel.py `
  --channel "https://www.youtube.com/@StreamerHandle/streams" `
  --streamer streamer-a --platform tradingview --dry-run

# Ingest, auto-label around Whisper cues, and train/evaluate on the next unseen VOD:
python scripts/train_channel.py `
  --channel "https://www.youtube.com/@StreamerHandle/streams" `
  --streamer streamer-a --platform tradingview `
  --roi 0.02,0.11,0.69,0.87 --cleanup-media

# Evaluate an existing checkpoint against a held-out VOD session:
python scripts/train_channel.py `
  --session-dir data/raw/streamer-a-vod-02 `
  --streamer streamer-a --platform tradingview `
  --roi 0.02,0.11,0.69,0.87 --mode evaluate `
  --model runs/classify/runs/vision/<run>/weights/best.pt
```

This classifier answers only `open` vs `not_open`. The separate object detector
for `order_ticket`, position badges, entry, stop, and target elements is still a
distinct training task (`--train-detector`). Do not forward predictions to
Discord until held-out sessions, visual level extraction, and event-state
transitions have been reviewed.
