---
license: agpl-3.0
base_model: ultralytics/yolo26n-cls
tags:
- computer-vision
- trading
- finance
- classification
- yolo
- stream-analysis
library_name: ultralytics
---

# Trade State Classifier: Open vs Not-Open Position Detection

A fine-tuned YOLO26n-cls model that classifies livestream trading screen captures as showing an **open position** or **no open position**. Designed for real-time mirroring of trades shown on trading livestreams (TradingView, TopstepX, Tradovate, and similar platforms).

## What it does

Given a screenshot (or video frame) from a futures trading livestream, the model answers one binary question: **is there a filled/current position visible on screen right now?**

```
Input:  1280×720 screenshot from a trading livestream
Output: open (position visible) | not_open (no position)
```

This is the state-gate component of a larger pipeline that:
1. Detects position state transitions (flat → open → flat)
2. Extracts trade details (direction, entry, SL, TP, quantity) at the transition moment
3. Emits validated signals to Discord → NinjaTrader 8 for execution

## Training approach

- **Base model:** YOLO26n-cls (Ultralytics, AGPL-3.0)
- **Fine-tuning data:** ~1,700 owner-verified and confluence-confirmed crops from screen captures the author collected directly, spanning two independent channels and two UI layouts (a TradingView-style and a TopstepX-style interface)
- **Training:** 20 epochs, 512×512, batch 16
- **Labeling pipeline:** Automated Whisper speech-cue extraction → local VLM (qwen2.5vl:7b) keyframe labeling → speech+visual confluence confirmation → dense segment extraction around confirmed trade windows

### Performance

| Metric | Result |
|---|---|
| Temporal validation accuracy | **100%** (341/341) |
| Held-out false positives | **0/165** |
| Entry detection latency | **+2 seconds** after fill |
| Exit detection latency | **+10 seconds** after spoken exit |
| Inference speed | ~15ms/frame on CPU |

## Usage

```python
from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
image = cv2.imread("screenshot.png")
results = model.predict(image, imgsz=512, device="cpu")
class_name = model.names[int(results[0].probs.top1)]
confidence = float(results[0].probs.top1conf)
print(f"{class_name} ({confidence:.3f})")
# "open (0.950)" or "not_open (0.980)"
```

### With ROI cropping

For streamers whose chart occupies a known region, crop first:

```python
roi = (0.0, 0.08, 0.78, 0.82)  # normalized x,y,w,h
h, w = image.shape[:2]
crop = image[int(roi[1]*h):int((roi[1]+roi[3])*h),
             int(roi[0]*w):int((roi[0]+roi[2])*w)]
results = model.predict(crop, imgsz=512, device="cpu")
```

## Important note on test data

This model was trained on real trading-platform screen captures and
learns platform-specific visual features (position badges, PnL overlays, order
tool rendering). It does **not** generalize to simplified synthetic charts or
mock screenshots — it requires real platform UI captures to function correctly.
No test images are included to protect the privacy of everyone involved; see the
GitHub repo for the full evaluation pipeline.

## Limitations

1. **Visual-only:** The model cannot detect a position if the position badge/box is not visible on screen (e.g., the streamer switched to a different chart mid-trade).
2. **Platform-specific appearance:** A position box on TradingView looks different from TopstepX. This model was trained on both but may need additional data for other platforms.
3. **Hypothetical vs filled:** A drawn position-tool box (hypothetical trade plan) looks identical to a filled position. This model detects the visual state; confirming an actual fill requires the full pipeline (speech confluence + state transition logic).
4. **GEX charts:** Gamma exposure screens with green/red bars can produce false positives. Use platform-anchor templates to gate classification to the execution platform screen only.

## Ethics & privacy

- Training data consists of screen captures collected directly by the author, handled with respect for the rights and privacy of everyone involved; no raw frames are distributed
- No raw training images are included in this repository
- The model should not be used to copy trades without the knowledge or consent of the person presenting them
- This model is a research prototype, not financial advice

## License

AGPL-3.0 (inherited from the Ultralytics YOLO base model).

## Links

- [GitHub repository](https://github.com/Reaver1000/stream-signal-vision) — full pipeline: VOD ingestion, Whisper transcription, VLM labeling, confluence curation, training, and live inference
