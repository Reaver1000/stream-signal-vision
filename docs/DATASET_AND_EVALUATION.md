# Dataset, Annotation, and Evaluation Plan

## Data policy

- Use only streams/screens you own or have explicit permission to capture and
  process. Publicly viewable is not the same as permission to redistribute or
  commercially process.
- No third-party captures, credentials, account data, or provider examples are
  committed to this repository. Public examples are synthetic.
- Keep raw capture data local/private; version only schemas, synthetic fixtures,
  and aggregate evaluation reports with permission.
- If a streamer partnership is pursued, obtain written consent covering capture,
  model training, derived annotations, retention, and any commercial use.

## Annotation unit

Label a **transition**, not a random frame: one before-frame (trade state absent)
and one or more after-frames (trade state present/changed). Split train/validation/
would leak nearly identical images across splits).

Proposed annotation record:

```json
{
  "sample_id": "synthetic-session-01-event-0001",
  "platform_id": "tradingview",
  "layout_id": "dark-1920x1080-crop-a",
  "session_id": "permissioned-session-id",
  "before_frame": "before.png",
  "after_frame": "after.png",
  "event": {
    "action": "entry",
    "instrument": "MNQ",
    "direction": "short",
    "entry_price": "20000.00",
    "stop_price": "20050.00",
    "target_prices": ["19975.00", "19950.00"],
    "displayed_quantity": 4
  },
  "objects": [
    {"kind": "entry_marker", "box": [0.61, 0.42, 0.03, 0.04], "label": "short"},
    {"kind": "stop_line", "box": [0.12, 0.30, 0.76, 0.01], "text": "20050.00"},
    {"kind": "target_line", "box": [0.12, 0.58, 0.76, 0.01], "text": "19975.00"},
    {"kind": "quantity_badge", "box": [0.82, 0.18, 0.08, 0.04], "text": "4"}
  ],
  "consent_reference": "private-record-not-committed",
  "reviewer": "annotator-id"
}
```

Coordinates are normalized [x, y, width, height] fractions of the captured
frame. Prices are strings in labels to preserve decimal precision.

## Evaluation metrics

Evaluate at the **event level**, not only frame/object level:

| Metric | Meaning |
|---|---|
| Event precision / recall / F1 | Did it detect a real entry/change, and avoid false events? |
| Direction accuracy | Long vs short exact match |
| Entry/SL/TP exact tick accuracy | Price matches after platform tick-size normalization |
| Quantity exact accuracy | OCR quantity equals the displayed quantity |
| Confidence calibration | High-confidence predictions are actually more accurate |
| Duplicate event rate | Same visible order change emitted more than once |
| False executable events per hour | Safety-critical false-positive measure |
| Capture-to-candidate latency (p50/p95) | Recognition time, separate from stream transport delay |
| Abstention rate | How often the system correctly refuses uncertain input |

## Proposed acceptance gates (not yet measured)

- Zero false executable events over 1,000 non-event holdout frames.
- At least 99% event precision and at least 98% exact price/side match on a
  session-held-out set before enabling auto-forward.
- Any missing/ambiguous stop, unknown instrument, contradictory direction, or
  below-threshold confidence must abstain (no event sent).
- Use one pooled model, but report results separately per platform/layout; do
  not average away a weak calibration/profile.

## Dataset bootstrapping

1. Obtain streamer permission and a source trade log or timestamped annotations.
2. Capture small clips around known order changes (e.g. ±3 seconds).
3. Generate before/after frame pairs and anonymize overlays/usernames where
   possible.
4. Label UI objects and structured event fields; double-review ambiguous labels.
5. Split by session/VOD; retain a never-touched holdout set.
6. Start with classical ROI/color/geometry + OCR baseline; train a detector only
   where the baseline fails measurably.
