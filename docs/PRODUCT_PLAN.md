# Product Plan: Stream Signal Vision

**Version:** 0.1 planning baseline
**Date:** 17 October 2026
**Status:** research prototype; no trained detector or live order automation

## 1. Goals, scope, and impact

### 1.1 Problem

Some traders communicate by streaming their platform rather than publishing
explicit text alerts. Viewers must watch the screen, infer when an order was
placed or changed, read the entry/stop/target/quantity, then manually enter it
elsewhere. That process is slow and error-prone. A screen-based observer could
turn visible UI state changes into structured events and hand them to an
existing execution node.

The project recognises **what the trader visibly did**, not whether the trade
is profitable or advisable. It is an observation and automation tool, not a
strategy or recommendation engine.

### 1.2 Refined goals

- **G1 — Capture observable trade events:** detect an entry, stop, target,
  quantity, scale, or exit change from a stream or authorized screen capture.
- **G2 — Platform-aware, not platform-hardcoded:** use a profile per platform
  and layout (TopstepX, TradingView, Tradovate, WealthCharts, MT5 are initial
  research targets) with user calibration for resolution, theme, and panels.
- **G3 — Produce normalized, auditable events:** output a strict event schema
  with timestamp, confidence, evidence references, and extracted levels.
- **G4 — Fail closed:** uncertain, invalid, contradictory, or unstable
  observations produce no executable event; low-confidence cases request
  review rather than guessing.
- **G5 — Reuse the existing NT8 node:** an accepted candidate can be sent to
  its authenticated localhost webhook; position sizing and execution remain
  local to the member's NinjaTrader.
- **G6 — Build evidence of quality:** maintain consented labelled data and
  measure event-level accuracy and latency before enabling automated use.

### 1.3 Scope boundaries

**In scope:** local/authorized frame ingestion, platform profiles and ROI
calibration, change detection, UI-element detection, OCR of visible levels,
event output, optional local NT8 webhook adapter.

**Out of scope for v0.1:** strategy prediction, inferring an unseen order,
live trading, central account routing, generic support for arbitrary layouts
without calibration, and training on third-party streams without permission.

### 1.4 Service objective (proposed, not yet measured)

| Metric | Prototype acceptance target |
|---|---:|
| False executable events on non-event frames | 0 in 1,000-frame holdout |
| Entry side classification | >= 99% on consented holdout events |
| Entry/SL/TP exact tick match | >= 98% per platform profile |
| Event confirmation latency after visible UI change | p95 < 2 seconds, excluding stream transport latency |
| Uncertain observation behavior | reject/review; never guess a price |

These are gates to be validated against data, not advertised performance claims.

## 2. Users and primary workflow

### 2.1 User roles

- **Streamer / data owner:** consents to capture and supplies the platform
  layout and, ideally, an execution log for ground truth.
- **Node operator:** calibrates a profile, chooses a capture source, reviews
  candidates, and controls whether accepted events are forwarded.
- **Developer / maintainer:** labels data, trains or tunes detectors, reviews
  errors, and releases versioned platform profiles.

### 2.2 Primary workflow

1. Operator chooses a platform profile and calibrates chart/order-panel ROIs.
2. Frame source captures authorized video or the operator's own screen.
3. Change detector observes a likely trade-state transition.
4. Platform adapter finds UI objects and OCR reads visible numbers.
5. Temporal gate confirms the event across consecutive frames and deduplicates it.
6. Validator checks direction, stop/target geometry, confidence, and schema.
7. Valid event is displayed/logged; optional keyed local webhook forwards it to
   the NT8 node. Invalid/ambiguous events are rejected or held for review.

## 3. Prioritised delivery (MoSCoW)

### Must have — v0.1 foundation

- Platform-neutral event schema, validator, temporal confirmation, deduplication.
- Platform profile format with normalized ROI coordinates.
- Tests for stop/target geometry, confidence rejection, temporal stability,
  and canonical webhook payload.
- Synthetic examples only; no third-party stream captures committed.
- Design documents, dataset plan, privacy and licensing notes.

### Should have — v0.2 research MVP

- Local screen/stream frame source with explicit operator consent.
- ROI calibration UI or saved profile editor.
- OCR baseline for prices/quantity and geometry/color detector for entry/stop/
  target markers.
- Human-review screen showing before/after frames, candidate fields, and
  confidence before forwarding.
- Replay mode against labelled clips; event-level metrics and error report.

### Could have — v0.3 onward

- Fine-tuned detector for UI elements (not market prediction).
- Optional vision-language model for ambiguous frames, constrained to a strict
  JSON schema and never trusted without deterministic validation.
- Platform-specific adapters for TopstepX, TradingView, Tradovate,
  WealthCharts, MT5.
- Stream transport adapters (e.g. local OBS capture, authorized VOD/stream
  ingestion) with latency reporting.

### Won't have in this product

- A central hosted trade router or custody of member broker credentials.
- Profit guarantees, signal recommendations, or black-box autonomous strategy.
- Unauthenticated remote webhook execution.

## 4. Development and release process

Iterative, evidence-led delivery. A model or adapter is not considered usable
until the platform-specific holdout set passes the acceptance gates above.

**Definition of Done:** code reviewed; unit tests pass; detector evaluation
reports event precision/recall and exact price/side accuracy; no secrets or
third-party captures in Git; dry-run/review mode observed end-to-end; docs and
changelog updated.

**Release stages:**

1. v0.1: schema, profiles, validator, temporal gate, tests (this repository).
2. v0.2: one consenting streamer, one platform, local capture, OCR baseline,
   human review only.
3. v0.3: detector training and session-separated holdout evaluation.
4. v1.0: multiple calibrated profiles, authenticated local NT8 handoff,
   complete operational docs, licensing and support process.

## 5. Key risks and mitigations

| Risk | Likelihood | Impact | Mitigation / evidence required |
|---|---:|---:|---|
| Stream latency makes follower fill materially worse | High | High | Report capture-to-event latency; distinguish platform stream delay from detector latency; validate on intended hold duration |
| UI layout/theme changes break recognition | High | Medium | Per-layout profiles, calibration, change detection, versioned adapters, low-confidence rejection |
| OCR misreads price/quantity | Medium | Critical | Tick-size and stop-side validation; require repeated frame confirmation; review mode; zero-false-event gate |
| Stream overlays obscure the chart/order panel | High | Medium | ROI calibration, detector confidence, reject/review when obstructed |
| False transition / duplicate order event | Medium | Critical | Stable-frame gate, event fingerprint/deduplication, idempotency key at webhook boundary |
| Model hallucinates a missing level | Medium | Critical | Never ask model to invent; missing field => invalid candidate; structured output + deterministic validator |
| Third-party stream capture violates terms/rights | Medium | High | Capture only authorized streams; streamer partnership for commercial product; no third-party captures in repository |
| Model/tool license incompatible with closed-source sale | Medium | High | Prefer permissive components; audit each dependency/model license before bundling; Ultralytics licensing review required |
| Public portfolio repo leaks private data | Low | Critical | `.gitignore` excludes captures, credentials, weights; only synthetic fixtures; inspect staged files before publish |
