# Architecture and UML

## 1. System context

```mermaid
flowchart LR
    subgraph Input["Authorized visual input"]
        VOD["Streamer VOD / live stream"]
        SCREEN["Operator-owned screen capture"]
    end
    subgraph Vision["Stream Signal Vision - local process"]
        CAP["FrameSource adapter"]
        PROF["PlatformProfile + calibrated ROIs"]
        CHANGE["Frame/change detector"]
        DET["UI-element detector"]
        OCR["OCR / numeric reader"]
        TEMP["Temporal confirmation + dedupe"]
        VALID["Deterministic validator"]
    end
    subgraph Output["Optional output"]
        REVIEW["Review/log UI"]
        HOOK["Keyed localhost webhook"]
    end
    NT["Existing NT8 node\n(local sizing + orders)"]

    VOD --> CAP
    SCREEN --> CAP
    PROF --> CHANGE
    CAP --> CHANGE --> DET --> OCR --> TEMP --> VALID
    VALID -->|uncertain| REVIEW
    VALID -->|accepted| REVIEW
    VALID -->|accepted and enabled| HOOK --> NT
```

## 2. Component responsibilities

| Component | Responsibility | Deliberately does not do |
|---|---|---|
| `FrameSource` | Supplies timestamped frames from an authorized source | Interpret trades |
| `PlatformProfile` | Names UI family/layout and normalized regions | Assume fixed pixel coordinates across streamers |
| `ChangeDetector` | Detects visual changes and chooses frames for deeper inspection | Decide direction/price |
| `PlatformAdapter` | Detects entry/stop/target/quantity UI elements | Place orders |
| `TextRecognizer` | Reads visible price/quantity labels with confidence | Fill in missing values |
| `StableEventGate` | Requires consecutive observations; deduplicates repeat frames | Size positions |
| `validate_candidate` | Enforces event schema, confidence, stop/target geometry | Predict market direction |
| `SignalSink` | Emits validated event to review log or keyed local webhook | Hold broker credentials |
| NT8 node (existing) | Per-account sizing, orders, bracket/reconciliation | Capture streams or host a cloud router |

## 3. Class diagram

```mermaid
classDiagram
    class FramePacket {
        +frame_id: str
        +captured_at: datetime
        +width: int
        +height: int
        +image: Any
    }
    class PlatformProfile {
        +platform_id: str
        +display_name: str
        +layout_id: str
        +reference_width: int
        +reference_height: int
        +regions: dict~str,Box~
        +validate() list~str~
    }
    class Box {
        +x: float
        +y: float
        +width: float
        +height: float
        +is_valid() bool
    }
    class DetectedElement {
        +element_id: str
        +kind: str
        +box: Box
        +confidence: float
        +text: str
        +color_label: str
    }
    class FrameObservation {
        +frame_id: str
        +captured_at: datetime
        +platform_id: str
        +layout_id: str
        +elements: tuple
    }
    class SignalCandidate {
        +event_id: str
        +source_id: str
        +platform_id: str
        +action: Action
        +confidence: float
        +instrument: str
        +direction: Direction
        +entry_price: Decimal
        +stop_price: Decimal
        +target_prices: tuple~Decimal~
        +displayed_quantity: int
        +canonical_dict() dict
    }
    class PlatformAdapter {
        <<Protocol>>
        +observe(frame, profile) Iterable~SignalCandidate~
    }
    class StableEventGate {
        +confirmations: int
        +observe(candidate, now) SignalCandidate
    }
    class ValidationPolicy {
        +minimum_confidence: float
        +allowed_instruments: frozenset
        +maximum_targets: int
    }
    class CandidatePipeline {
        +submit_observation(candidate, now) PipelineResult
    }
    class SignalSink {
        <<Protocol>>
        +publish(candidate)
    }
    class LocalNodeWebhookSink {
        +url: str
        +node_key: str
        +publish(candidate)
    }

    PlatformProfile o-- Box
    FrameObservation o-- DetectedElement
    DetectedElement --> Box
    PlatformAdapter ..> PlatformProfile
    PlatformAdapter ..> FramePacket
    PlatformAdapter ..> SignalCandidate
    CandidatePipeline --> StableEventGate
    CandidatePipeline --> ValidationPolicy
    CandidatePipeline --> SignalSink
    LocalNodeWebhookSink ..|> SignalSink
```

## 4. Sequence: frame transition to candidate event

```mermaid
sequenceDiagram
    autonumber
    participant S as FrameSource
    participant C as ChangeDetector
    participant A as PlatformAdapter
    participant O as OCR
    participant G as StableEventGate
    participant V as Validator
    participant H as HumanReview / SignalSink

    S->>C: FramePacket (timestamp, pixels)
    C->>C: compare ROI to prior frame
    alt no meaningful change
        C-->>S: skip expensive inference
    else possible trade-state change
        C->>A: inspect frame + calibrated PlatformProfile
        A->>A: locate entry/stop/target/quantity UI elements
        A->>O: OCR numeric labels + boxes
        O-->>A: text + per-field confidence
        A-->>G: SignalCandidate (event fingerprint)
        G->>G: require N stable frames; dedupe event_id
        G-->>V: confirmed candidate
        V->>V: confidence + numeric + direction/level checks
        alt valid and confidence sufficient
            V-->>H: accepted candidate + evidence refs
        else uncertain / inconsistent
            V-->>H: reject or request human review; no webhook
        end
    end
```

## 5. State machines

### Candidate transition gate

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> CandidateObserved : candidate fingerprint appears
    CandidateObserved --> CandidateObserved : same fingerprint in next frame
    CandidateObserved --> Idle : missing or different candidate
    CandidateObserved --> Confirmed : stable for N frames
    Confirmed --> Emitted : validation passes and not deduped
    Confirmed --> Rejected : validation fails / confidence low
    Emitted --> Deduped : same event repeats within TTL
    Emitted --> Idle : event expires / next transition
    Rejected --> Idle
```

### Trade event lifecycle (vision side)

```mermaid
stateDiagram-v2
    [*] --> Watching
    Watching --> PossibleChange : ROI difference exceeds threshold
    PossibleChange --> Watching : no stable UI event
    PossibleChange --> Candidate : detector + OCR produce fields
    Candidate --> Review : confidence below auto-accept threshold
    Candidate --> Validated : schema and geometry checks pass
    Candidate --> Rejected : missing/contradictory fields
    Review --> Validated : operator confirms/corrects
    Review --> Rejected : operator rejects
    Validated --> Emitted : webhook enabled
    Validated --> LoggedOnly : dry-run/review mode
    Emitted --> Watching
    LoggedOnly --> Watching
    Rejected --> Watching
```

## 6. Platform adapter strategy

The five named targets are profile families, not a claim of universal
compatibility:

| Profile id | Target | Likely visual anchors (to validate from authorized examples) |
|---|---|---|
| `topstepx` | TopstepX (Topstep platform) | chart/order panel, position quantity, order lines |
| `tradingview` | TradingView chart / broker panel | colored position tool, order labels, price scale |
| `tradovate` | Tradovate | DOM/ladder, working orders, position panel |
| `wealthcharts` | WealthCharts | chart annotations, position/order panel |
| `mt5` | MetaTrader 5 | chart trade levels, terminal trade tab, volume |

Each streamer needs a calibrated profile (ROIs, scaling, theme, stream crop).
Common detection is shared; layout-specific anchoring is not. The example
profile file intentionally has empty regions until a real, permissioned layout
is calibrated.
