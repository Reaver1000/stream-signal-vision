# Architecture and UML

## 1. System context

```mermaid
flowchart LR
    subgraph Input["Authorized synchronized input"]
        VIDEO["Streamer live video / VOD"]
        AUDIO["Stream audio / captions"]
        SCREEN["Operator-owned screen capture"]
    end
    subgraph Vision["Stream Signal Vision - local process"]
        CAP["FrameSource adapter"]
        PROF["PlatformProfile + calibrated ROIs"]
        CHANGE["Frame/change detector"]
        DET["UI-element detector"]
        OCR["OCR / numeric reader"]
        ASR["Subtitle adapter / local ASR"]
        CUE["Speech cue parser"]
        TEMP["Audio/video temporal fusion"]
        VALID["Deterministic validator"]
    end
    subgraph Output["Optional output"]
        REVIEW["Review/log UI"]
        DISCORD["Canonical Discord relay post"]
    end
    NT["Existing Discord poller + NT8 node\n(local sizing + orders)"]

    VIDEO --> CAP
    AUDIO --> ASR --> CUE --> TEMP
    SCREEN --> CAP
    PROF --> CHANGE
    CAP --> CHANGE --> DET --> OCR --> TEMP --> VALID
    VALID -->|uncertain| REVIEW
    VALID -->|accepted| REVIEW
    VALID -->|accepted and enabled| DISCORD --> NT
```

## 2. Component responsibilities

| Component | Responsibility | Deliberately does not do |
|---|---|---|
| `FrameSource` | Supplies timestamped frames from an authorized source | Interpret trades |
| `CaptionSource` / `ASR` | Supplies timestamped transcript segments; prefers captions, falls back to local speech recognition | Declare a trade from words alone |
| `SpeechCueParser` | Tags entry/position, stop/target update, and exit utterances | Supply missing prices or quantity |
| `PlatformProfile` | Names UI family/layout and normalized regions | Assume fixed pixel coordinates across streamers |
| `ChangeDetector` | Detects visual changes and chooses frames for deeper inspection | Decide direction/price |
| `PlatformAdapter` | Detects entry/stop/target/quantity UI elements and separates order tickets from filled-position evidence | Place orders |
| `TextRecognizer` | Reads visible price/quantity labels with confidence | Fill in missing values |
| `StableEventGate` | Requires consecutive observations; deduplicates repeat frames | Size positions |
| `TemporalFusion` | Matches audio cues with visual state changes on a shared clock; derives entry/modify/exit transitions | Treat a pending order as a filled position |
| `validate_candidate` | Enforces event schema, confidence, stop/target geometry | Predict market direction |
| `SignalSink` | Emits validated canonical event to review log or Discord relay channel | Hold broker credentials or submit orders |
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
    class TranscriptSegment {
        +start_at: datetime
        +end_at: datetime
        +text: str
        +asr_confidence: float
    }
    class SpeechCue {
        +cue_type: CueType
        +direction: Direction
        +confidence: float
    }
    class SignalCandidate {
        +event_id: str
        +source_id: str
        +platform_id: str
        +action: Action
        +trade_state: TradeState
        +confidence: float
        +instrument: str
        +direction: Direction
        +order_type: OrderType
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
    class TemporalFusion {
        +submit_frame_observation(observation)
        +submit_speech_cue(cue)
        +derive_state_transition() SignalCandidate
    }
    class SignalSink {
        <<Protocol>>
        +publish(candidate)
    }
    class DiscordRelaySink {
        +channel_id: str
        +publish(candidate)
    }
    class ExistingDiscordPoller {
        +publish(candidate)
    }

    PlatformProfile o-- Box
    FrameObservation o-- DetectedElement
    DetectedElement --> Box
    PlatformAdapter ..> PlatformProfile
    PlatformAdapter ..> FramePacket
    PlatformAdapter ..> SignalCandidate
    TranscriptSegment --> SpeechCue
    SpeechCue --> TemporalFusion
    FrameObservation --> TemporalFusion
    TemporalFusion --> CandidatePipeline
    CandidatePipeline --> StableEventGate
    CandidatePipeline --> ValidationPolicy
    CandidatePipeline --> SignalSink
    DiscordRelaySink ..|> SignalSink
    DiscordRelaySink --> ExistingDiscordPoller
```

## 4. Sequence: frame transition to candidate event

```mermaid
sequenceDiagram
    autonumber
    participant S as FrameSource
    participant A as Captions / ASR
    participant P as SpeechCueParser
    participant C as ChangeDetector
    participant D as PlatformAdapter
    participant O as OCR
    participant F as TemporalFusion / StateMachine
    participant V as Validator
    participant H as Review / DiscordRelaySink
    participant N as Existing NT8 Poller

    A->>P: TranscriptSegment (timestamp, text)
    P->>F: SpeechCue (position/entry/modify/exit claim)
    S->>C: FramePacket (timestamp, pixels; continuous video)
    C->>C: compare ROI to prior frame
    alt no meaningful change
        C-->>S: skip expensive inference
    else possible trade-state change
        C->>D: inspect frame + universal classes / calibrated regions
        D->>D: classify state: open / pending / flat / unknown
        D->>O: OCR numeric labels + boxes
        O-->>D: text + per-field confidence
        D-->>F: FrameObservation (state + elements + values)
        F->>F: match speech cue; derive state/level transition; dedupe
        F-->>V: candidate (open only with visual fill evidence)
        V->>V: confidence + numeric + direction/level checks
        alt valid and confidence sufficient
            V-->>H: accepted canonical event + evidence refs
            H->>N: post to Discord relay channel
            N->>N: existing relay executes via NT8
        else uncertain / inconsistent
            V-->>H: reject or request human review; no Discord post
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
    Validated --> Emitted : Discord relay enabled
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
