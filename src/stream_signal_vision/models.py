"""Platform-neutral data contracts used by detectors, validators, and sinks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"


class Action(str, Enum):
    ENTRY = "entry"
    SCALE_OUT = "scale_out"
    MOVE_STOP = "move_stop"
    FLATTEN = "flatten"
    NO_ACTION = "no_action"


@dataclass(frozen=True, slots=True)
class Box:
    """Normalized ROI (x/y/width/height each in [0, 1])."""

    x: float
    y: float
    width: float
    height: float

    def is_valid(self) -> bool:
        return (
            0 <= self.x < 1
            and 0 <= self.y < 1
            and 0 < self.width <= 1
            and 0 < self.height <= 1
            and self.x + self.width <= 1
            and self.y + self.height <= 1
        )


@dataclass(frozen=True, slots=True)
class DetectedElement:
    """A visual finding before it is assembled into a trade event."""

    element_id: str
    kind: str  # entry_marker, stop_line, target_line, quantity_badge, etc.
    box: Box
    confidence: float
    text: str | None = None
    color_label: str | None = None


@dataclass(frozen=True, slots=True)
class FrameObservation:
    frame_id: str
    captured_at: datetime
    platform_id: str
    layout_id: str
    width: int
    height: int
    elements: tuple[DetectedElement, ...] = ()


@dataclass(frozen=True, slots=True)
class SignalCandidate:
    """Candidate event. It is not executable until validation + temporal gating pass."""

    event_id: str
    source_id: str
    platform_id: str
    captured_at: datetime
    action: Action
    confidence: float
    instrument: str | None = None
    direction: Direction | None = None
    entry_price: Decimal | None = None
    stop_price: Decimal | None = None
    target_prices: tuple[Decimal, ...] = ()
    displayed_quantity: int | None = None
    scale_fraction: Decimal | None = None
    new_stop_price: Decimal | None = None
    evidence_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def canonical_dict(self) -> dict[str, Any]:
        """JSON-safe representation; Decimals become strings to avoid float drift."""
        return {
            "event_id": self.event_id,
            "source_id": self.source_id,
            "platform_id": self.platform_id,
            "captured_at": self.captured_at.isoformat(),
            "action": self.action.value,
            "confidence": self.confidence,
            "instrument": self.instrument,
            "direction": self.direction.value if self.direction else None,
            "entry_price": str(self.entry_price) if self.entry_price is not None else None,
            "stop_price": str(self.stop_price) if self.stop_price is not None else None,
            "target_prices": [str(p) for p in self.target_prices],
            "displayed_quantity": self.displayed_quantity,
            "scale_fraction": str(self.scale_fraction) if self.scale_fraction is not None else None,
            "new_stop_price": str(self.new_stop_price) if self.new_stop_price is not None else None,
            "evidence_ids": list(self.evidence_ids),
            "notes": list(self.notes),
        }
