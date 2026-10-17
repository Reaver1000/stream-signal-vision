"""Deterministic gate between visual interpretation and downstream action."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math

from .models import Action, Direction, SignalCandidate


@dataclass(frozen=True, slots=True)
class ValidationResult:
    accepted: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ValidationPolicy:
    minimum_confidence: float = 0.95
    allowed_instruments: frozenset[str] = frozenset()
    maximum_targets: int = 5


def validate_candidate(
    candidate: SignalCandidate,
    policy: ValidationPolicy = ValidationPolicy(),
) -> ValidationResult:
    errors: list[str] = []
    if not math.isfinite(candidate.confidence) or not 0 <= candidate.confidence <= 1:
        errors.append("confidence must be finite and between 0 and 1")
    elif candidate.confidence < policy.minimum_confidence:
        errors.append("confidence below acceptance threshold")

    if candidate.action == Action.ENTRY:
        if not candidate.instrument:
            errors.append("entry missing instrument")
        elif policy.allowed_instruments and candidate.instrument not in policy.allowed_instruments:
            errors.append("instrument not in allowlist")
        if candidate.direction is None:
            errors.append("entry missing direction")
        if not _positive(candidate.entry_price):
            errors.append("entry price missing or non-positive")
        if not _positive(candidate.stop_price):
            errors.append("stop price missing or non-positive")
        if candidate.displayed_quantity is not None and candidate.displayed_quantity <= 0:
            errors.append("displayed quantity must be positive when present")
        if candidate.entry_price is not None and candidate.stop_price is not None and candidate.direction:
            if candidate.direction == Direction.LONG and candidate.stop_price >= candidate.entry_price:
                errors.append("long stop must be below entry")
            if candidate.direction == Direction.SHORT and candidate.stop_price <= candidate.entry_price:
                errors.append("short stop must be above entry")
        if len(candidate.target_prices) > policy.maximum_targets:
            errors.append("target count exceeds configured maximum")
        if candidate.entry_price is not None and candidate.direction:
            previous = candidate.entry_price
            for target in candidate.target_prices:
                if not _positive(target):
                    errors.append("target prices must be positive")
                    break
                if candidate.direction == Direction.LONG and target <= previous:
                    errors.append("long targets must increase from entry")
                    break
                if candidate.direction == Direction.SHORT and target >= previous:
                    errors.append("short targets must decrease from entry")
                    break
                previous = target

    elif candidate.action == Action.SCALE_OUT:
        if candidate.scale_fraction is None or not (
            Decimal("0") < candidate.scale_fraction <= Decimal("1")
        ):
            errors.append("scale-out fraction must be in (0, 1]")
    elif candidate.action == Action.MOVE_STOP:
        if not _positive(candidate.new_stop_price):
            errors.append("stop move missing a positive new stop price")

    return ValidationResult(accepted=not errors, reasons=tuple(errors))


def _positive(value: Decimal | None) -> bool:
    return value is not None and value.is_finite() and value > 0
