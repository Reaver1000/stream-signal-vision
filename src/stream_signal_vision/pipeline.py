"""Detector-independent candidate pipeline.

Platform-specific capture/detection adapters produce SignalCandidate objects;
this module confirms, validates, and forwards them. The vision models are
intentionally behind protocols so OCR/detector implementations can evolve
without changing risk validation or downstream integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from .models import SignalCandidate
from .temporal import StableEventGate
from .validation import ValidationPolicy, ValidationResult, validate_candidate


class SignalSink(Protocol):
    def publish(self, candidate: SignalCandidate) -> None: ...


@dataclass(frozen=True, slots=True)
class PipelineResult:
    validation: ValidationResult
    confirmed: bool
    published: bool


class CandidatePipeline:
    """Safe, testable boundary from candidate detection to signal output."""

    def __init__(
        self,
        sink: SignalSink,
        policy: ValidationPolicy | None = None,
        gate: StableEventGate | None = None,
    ) -> None:
        self.sink = sink
        self.policy = policy or ValidationPolicy()
        self.gate = gate or StableEventGate()

    def submit_observation(
        self,
        candidate: SignalCandidate | None,
        now: datetime | None = None,
    ) -> PipelineResult:
        if candidate is None:
            self.gate.observe(None, now)
            return PipelineResult(ValidationResult(False, ("no candidate",)), False, False)

        validation = validate_candidate(candidate, self.policy)
        if not validation.accepted:
            self.gate.observe(None, now)
            return PipelineResult(validation, False, False)

        confirmed = self.gate.observe(candidate, now)
        if confirmed is None:
            return PipelineResult(validation, False, False)

        self.sink.publish(confirmed)
        return PipelineResult(validation, True, True)
