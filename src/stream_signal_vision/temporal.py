"""Temporal confirmation and event de-duplication for frame-by-frame detections."""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta, timezone

from .models import SignalCandidate


class StableEventGate:
    """Emit a candidate only after it persists across N observations.

    A stream repeats the same visible order state across many frames. This gate
    confirms the same event_id over consecutive observations and emits it once
    within the dedupe window. A missing/changed candidate resets confirmation.
    """

    def __init__(self, confirmations: int = 2, dedupe_seconds: float = 120.0):
        if confirmations < 1:
            raise ValueError("confirmations must be >= 1")
        if dedupe_seconds < 0:
            raise ValueError("dedupe_seconds must be >= 0")
        self.confirmations = confirmations
        self.dedupe_window = timedelta(seconds=dedupe_seconds)
        self._current_id: str | None = None
        self._count = 0
        self._emitted: OrderedDict[str, datetime] = OrderedDict()

    def observe(
        self,
        candidate: SignalCandidate | None,
        now: datetime | None = None,
    ) -> SignalCandidate | None:
        now = now or datetime.now(timezone.utc)
        self._prune(now)
        if candidate is None:
            self._current_id = None
            self._count = 0
            return None

        if candidate.event_id != self._current_id:
            self._current_id = candidate.event_id
            self._count = 1
        else:
            self._count += 1

        if self._count < self.confirmations:
            return None
        if candidate.event_id in self._emitted:
            return None

        self._emitted[candidate.event_id] = now
        return candidate

    def _prune(self, now: datetime) -> None:
        while self._emitted:
            _, first_seen = next(iter(self._emitted.items()))
            if now - first_seen <= self.dedupe_window:
                break
            self._emitted.popitem(last=False)
