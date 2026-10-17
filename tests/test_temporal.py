import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from stream_signal_vision.models import Action, Direction, SignalCandidate
from stream_signal_vision.temporal import StableEventGate


def candidate(event_id="trade-1"):
    return SignalCandidate(
        event_id=event_id,
        source_id="stream-1",
        platform_id="topstepx",
        captured_at=datetime(2026, 10, 17, 14, 0, tzinfo=timezone.utc),
        action=Action.ENTRY,
        confidence=0.99,
        instrument="MNQ",
        direction=Direction.SHORT,
        entry_price=Decimal("100"),
        stop_price=Decimal("105"),
        target_prices=(Decimal("95"),),
    )


class TemporalGateTests(unittest.TestCase):
    def test_requires_consecutive_confirmation(self):
        gate = StableEventGate(confirmations=2)
        t = datetime.now(timezone.utc)
        self.assertIsNone(gate.observe(candidate(), t))
        self.assertIsNotNone(gate.observe(candidate(), t + timedelta(milliseconds=250)))

    def test_changed_candidate_resets_confirmation(self):
        gate = StableEventGate(confirmations=2)
        t = datetime.now(timezone.utc)
        self.assertIsNone(gate.observe(candidate("a"), t))
        self.assertIsNone(gate.observe(candidate("b"), t + timedelta(milliseconds=100)))
        self.assertIsNotNone(gate.observe(candidate("b"), t + timedelta(milliseconds=200)))

    def test_same_event_is_emitted_only_once_in_dedupe_window(self):
        gate = StableEventGate(confirmations=1, dedupe_seconds=60)
        t = datetime.now(timezone.utc)
        self.assertIsNotNone(gate.observe(candidate(), t))
        self.assertIsNone(gate.observe(candidate(), t + timedelta(seconds=1)))

    def test_missing_candidate_resets_stability(self):
        gate = StableEventGate(confirmations=2)
        t = datetime.now(timezone.utc)
        self.assertIsNone(gate.observe(candidate(), t))
        self.assertIsNone(gate.observe(None, t + timedelta(milliseconds=100)))
        self.assertIsNone(gate.observe(candidate(), t + timedelta(milliseconds=200)))
        self.assertIsNotNone(gate.observe(candidate(), t + timedelta(milliseconds=300)))


if __name__ == "__main__":
    unittest.main()
