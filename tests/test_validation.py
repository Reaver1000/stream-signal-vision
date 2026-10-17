import unittest
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal

from stream_signal_vision.models import Action, Direction, SignalCandidate
from stream_signal_vision.validation import ValidationPolicy, validate_candidate


def entry(*, direction=Direction.SHORT, stop=Decimal("101"), targets=(Decimal("95"), Decimal("90")), confidence=0.99):
    return SignalCandidate(
        event_id="evt-1",
        source_id="local-stream",
        platform_id="tradovate",
        captured_at=datetime.now(timezone.utc),
        action=Action.ENTRY,
        confidence=confidence,
        instrument="MNQ",
        direction=direction,
        entry_price=Decimal("100"),
        stop_price=stop,
        target_prices=targets,
    )


class ValidationTests(unittest.TestCase):
    def test_valid_short_entry_is_accepted(self):
        result = validate_candidate(entry())
        self.assertTrue(result.accepted, result.reasons)

    def test_valid_long_entry_is_accepted(self):
        result = validate_candidate(
            entry(direction=Direction.LONG, stop=Decimal("99"), targets=(Decimal("105"), Decimal("110")))
        )
        self.assertTrue(result.accepted, result.reasons)

    def test_short_stop_on_profit_side_is_rejected(self):
        result = validate_candidate(entry(stop=Decimal("99")))
        self.assertFalse(result.accepted)
        self.assertIn("short stop must be above entry", result.reasons)

    def test_short_target_on_loss_side_is_rejected(self):
        result = validate_candidate(entry(targets=(Decimal("105"),)))
        self.assertFalse(result.accepted)
        self.assertIn("short targets must decrease from entry", result.reasons)

    def test_low_confidence_is_rejected(self):
        result = validate_candidate(entry(confidence=0.7), ValidationPolicy(minimum_confidence=0.95))
        self.assertFalse(result.accepted)
        self.assertIn("confidence below acceptance threshold", result.reasons)

    def test_unknown_instrument_can_be_allowlisted_out(self):
        policy = ValidationPolicy(allowed_instruments=frozenset({"MNQ", "MES"}))
        candidate = entry()
        bad = replace(candidate, instrument="GC")
        result = validate_candidate(bad, policy)
        self.assertFalse(result.accepted)
        self.assertIn("instrument not in allowlist", result.reasons)


if __name__ == "__main__":
    unittest.main()
