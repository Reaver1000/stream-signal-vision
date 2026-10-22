import unittest
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal

from stream_signal_vision.models import Action, Direction, SignalCandidate
from stream_signal_vision.pipeline import CandidatePipeline
from stream_signal_vision.temporal import StableEventGate
from stream_signal_vision.validation import ValidationPolicy


class RecordingSink:
    def __init__(self):
        self.events = []

    def publish(self, candidate):
        self.events.append(candidate)


def valid_candidate(event_id="e1"):
    return SignalCandidate(
        event_id=event_id,
        source_id="screen-capture-01",
        platform_id="tradingview",
        captured_at=datetime.now(timezone.utc),
        action=Action.ENTRY,
        confidence=0.99,
        instrument="NQ",
        direction=Direction.SHORT,
        entry_price=Decimal("20000"),
        stop_price=Decimal("20050"),
        target_prices=(Decimal("19975"), Decimal("19950")),
        evidence_ids=("frame-101", "frame-102"),
    )


class CandidatePipelineTests(unittest.TestCase):
    def test_publishes_only_after_valid_stable_candidate(self):
        sink = RecordingSink()
        pipeline = CandidatePipeline(
            sink,
            policy=ValidationPolicy(minimum_confidence=0.9),
            gate=StableEventGate(confirmations=2),
        )
        self.assertFalse(pipeline.submit_observation(valid_candidate()).published)
        result = pipeline.submit_observation(valid_candidate())
        self.assertTrue(result.published)
        self.assertEqual(len(sink.events), 1)

    def test_invalid_candidate_never_reaches_sink(self):
        sink = RecordingSink()
        pipeline = CandidatePipeline(sink, gate=StableEventGate(confirmations=1))
        bad = replace(valid_candidate(), stop_price=Decimal("19990"))
        result = pipeline.submit_observation(bad)
        self.assertFalse(result.validation.accepted)
        self.assertEqual(sink.events, [])

    def test_low_confidence_candidate_never_reaches_sink(self):
        sink = RecordingSink()
        pipeline = CandidatePipeline(sink, gate=StableEventGate(confirmations=1))
        bad = replace(valid_candidate(), confidence=0.4)
        result = pipeline.submit_observation(bad)
        self.assertFalse(result.published)
        self.assertEqual(sink.events, [])


if __name__ == "__main__":
    unittest.main()
