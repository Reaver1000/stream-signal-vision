import json
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from stream_signal_vision.models import Action, Direction, SignalCandidate
from stream_signal_vision.platforms import Box, load_profiles
from stream_signal_vision.webhook import LocalNodeWebhookSink, format_node_message


ROOT = Path(__file__).resolve().parents[1]


class PlatformAndWebhookTests(unittest.TestCase):
    def test_example_profiles_load_and_are_uncalibrated(self):
        profiles = load_profiles(ROOT / "configs" / "platform_profiles.example.json")
        self.assertEqual(set(profiles), {"topstepx", "tradingview", "tradovate", "wealthcharts", "mt5"})
        self.assertEqual(profiles["tradingview"].regions, {})

    def test_roi_rejects_out_of_bounds(self):
        self.assertTrue(Box(0.1, 0.1, 0.5, 0.5).is_valid())
        self.assertFalse(Box(0.8, 0.1, 0.5, 0.5).is_valid())

    def test_formats_entry_in_node_canonical_dialect(self):
        signal = SignalCandidate(
            event_id="e1", source_id="s1", platform_id="mt5",
            captured_at=datetime.now(timezone.utc), action=Action.ENTRY,
            confidence=0.99, instrument="NQ", direction=Direction.SHORT,
            entry_price=Decimal("30752"), stop_price=Decimal("30802"),
            target_prices=(Decimal("30727"), Decimal("30702")),
        )
        self.assertEqual(
            format_node_message(signal),
            "NQ Shorts @ 30752\nPT: 30727, 30702\nS/L: 30802",
        )

    def test_sink_payload_contains_key_and_canonical_text(self):
        signal = SignalCandidate(
            event_id="e2", source_id="s1", platform_id="mt5",
            captured_at=datetime.now(timezone.utc), action=Action.ENTRY,
            confidence=0.99, instrument="MNQ", direction=Direction.LONG,
            entry_price=Decimal("20000"), stop_price=Decimal("19950"),
        )
        sink = LocalNodeWebhookSink("http://127.0.0.1:8788/", "test-secret")
        payload = json.loads(sink.build_payload(signal))
        self.assertEqual(payload["key"], "test-secret")
        self.assertEqual(payload["event_id"], "e2")
        self.assertIn("MNQ Longs @ 20000", payload["content"])

    def test_sink_rejects_non_local_url(self):
        with self.assertRaises(ValueError):
            LocalNodeWebhookSink("https://example.com/hook", "secret")


if __name__ == "__main__":
    unittest.main()
