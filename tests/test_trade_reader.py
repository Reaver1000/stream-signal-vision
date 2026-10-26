import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from trade_reader import Marker, _assemble, dottedness, parse_pill_text


class TradeReaderTests(unittest.TestCase):
    def test_pill_text_separates_quantity_from_pnl(self):
        qty, pnl = parse_pill_text("1 +827.00 1 X")
        self.assertEqual(qty, 1)
        self.assertEqual(pnl, 827.0)

    def test_stop_above_entry_is_short(self):
        state = type("S", (), {})()
        state.in_trade = False
        state.instrument = None
        state.direction = None
        state.quantity = None
        state.entry = None
        state.stop = None
        state.targets = []
        state.markers = []
        state.scale = None
        state.confidence = {}
        state.notes = []
        markers = [
            Marker(color="green", y=100, x0=0, x1=10, price=29831.0, solid=True, qty=2),
            Marker(color="red", y=60, x0=0, x1=10, price=29934.0, solid=False, tag_color="red"),
            Marker(color="green", y=140, x0=0, x1=10, price=29680.0, solid=False, tag_color="green"),
        ]
        _assemble(state, markers)
        self.assertEqual(state.direction, "short")
        self.assertEqual(state.entry, 29831.0)
        self.assertEqual(state.stop, 29934.0)
        self.assertEqual(state.targets, [29680.0])

    def test_solid_line_has_low_gap_fraction(self):
        mask = np.zeros((10, 200), dtype=np.uint8)
        mask[5, :] = 255
        self.assertLess(dottedness(mask, 5, 0, 200), 0.2)
        dotted = np.zeros((10, 200), dtype=np.uint8)
        dotted[5, ::12] = 255
        self.assertGreater(dottedness(dotted, 5, 0, 200), 0.4)


if __name__ == "__main__":
    unittest.main()
