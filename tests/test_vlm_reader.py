import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from vlm_reader import assemble


SCALE = {"price_range": [29120.0, 29235.0], "slope": -0.145, "intercept": 29260.0}


class VlmReaderTests(unittest.TestCase):
    def test_sell_brackets_close_a_long(self):
        result = assemble(
            [
                {"type": "limit", "side": "sell", "qty": 1, "price": 29214.75},
                {"type": "stop", "side": "sell", "qty": 2, "price": 29145.75},
            ],
            2,
            SCALE,
        )
        self.assertEqual(result["direction"], "long")
        self.assertEqual(result["stop"], 29145.75)
        self.assertEqual(result["targets"], [29214.75])
        self.assertEqual(result["quantity"], 2)

    def test_buy_brackets_close_a_short(self):
        result = assemble(
            [
                {"type": "limit", "side": "buy", "qty": 3, "price": 29150.0},
                {"type": "stop", "side": "buy", "qty": 3, "price": 29230.0},
            ],
            None,
            SCALE,
        )
        self.assertEqual(result["direction"], "short")
        self.assertEqual(result["stop"], 29230.0)
        self.assertEqual(result["quantity"], 3)

    def test_prices_outside_the_axis_are_rejected(self):
        result = assemble(
            [{"type": "limit", "side": "sell", "qty": 1, "price": 1500.0}],
            None,
            SCALE,
        )
        self.assertEqual(result["targets"], [])
        self.assertIsNone(result["stop"])


if __name__ == "__main__":
    unittest.main()
