import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from stream_signal_vision.dataset import CLASS_IDS, DatasetError, split_by_session, validate_record, yolo_label_lines


def record(session: str, name: str) -> dict:
    return {
        "image_path": name,
        "platform_id": "tradingview",
        "session_id": session,
        "elements": [{"kind": "stop_line", "bbox": [0.1, 0.2, 0.7, 0.01]}],
    }


class DatasetTests(unittest.TestCase):
    def test_order_ticket_is_a_distinct_visual_class(self):
        self.assertIn("order_ticket", CLASS_IDS)

    def test_session_split_never_leaks_a_session(self):
        rows = [record(f"session-{i}", f"frame-{i}.png") for i in range(12)]
        splits = split_by_session(rows, seed=7)
        owner = {}
        for split, split_rows in splits.items():
            for row in split_rows:
                previous = owner.setdefault(row["session_id"], split)
                self.assertEqual(previous, split)
        self.assertEqual(sum(map(len, splits.values())), len(rows))

    def test_requires_three_sessions(self):
        with self.assertRaises(DatasetError):
            split_by_session([record("one", "a.png"), record("two", "b.png")])

    def test_record_requires_image_inside_root(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "frame.png").write_bytes(b"synthetic")
            self.assertEqual(validate_record(record("s1", "frame.png"), root), [])
            outside = record("s1", "../escape.png")
            self.assertTrue(validate_record(outside, root))

    def test_yolo_label_conversion(self):
        line = yolo_label_lines(record("s1", "frame.png"))[0]
        self.assertEqual(line, "1 0.450000 0.205000 0.700000 0.010000")


if __name__ == "__main__":
    unittest.main()
