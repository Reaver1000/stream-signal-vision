import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_channel import discover_channel_vods, seed_pooled_state_intervals
import gui_app
from train_streams_batch import (
    candidate_is_not_worse,
    defer_vod,
    is_completed_vod,
    load_deferred_ids,
    next_unseen_batch,
    parse_external_confusion,
)
from label_cue_windows import allocate_window_limits
from vlm_label import sample_evenly


class TrainingDiscoveryTests(unittest.TestCase):
    def test_full_page_discovery_omits_playlist_end(self):
        output = "\n".join(
            json.dumps({"id": f"vid-{i}", "url": f"https://www.youtube.com/watch?v=vid-{i}", "title": f"VOD {i}"})
            for i in range(2)
        )
        with patch("train_channel.run_cmd") as run_cmd:
            run_cmd.return_value.stdout = output
            vods = discover_channel_vods("https://www.youtube.com/@example/streams", max_vods=None)
        command = run_cmd.call_args.args[0]
        self.assertNotIn("--playlist-end", command)
        self.assertEqual([v["vod_id"] for v in vods], ["vid-0", "vid-1"])

    def test_paged_discovery_uses_inclusive_five_item_range(self):
        with patch("train_channel.run_cmd") as run_cmd:
            run_cmd.return_value.stdout = ""
            discover_channel_vods("https://www.youtube.com/@example/streams", max_vods=5, playlist_start=6)
        command = run_cmd.call_args.args[0]
        self.assertEqual(command[command.index("--playlist-start") + 1], "6")
        self.assertEqual(command[command.index("--playlist-end") + 1], "10")

    def test_session_local_intervals_seed_pooled_file_without_duplicates(self):
        row = {
            "source_id": "session-1",
            "start": "00:01:00",
            "end": "00:01:20",
            "trade_state": "open",
            "split": "train",
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            local = root / "session-1" / "state_intervals.jsonl"
            local.parent.mkdir()
            local.write_text(json.dumps(row) + "\n", encoding="utf-8")
            pooled = root / "state_intervals.jsonl"
            rows = seed_pooled_state_intervals(root, pooled)
            self.assertEqual(rows, [row])
            self.assertEqual(seed_pooled_state_intervals(root, pooled), [row])

    def test_batch_excludes_live_and_upcoming_entries(self):
        self.assertTrue(gui_app.is_completed_vod({"live_status": "was_live", "duration": 100}))
        self.assertFalse(gui_app.is_completed_vod({"live_status": "is_live", "is_live": True}))
        self.assertFalse(gui_app.is_completed_vod({"live_status": "is_upcoming", "duration": None}))
        self.assertEqual(gui_app.BatchTab.BATCH_SIZE, 5)

    def test_batch_picker_skips_seen_and_non_vod_entries(self):
        vods = [
            {"vod_id": "live1", "live_status": "is_live", "is_live": True},
            {"vod_id": "seen1", "live_status": "was_live", "duration": 20},
            {"vod_id": "new1", "live_status": "was_live", "duration": 30},
            {"vod_id": "new2", "live_status": "was_live", "duration": 40},
        ]
        chosen = next_unseen_batch(vods, {"SEEN1"}, 5)
        self.assertEqual([v["vod_id"] for v in chosen], ["new1", "new2"])

    def test_batch_picker_matches_slugified_ledger_ids(self):
        vods = [
            {"vod_id": "-A9kRAyOvOU", "live_status": "was_live", "duration": 100},
            {"vod_id": "LI3MYdSQwAk", "live_status": "was_live", "duration": 100},
        ]
        chosen = next_unseen_batch(vods, {"a9krayovou"}, 5)
        self.assertEqual([v["vod_id"] for v in chosen], ["LI3MYdSQwAk"])

    def test_vlm_budget_is_spread_over_cue_windows(self):
        allocations = allocate_window_limits(49, 120)
        self.assertEqual(sum(limit for _, limit in allocations), 120)
        self.assertEqual(len(allocations), 49)
        self.assertEqual(sample_evenly(list(range(10)), 4), [0, 3, 6, 9])

    def test_checkpoint_gate_rejects_external_regressions(self):
        baseline = parse_external_confusion(
            "confusion={'open': {'open': 174, 'not_open': 78}, "
            "'not_open': {'open': 0, 'not_open': 165}}"
        )
        candidate = parse_external_confusion(
            "confusion={'open': {'open': 169, 'not_open': 83}, "
            "'not_open': {'open': 1, 'not_open': 164}}"
        )
        self.assertIsNotNone(baseline)
        self.assertIsNotNone(candidate)
        self.assertFalse(candidate_is_not_worse(candidate, baseline))

    def test_failed_vod_is_temporarily_deferred(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            defer_vod(root, "Rips", "Om8u-vmWKkM", "audio download timeout")
            deferred = load_deferred_ids(root, "rips")
            self.assertEqual(deferred, {"om8u-vmwkkm"})
            vods = [
                {"vod_id": "Om8u-vmWKkM", "live_status": "was_live", "duration": 100},
                {"vod_id": "next", "live_status": "was_live", "duration": 100},
            ]
            self.assertEqual([v["vod_id"] for v in next_unseen_batch(vods, set(), 1, deferred)], ["next"])


if __name__ == "__main__":
    unittest.main()
