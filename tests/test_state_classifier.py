import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_state_classifier import interval_for, parse_timestamp, should_sample
from train_channel import (
    derive_state_intervals_from_annotations,
    ensure_three_session_ids_for_bootstrap,
    select_unseen_vods,
)
from watch_live import StateHysteresis


def test_interval_lookup_respects_source_and_inclusive_bounds():
    intervals = [
        {"source_id": "kf", "start_sec": 10, "end_sec": 20, "trade_state": "open"},
        {"source_id": "channel-b", "start_sec": 10, "end_sec": 20, "trade_state": "flat"},
    ]
    assert interval_for("kf", 20, intervals)["trade_state"] == "open"
    assert interval_for("channel-b", 15, intervals)["trade_state"] == "flat"
    assert interval_for("kf", 21, intervals) is None


def test_video_timestamp_and_boundary_sampling():
    assert parse_timestamp("01:27:44") == 5264
    interval = {"start_sec": 100, "end_sec": 200}
    assert should_sample(101, interval, stride=10, edge_window=2)
    assert should_sample(200, interval, stride=10, edge_window=2)
    assert not should_sample(105, interval, stride=10, edge_window=2)


def test_select_unseen_vods_skips_already_processed_ids():
    catalogue = [
        {"vod_id": "v1", "url": "https://youtu.be/v1", "title": "One"},
        {"vod_id": "v2", "url": "https://youtu.be/v2", "title": "Two"},
        {"vod_id": "v3", "url": "https://youtu.be/v3", "title": "Three"},
    ]
    unseen = select_unseen_vods(catalogue, {"v1"}, count=2)
    assert [v["vod_id"] for v in unseen] == ["v2", "v3"]


def test_derive_state_intervals_treats_ticket_only_as_pending():
    manifest = [
        {"image_path": "f1.png", "video_timestamp": "00:01:00"},
        {"image_path": "f2.png", "video_timestamp": "00:01:01"},
        {"image_path": "f3.png", "video_timestamp": "00:01:10"},
    ]
    annotations = [
        {
            "image_path": "f1.png",
            "elements": [{"kind": "order_ticket", "bbox": [0.4, 0.2, 0.2, 0.5]}],
            "event": {"trade_state": "open"},
        },
        {
            "image_path": "f2.png",
            "elements": [{"kind": "order_ticket", "bbox": [0.4, 0.2, 0.2, 0.5]}],
            "event": {"trade_state": "pending"},
        },
        {
            "image_path": "f3.png",
            "elements": [{"kind": "position_panel", "bbox": [0.1, 0.2, 0.1, 0.05]}],
            "event": {"trade_state": "open"},
        },
    ]
    intervals = derive_state_intervals_from_annotations(annotations, manifest, source_id="kf")
    assert len(intervals) == 2
    assert intervals[0]["trade_state"] == "pending"
    assert intervals[0]["start"] == "00:01:00"
    assert intervals[0]["end"] == "00:01:01"
    assert intervals[1]["trade_state"] == "open"


def test_bootstrap_session_partition_creates_three_chronological_groups():
    rows = [{"session_id": "s1", "image_path": f"frame_{i:03d}.png"} for i in range(6)]
    partitioned = ensure_three_session_ids_for_bootstrap(rows)
    assert len({r["session_id"] for r in partitioned}) == 3


def test_hysteresis_requires_consecutive_confirmation_and_survives_blips():
    gate = StateHysteresis(threshold=0.85, open_confirm=2, close_confirm=3)
    assert gate.update(0.90, 1) is None
    opened = gate.update(0.90, 2)
    assert opened is not None and opened["event"] == "position_open"
    assert gate.update(0.20, 3) is None
    assert gate.update(0.20, 4) is None
    assert gate.update(0.90, 5) is None
    assert gate.update(0.10, 6) is None
    assert gate.update(0.10, 7) is None
    closed = gate.update(0.10, 8)
    assert closed is not None and closed["event"] == "position_closed"


def test_hysteresis_resets_confirmation_runs_on_long_gap():
    gate = StateHysteresis(threshold=0.85, open_confirm=2, close_confirm=2, gap_seconds=30)
    assert gate.update(0.90, 0) is None
    assert gate.update(0.90, 100) is None
    opened = gate.update(0.90, 101)
    assert opened is not None and opened["event"] == "position_open"


def test_vlm_open_clusters_merge_by_gap():
    from confluence_windows import vlm_open_clusters
    annotations = [
        {"image_path": "f1.png", "event": {"trade_state": "open"}},
        {"image_path": "f2.png", "event": {"trade_state": "open"}},
        {"image_path": "f3.png", "event": {"trade_state": "flat"}},
        {"image_path": "f4.png", "event": {"trade_state": "open"}},
    ]
    manifest = [
        {"image_path": "f1.png", "video_timestamp": "00:01:00"},
        {"image_path": "f2.png", "video_timestamp": "00:02:30"},
        {"image_path": "f3.png", "video_timestamp": "00:03:00"},
        {"image_path": "f4.png", "video_timestamp": "00:45:00"},
    ]
    clusters = vlm_open_clusters(annotations, manifest, cluster_gap_s=180.0)
    assert len(clusters) == 2
    assert clusters[0]["start"] == 60 and clusters[0]["end"] == 150
    assert clusters[0]["frames"] == [60.0, 150.0]
    assert clusters[1]["frames"] == [2700.0]


def test_window_proposals_require_multi_open_exit_and_intrade_speech():
    from confluence_windows import detect_window_proposals
    clusters = [
        {"start": 100, "end": 100, "frames": [100]},
        {"start": 200, "end": 260, "frames": [200, 260]},
        {"start": 600, "end": 700, "frames": [600, 700]},
    ]
    transcript = [
        {"start_sec": 650, "end_sec": 651, "cue": "exit_warning", "text": "might get stopped out"},
        {"start_sec": 900, "end_sec": 901, "cue": "exit_claim", "text": "I am stopped out"},
    ]
    windows, rejected = detect_window_proposals(clusters, transcript)
    assert len(windows) == 1
    assert windows[0]["start"] == 600 and windows[0]["end"] == 900
    assert {r["reason"] for r in rejected} == {
        "fewer than min_cluster VLM open frames",
        "no spoken exit within window",
    }
