import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from transcribe_audio import classify_cue
from label_cue_windows import merge_windows
from extract_cue_frames import format_timestamp, merge_intervals, parse_timestamp


def test_negated_long_does_not_override_short_position_claim():
    assert classify_cue("I'm not, I'm not, uh, long bro, I'm short") == (
        "position_claim",
        "short",
    )


def test_order_panel_language_is_not_a_fill_claim():
    assert classify_cue("The limit sell order is still working") == (
        "order_intent",
        "short",
    )


def test_stopped_out_is_exit_claim():
    assert classify_cue("Yeah, I stopped out") == ("exit_claim", None)


def test_may_stop_me_out_is_warning_not_exit_claim():
    assert classify_cue("It may go and stop me out") == ("exit_warning", None)


def test_stop_discussion_is_only_a_cue():
    assert classify_cue("We should consider moving the stop to breakeven") == (
        "stop_update",
        None,
    )


def test_short_setup_phrase_is_order_intent_not_position_claim():
    assert classify_cue("Short off the CISD, if we can get a CISD here") == (
        "order_intent",
        "short",
    )


def test_try_it_phrase_prioritizes_visual_check():
    assert classify_cue("I'm gonna try it, guys, 11 point stop") == (
        "order_intent",
        None,
    )


def test_rejected_short_idea_is_not_order_intent():
    assert classify_cue("I wouldn't try shorting it out of the same gap") == (
        "none",
        None,
    )


def test_short_trade_comment_is_position_claim_not_entry_intent():
    assert classify_cue("Out this short trade, we have almost a 3R") == (
        "position_claim",
        "short",
    )


def test_rejected_later_short_setup_is_not_order_intent():
    assert classify_cue("Nowhere else that I would like to take a short from") == (
        "none",
        "short",
    )


def test_overlapping_cue_windows_merge_and_keep_both_cues():
    merged = merge_windows([
        (10, 20, {"cue": "position_claim"}),
        (18, 30, {"cue": "stop_update"}),
        (40, 50, {"cue": "exit_claim"}),
    ])
    assert merged == [
        [10, 30, [{"cue": "position_claim"}, {"cue": "stop_update"}]],
        [40, 50, [{"cue": "exit_claim"}]],
    ]


def test_cue_frame_windows_merge_and_timestamp_roundtrip():
    merged = merge_intervals([
        (600, 620, [{"cue": "position_claim"}]),
        (618, 635, [{"cue": "stop_update"}]),
        (700, 710, [{"cue": "exit_claim"}]),
    ], gap_seconds=3)
    assert merged == [
        [600, 635, [{"cue": "position_claim"}, {"cue": "stop_update"}]],
        [700, 710, [{"cue": "exit_claim"}]],
    ]
    assert parse_timestamp("01:27:44") == 5264
    assert format_timestamp(5264) == "01:27:44"
