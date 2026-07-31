from __future__ import annotations
"""text_summary must keep a navigation trail after mechanism events.

The sweep's summary was mechanism-events XOR waypoints: from the first
pickup/door/gate onward it carried zero spatial information (29/45 episodes) —
e.g. 31 steps of exploration collapsed to "first you picked up the red key".
Now the chain is mechanism events (chronological) plus the recent movement
trail SINCE the last mechanism event; pre-event behavior is unchanged.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gridworld.task_spec import TaskSpecification
from interface.observation import text_summary_history


SPEC = TaskSpecification.from_dict(
    {
        "task_id": "summary_case",
        "seed": 1,
        "difficulty_tier": 1,
        "maze": {"dimensions": [9, 9], "walls": [], "start": [1, 1], "goal": [7, 7]},
        "mechanisms": {
            "keys": [{"id": "kR", "position": [3, 1], "color": "red"}],
            "doors": [
                {
                    "id": "DR",
                    "position": [5, 1],
                    "requires_key": "red",
                    "initial_state": "locked",
                }
            ],
        },
        "rules": {"observability": "full", "view_size": 7},
        "goal": {"type": "reach_position", "target": [7, 7]},
        "max_steps": 60,
    }
)


def _move(row: int, col: int) -> dict:
    return {
        "kind": "step",
        "event_type": "MOVED",
        "position_after_row_col": [row, col],
        "state_before": {},
        "state_after": {},
    }


def _pickup(key_id: str = "kR") -> dict:
    return {
        "kind": "step",
        "event_type": "PICKUP",
        "position_after_row_col": [1, 3],
        "state_before": {"collected_keys": []},
        "state_after": {"collected_keys": [key_id], "agent_carrying": key_id},
    }


def test_waypoints_only_before_any_mechanism_event():
    transcript = [_move(1, 2), _move(1, 3), _move(2, 3), _move(3, 3)]

    summary = text_summary_history(transcript, SPEC)

    assert "navigated to" in summary
    assert "passed (3, 3)" in summary
    assert "key" not in summary


def test_mechanism_events_keep_the_recent_trail():
    transcript = [
        _move(1, 2),
        _move(1, 3),
        _pickup(),
        _move(2, 3),
        _move(3, 3),
        _move(3, 4),
    ]

    summary = text_summary_history(transcript, SPEC)

    assert "picked up the red key" in summary
    assert "passed (3, 4)" in summary
    assert summary.index("red key") < summary.index("(3, 4)")
    assert "(1, 2)" not in summary


def test_event_with_no_subsequent_moves_is_unchanged():
    transcript = [_move(1, 2), _move(1, 3), _pickup()]

    summary = text_summary_history(transcript, SPEC)

    assert "first you picked up the red key" in summary
    assert "navigated to" not in summary


def test_empty_transcript_message():
    assert "haven't done anything" in text_summary_history([], SPEC)


def _reset(row: int = 1, col: int = 1, facing: str = "EAST") -> dict:
    return {"kind": "reset", "state": {"position_row_col": [row, col], "facing": facing}}


def test_start_pose_prefixes_summary_from_reset():
    # A reset record supplies the start anchor; the trail still follows it.
    transcript = [_reset(1, 1, "EAST"), _move(1, 2), _move(1, 3)]

    summary = text_summary_history(transcript, SPEC)

    assert summary.startswith("You started at (1, 1) facing EAST.")
    assert "navigated to" in summary
    assert summary.index("You started at") < summary.index("Activity summary")


def test_start_pose_shown_even_with_no_moves():
    # Grounding must appear on turn 1 (empty body) so image_only has an anchor.
    summary = text_summary_history([_reset(2, 3, "SOUTH")], SPEC)

    assert "You started at (2, 3) facing SOUTH." in summary
    assert "haven't done anything" in summary


def test_start_pose_falls_back_to_first_step_before_pose():
    # No reset record: use the first step's before-pose instead.
    step = {
        "kind": "step",
        "event_type": "MOVED",
        "position_before_row_col": [4, 5],
        "facing_before": "WEST",
        "position_after_row_col": [4, 6],
        "state_before": {},
        "state_after": {},
    }

    assert "You started at (4, 5) facing WEST." in text_summary_history([step], SPEC)


def test_no_start_line_when_pose_unavailable():
    # Minimal transcript (no reset, no before-pose): omit grounding, do not crash.
    summary = text_summary_history([_move(1, 2)], SPEC)

    assert "You started at" not in summary
    assert "passed (1, 2)" in summary
