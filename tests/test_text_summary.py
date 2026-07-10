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
                {"id": "DR", "position": [5, 1], "requires_key": "red", "initial_state": "locked"}
            ],
        },
        "rules": {"observability": "full", "view_size": 7},
        "goal": {"type": "reach_position", "target": [7, 7]},
        "max_steps": 60,
    }
)


def _move(row, col):
    return {
        "kind": "step",
        "event_type": "MOVED",
        "position_after": [row, col],
        "state_before": {},
        "state_after": {},
    }


def _pickup(key_id="kR"):
    return {
        "kind": "step",
        "event_type": "PICKUP",
        "position_after": [1, 3],
        "state_before": {"collected_keys": []},
        "state_after": {"collected_keys": [key_id], "agent_carrying": key_id},
    }


def test_waypoints_only_before_any_mechanism_event():
    transcript = [_move(1, 2), _move(1, 3), _move(2, 3), _move(3, 3)]
    s = text_summary_history(transcript, SPEC)
    assert "navigated to" in s and "passed (3, 3)" in s
    assert "key" not in s


def test_mechanism_events_keep_the_recent_trail():
    transcript = [
        _move(1, 2),
        _move(1, 3),
        _pickup(),
        _move(2, 3),
        _move(3, 3),
        _move(3, 4),
    ]
    s = text_summary_history(transcript, SPEC)
    assert "picked up the red key" in s
    # the trail since the pickup must survive, ending at the latest position
    assert "passed (3, 4)" in s
    # chronology: the event comes before the trail
    assert s.index("red key") < s.index("(3, 4)")
    # pre-event wandering is not re-listed once an event anchors the summary
    assert "(1, 2)" not in s


def test_event_with_no_subsequent_moves_is_unchanged():
    transcript = [_move(1, 2), _move(1, 3), _pickup()]
    s = text_summary_history(transcript, SPEC)
    assert "first you picked up the red key" in s
    assert "navigated to" not in s


def test_empty_transcript_message():
    assert "haven't done anything" in text_summary_history([], SPEC)
