"""text_summary must keep a navigation trail after mechanism events."""

from __future__ import annotations

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
        "position_after": [row, col],
        "state_before": {},
        "state_after": {},
    }


def _pickup(key_id: str = "kR") -> dict:
    return {
        "kind": "step",
        "event_type": "PICKUP",
        "position_after": [1, 3],
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
