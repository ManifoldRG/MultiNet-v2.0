"""Display-only turn animation timing for the 3D demo (no pygame or mujoco)."""

from __future__ import annotations

from types import SimpleNamespace

from demo.fx import TURN_ANIM_MS, TurnAnimation
from gridworld.backends.base import GridState

TURNING = SimpleNamespace(view_turns_with_agent=True)
FIXED = SimpleNamespace(view_turns_with_agent=False)


def _facing(direction: int) -> GridState:
    return GridState(agent_position=(1, 1), agent_direction=direction)


def test_nothing_to_draw_before_a_turn():
    assert TurnAnimation().frame_request(0) is None


def test_request_runs_from_the_old_heading_then_stops():
    anim = TurnAnimation()
    assert anim.maybe_start(TURNING, _facing(3), _facing(0), now_ms=1000)
    assert anim.frame_request(1000) == (3, 0.0)
    direction, fraction = anim.frame_request(1000 + TURN_ANIM_MS // 2)
    assert direction == 3 and 0.4 < fraction < 0.6
    assert anim.frame_request(1000 + TURN_ANIM_MS) is None


def test_starts_only_when_the_heading_changes_on_a_view_that_turns():
    anim = TurnAnimation()
    assert not anim.maybe_start(TURNING, _facing(0), _facing(0), now_ms=0)  # a step, not a turn
    assert not anim.maybe_start(FIXED, _facing(0), _facing(1), now_ms=0)  # north-up view
    assert not anim.maybe_start(object(), _facing(0), _facing(1), now_ms=0)  # 2D backend
    assert anim.frame_request(0) is None


def test_clear_cancels_a_running_turn():
    anim = TurnAnimation()
    anim.maybe_start(TURNING, _facing(0), _facing(1), now_ms=0)
    anim.clear()
    assert anim.frame_request(1) is None
