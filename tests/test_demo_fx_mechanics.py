"""Demo warp / kill effects follow the world model's mechanic tag.

The play demo animates a portal warp or a death reset (visual + sound). It
must show what the rulebook actually did on this dispatch, not guess from the
agent's facing: on a turntable MOVE_FORWARD rides the arrow, and on ice a
MOVE_FORWARD is swallowed.
"""

from __future__ import annotations

import dataclasses
import json

from demo.fx import KILL_MS, WARP_MS, plan_effects
from demo.r1_config import R1_CONFIG
from demo.session import MiniGridPlaySession
from demo.sounds import sfx_for_dispatch

PORTAL = {
    "id": "portal_purple",
    "position_a": [4, 1],
    "position_b": [6, 3],
    "color": "purple",
    "bidirectional": True,
}


def _session(tmp_path, mechanisms: dict, goal=(7, 1), **config) -> MiniGridPlaySession:
    spec = {
        "task_id": "demo_fx_mechanics",
        "seed": 0,
        "difficulty_tier": 1,
        "maze": {"dimensions": [9, 5], "walls": [], "start": [1, 1], "goal": list(goal)},
        "mechanisms": mechanisms,
        "goal": {"type": "reach_position", "target": list(goal)},
        "max_steps": 60,
    }
    path = tmp_path / "maze.json"
    path.write_text(json.dumps(spec))
    return MiniGridPlaySession(
        task_path=str(path), config=dataclasses.replace(R1_CONFIG, **config)
    )


def _dispatch(session, token: str):
    """Dispatch like demo/ui.py and demo/api/app.py; return (fx kinds, sfx)."""
    events_before = len(session.event_log)
    prev_state = session.state
    session._dispatch_token(token)
    plan = plan_effects(session, token, prev_state, events_before)
    return plan, sfx_for_dispatch(session, events_before, prev_state)


def _portal_effects(plan):
    return [item for item in plan if item["kind"] in ("warp", "kill")]


def _ride_session(tmp_path, **extra):
    # Turntable at (2, 1) points EAST and spins clockwise every step: after the
    # agent boards it (still facing EAST) the arrow points SOUTH, so the next
    # MOVE_FORWARD rides SOUTH to (2, 2) while the agent keeps facing EAST.
    return _session(
        tmp_path, {"rotating_tiles": [[2, 1]], "rotating_initial_directions": [0], **extra}
    )


# --- direct moves animate exactly as before -----------------------------------------


def test_direct_move_into_a_portal_warps_as_before(tmp_path):
    session = _session(tmp_path, {"teleporters": [PORTAL]})
    for _ in range(2):
        plan, sfx = _dispatch(session, "MOVE_FORWARD")
        assert _portal_effects(plan) == [] and sfx == "step"
    plan, sfx = _dispatch(session, "MOVE_FORWARD")

    assert _portal_effects(plan) == [
        {"kind": "warp", "cell": [4, 1], "dest": [6, 3], "durationMs": WARP_MS}
    ]
    assert sfx == "warp"


def test_direct_move_into_a_death_tile_kills_as_before(tmp_path):
    session = _session(
        tmp_path,
        {"keys": [{"id": "kR", "position": [2, 1], "color": "red"}], "death_portals": [[4, 1]]},
    )
    for token in ("MOVE_FORWARD", "PICKUP", "MOVE_FORWARD"):
        _dispatch(session, token)
    plan, sfx = _dispatch(session, "MOVE_FORWARD")

    assert _portal_effects(plan) == [
        {"kind": "kill", "cell": [4, 1], "dest": [1, 1], "durationMs": KILL_MS}
    ]
    assert sfx == "kill"


def test_cardinal_move_into_a_portal_warps_as_before(tmp_path):
    session = _session(tmp_path, {"teleporters": [PORTAL]}, action_space="cardinal")
    _dispatch(session, "MOVE_EAST")
    _dispatch(session, "MOVE_NORTH")  # bumps the north wall: now facing NORTH
    plan, sfx = _dispatch(session, "MOVE_EAST")  # TURN_RIGHT, MOVE_FORWARD
    assert session.state.agent_position == (3, 1)
    assert _portal_effects(plan) == []

    plan, sfx = _dispatch(session, "MOVE_EAST")
    assert _portal_effects(plan) == [
        {"kind": "warp", "cell": [4, 1], "dest": [6, 3], "durationMs": WARP_MS}
    ]
    assert sfx == "warp"


def test_a_dispatch_after_the_episode_ended_does_not_replay_the_warp(tmp_path):
    session = _session(tmp_path, {"teleporters": [PORTAL]}, goal=(6, 3))
    for _ in range(2):
        _dispatch(session, "MOVE_FORWARD")
    plan, _ = _dispatch(session, "MOVE_FORWARD")
    assert session.episode_done and session.episode_success
    assert [item["kind"] for item in plan] == ["warp", "pulse"]

    plan, _ = _dispatch(session, "TURN_LEFT")  # ignored: the episode is over
    assert _portal_effects(plan) == []
