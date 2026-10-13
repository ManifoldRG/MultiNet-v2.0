"""Front ends use only backend-neutral calls; 2D-only effects are gated."""

from __future__ import annotations

import numpy as np
import pytest

from render3d_test_utils import REPO_ROOT, make_play_session


def test_demo_never_reaches_into_backend_env():
    files = sorted((REPO_ROOT / "demo").rglob("*.py")) + [REPO_ROOT / "play_task.py"]
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{lineno}"
        for path in files
        for lineno, line in enumerate(path.read_text().splitlines(), start=1)
        if "backend.env" in line
    ]
    assert not offenders, offenders


def _dispatch(session, token):
    prev_state, events_before = session.state, len(session.event_log)
    prev_rgb = np.asarray(session.backend.render(), dtype=np.uint8)
    session._dispatch_token(token)
    return prev_state, events_before, prev_rgb


@pytest.mark.parametrize("backend, grid_aligned", [("minigrid", True), ("mujoco3d", False)])
def test_cell_effects_only_on_grid_aligned_frames(tmp_path, backend, grid_aligned):
    pytest.importorskip("minigrid")
    if backend == "mujoco3d":
        pytest.importorskip("mujoco")
    from demo.fx import effects_for_dispatch, plan_effects

    session = make_play_session(tmp_path, backend)
    try:
        session._dispatch_token("MOVE_FORWARD")
        prev_state, before, prev_rgb = _dispatch(session, "PICKUP")
        kinds = {item["kind"] for item in plan_effects(session, "PICKUP", prev_state, before)}
        web = effects_for_dispatch(session, "PICKUP", prev_state, before, prev_rgb=prev_rgb)
        if grid_aligned:
            assert "flash" in kinds and any(e["kind"] == "flash" for e in web)
        else:
            assert kinds <= {"bounce"} and all(e["kind"] == "bounce" for e in web)
    finally:
        session.close()


def test_wall_bounce_survives_on_3d(tmp_path):
    pytest.importorskip("minigrid")
    pytest.importorskip("mujoco")
    from demo.fx import plan_effects

    session = make_play_session(tmp_path, "mujoco3d")
    try:
        session._dispatch_token("TURN_LEFT")  # face north: border wall ahead
        prev_state, before, _ = _dispatch(session, "MOVE_FORWARD")
        assert [i["kind"] for i in plan_effects(session, "MOVE_FORWARD", prev_state, before)] == ["bounce"]
    finally:
        session.close()


# --- Wall bounce on views that turn with the agent --------------------------
# first_person / tilt>0 put the agent's heading at screen-up, so a
# bounce in world axes jerks sideways (facing E/W) or lurches into the wall
# (facing S). The bounce must recoil away from the travel direction AS DRAWN.


def _blocked_session(heading: int, *, turns: bool):
    from types import SimpleNamespace

    return SimpleNamespace(
        transcript=[{"kind": "step", "event_type": "BLOCKED"}],
        task_spec=None,
        event_log=[],
        state=SimpleNamespace(agent_direction=heading, agent_position=(1, 1)),
        episode_done=False,
        episode_success=False,
        backend=SimpleNamespace(frame_is_grid_aligned=False, view_turns_with_agent=turns),
    )


def _bounce(session, token: str, prev_heading: int) -> tuple[int, int]:
    from types import SimpleNamespace

    from demo.fx import plan_effects

    prev = SimpleNamespace(agent_direction=prev_heading, agent_position=(1, 1))
    [item] = plan_effects(session, token, prev, 0)
    assert item["kind"] == "bounce"
    return item["dx"], item["dy"]


@pytest.mark.parametrize("heading", range(4))  # 0=E 1=S 2=W 3=N
def test_turning_view_bounce_recoils_straight_back_down_the_screen(heading):
    from demo.fx import BOUNCE_PX

    assert _bounce(_blocked_session(heading, turns=True), "MOVE_FORWARD", heading) == (0, BOUNCE_PX)


def test_turning_view_cardinal_bounce_uses_the_heading_after_the_turn():
    from demo.fx import BOUNCE_PX

    # MOVE_NORTH while facing east: turns to face north, then bumps the wall.
    assert _bounce(_blocked_session(3, turns=True), "MOVE_NORTH", 0) == (0, BOUNCE_PX)


@pytest.mark.parametrize(
    "heading, expected",
    [(0, (-2, 0)), (1, (0, -2)), (2, (2, 0)), (3, (0, 2))],
)
def test_north_up_view_bounce_stays_in_world_axes(heading, expected):
    assert _bounce(_blocked_session(heading, turns=False), "MOVE_FORWARD", heading) == expected


@pytest.mark.parametrize(
    "travel, heading, turns, expected",
    [
        ((1, 0), 0, False, (-2, 0)),
        ((1, 0), 0, True, (0, 2)),   # forward -> recoil down the screen
        ((1, 0), 3, True, (-2, 0)),  # east while facing north = screen-right
        ((0, 1), 0, True, (-2, 0)),  # south while facing east = screen-right -> recoil left
        ((0, 0), 1, True, (0, 0)),
    ],
)
def test_bounce_offset_pure(travel, heading, turns, expected):
    from demo.fx import bounce_offset

    assert bounce_offset(travel, heading, turns_with_agent=turns) == expected
