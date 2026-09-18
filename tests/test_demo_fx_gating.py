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
