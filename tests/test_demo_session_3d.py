"""The human-play session runs unchanged on the 3D backend."""

from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("minigrid")

from demo.theme import GRID_DISPLAY_SIZE  # noqa: E402
from render3d_test_utils import CORRIDOR, REPO_ROOT, make_play_session  # noqa: E402

TOKENS = ["MOVE_FORWARD", "PICKUP", "TOGGLE", "MOVE_FORWARD", "MOVE_FORWARD", "MOVE_FORWARD"]


def _play(session):
    for token in TOKENS:
        session._dispatch_token(token)
    return session


@pytest.mark.parametrize("backend, camera", [("minigrid", None), ("mujoco3d", "top_down")])
def test_session_plays_corridor_to_success(tmp_path, backend, camera):
    session = _play(make_play_session(tmp_path, backend, camera))
    try:
        assert session.episode_done and session.episode_success
        assert any(event.icon == "door" for event in session.event_log)
    finally:
        session.close()


def test_3d_session_matches_2d_progress_and_steps(tmp_path):
    s2 = _play(make_play_session(tmp_path, "minigrid"))
    s3 = _play(make_play_session(tmp_path, "mujoco3d", "chase"))
    try:
        assert s3.state.step_count == s2.state.step_count == len(TOKENS)
        assert s3.event_log == s2.event_log
        assert [r.get("action") for r in s3.transcript] == [r.get("action") for r in s2.transcript]
    finally:
        s2.close()
        s3.close()


def test_3d_frame_and_camera_cycle_are_display_only(tmp_path):
    session = make_play_session(tmp_path, "mujoco3d", "top_down")
    try:
        assert session.backend.frame_is_grid_aligned is False
        # rendered at the demo panel size, not stretched from 512
        assert session.backend.render().shape == (GRID_DISPLAY_SIZE, GRID_DISPLAY_SIZE, 3)
        assert session.camera_names == ("top_down", "chase", "fixed_angled", "first_person", "first_person_narrow")
        assert session.cycle_camera() == "chase"
        assert session.backend.camera == "chase"
        assert session.state.step_count == 0 and len(session.transcript) == 1
        for _ in range(4):
            session.cycle_camera()
        assert session.backend.camera == "top_down"
    finally:
        session.close()


def test_2d_session_has_no_cameras(tmp_path):
    session = make_play_session(tmp_path, "minigrid")
    try:
        assert session.camera_names == ()
        assert session.cycle_camera() is None
    finally:
        session.close()


@pytest.mark.parametrize("backend", ["minigrid", "mujoco3d"])
def test_physical_door_log_follows_backend_door_states(tmp_path, backend):
    session = make_play_session(tmp_path, backend)
    try:
        for token in ["MOVE_FORWARD", "PICKUP"]:
            session._dispatch_token(token)
        assert session._physical_door_states() == {"d1": False}
        session._dispatch_token("TOGGLE")  # unlock
        assert session._physical_door_states() == {"d1": True}
        session._dispatch_token("TOGGLE")  # PR #57 rulebook: no re-close
        assert session._physical_door_states() == {"d1": True}
    finally:
        session.close()


def test_load_task_survives_a_maze_the_backend_cannot_render(tmp_path, capsys):
    """I1: switching (`[`/`]`) onto a spec the 3D backend rejects must not
    crash the demo loop or desync the session from the backend."""
    session = make_play_session(tmp_path, "mujoco3d", "top_down")
    try:
        prev_path = session.task_path
        prev_task_id = session.task_spec.task_id
        prev_index = session.task_index

        bad = copy.deepcopy(CORRIDOR)
        bad["task_id"] = "render3d_corridor_blocked"
        bad["mechanisms"]["blocks"] = [{"id": "b1", "position": [4, 1]}]
        bad_path = tmp_path / "corridor_blocked.json"
        bad_path.write_text(json.dumps(bad))

        session._load_task(str(bad_path))

        assert session.task_path == prev_path
        assert session.task_spec.task_id == prev_task_id
        assert session.task_index == prev_index
        assert session.backend.render().shape == (GRID_DISPLAY_SIZE, GRID_DISPLAY_SIZE, 3)
        session._dispatch_token("MOVE_FORWARD")
        assert session.state.step_count == 1

        captured = capsys.readouterr()
        assert "backend cannot load" in captured.out
    finally:
        session.close()


def test_play_task_rejects_camera_without_3d_backend():
    pytest.importorskip("pygame")
    result = subprocess.run(
        [sys.executable, "play_task.py", "--camera", "chase"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 2
    assert "--camera requires --backend mujoco3d" in result.stderr


def test_tilt_steps_from_the_current_preset_clamps_and_is_display_only(tmp_path):
    session = make_play_session(tmp_path, "mujoco3d", "chase")
    try:
        last = session.backend.tilt_levels - 1
        assert session.tilt_status() is None
        assert session.step_tilt(+1) == 3  # chase sits at level 2
        assert session.tilt_status() == f"Tilt 3/{last} · walls 0.7"
        assert session.step_tilt(-1) == 2
        for _ in range(last + 2):
            session.step_tilt(-1)
        assert session.backend.tilt == 0
        for _ in range(last + 2):
            session.step_tilt(+1)
        assert session.backend.tilt == last
        assert session.state.step_count == 0 and len(session.transcript) == 1
        session.cycle_camera()  # V leaves tilt mode
        assert session.backend.tilt is None and session.tilt_status() is None
    finally:
        session.close()


def test_2d_session_cannot_tilt(tmp_path):
    session = make_play_session(tmp_path, "minigrid")
    try:
        assert session.step_tilt(+1) is None
        assert session.tilt_status() is None
    finally:
        session.close()
