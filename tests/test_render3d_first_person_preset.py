"""first_person = the wide beacon-friendly eye (Sean, 2026-09-23: fovy 110,
pitch -5, walls 1.0, beacons 10 tall, from the visibility sweep); the
2026-09-18 camera-ablation eye (first_person_narrow) was removed."""

from __future__ import annotations

import pytest

from gridworld.render3d.cameras import DEFAULT_WALL_HEIGHT, PRESETS, pose_for
from gridworld.render3d.scene import PORTAL_BEACON_TOP

WHERE = dict(agent_cell=(3, 3), direction=0, maze_dims=(9, 9))


def test_the_wide_eye_is_the_only_first_person_preset():
    assert PRESETS == ("top_down", "first_person")


def test_eye_preset_has_its_own_fov_pitch_and_walls():
    assert DEFAULT_WALL_HEIGHT["first_person"] == 1.0
    pose = pose_for("first_person", wall_height=1.0, **WHERE)
    assert pose.fovy == 110.0 and pose.elevation == -5.0 and not pose.orthographic


def test_wide_eye_still_sees_its_own_tile_edge_and_the_beacons():
    pose = pose_for("first_person", wall_height=1.0, **WHERE)
    assert pose.elevation - pose.fovy / 2 <= -54.0  # own cell's front edge (eye 0.55, 0.1 forward)
    assert PORTAL_BEACON_TOP == 10.0


def test_renderer_hides_the_agent_in_its_own_eye():
    pytest.importorskip("mujoco")
    import numpy as np

    from gridworld.backends.base import GridState
    from gridworld.render3d.renderer import SceneRenderer
    from gridworld.task_spec import TaskSpecification
    from render3d_test_utils import CORRIDOR

    spec = TaskSpecification.from_dict(CORRIDOR)
    state = GridState(agent_position=(1, 1), agent_direction=0)
    for camera in ("first_person",):
        r = SceneRenderer(spec, camera=camera, resolution=128)
        try:
            frame = r.render(state, {"d1": False}).astype(int)
            red = ((frame[..., 0] > 150) & (frame[..., 1] < 25) & (frame[..., 2] < 25)).sum()
            assert red == 0, camera  # no wedge in the eye's own frame
        finally:
            r.close()
