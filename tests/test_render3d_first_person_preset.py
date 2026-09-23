"""first_person = the wide beacon-friendly eye (Sean, 2026-09-23: fovy 110,
pitch -5, walls 1.0, beacons 10 tall, from the visibility sweep); the
2026-09-18 camera-ablation eye survives as first_person_narrow."""

from __future__ import annotations

import pytest

from gridworld.render3d import hud
from gridworld.render3d.cameras import DEFAULT_WALL_HEIGHT, PRESETS, TILT_LEVELS, pose_for
from gridworld.render3d.scene import PORTAL_BEACON_TOP

WHERE = dict(agent_cell=(3, 3), direction=0, maze_dims=(9, 9))


def test_both_eyes_are_presets():
    assert PRESETS == ("top_down", "chase", "fixed_angled", "first_person", "first_person_narrow")


@pytest.mark.parametrize(
    "preset, fovy, elevation, walls",
    [("first_person", 110.0, -5.0, 1.0), ("first_person_narrow", 90.0, -12.0, 1.4)],
)
def test_eye_presets_have_their_own_fov_pitch_and_walls(preset, fovy, elevation, walls):
    assert DEFAULT_WALL_HEIGHT[preset] == walls
    pose = pose_for(preset, wall_height=DEFAULT_WALL_HEIGHT[preset], **WHERE)
    assert pose.fovy == fovy and pose.elevation == elevation and not pose.orthographic


def test_wide_eye_still_sees_its_own_tile_edge_and_the_beacons():
    pose = pose_for("first_person", wall_height=1.0, **WHERE)
    assert pose.elevation - pose.fovy / 2 <= -54.0  # own cell's front edge (eye 0.55, 0.1 forward)
    assert PORTAL_BEACON_TOP == 10.0


def test_narrow_eye_behaves_like_first_person_everywhere():
    assert hud.turns_with_agent("first_person_narrow")
    assert TILT_LEVELS[-1].preset == "first_person"


def test_renderer_hides_the_agent_for_both_eyes():
    pytest.importorskip("mujoco")
    import numpy as np

    from gridworld.backends.base import GridState
    from gridworld.render3d.renderer import SceneRenderer
    from gridworld.task_spec import TaskSpecification
    from render3d_test_utils import CORRIDOR

    spec = TaskSpecification.from_dict(CORRIDOR)
    state = GridState(agent_position=(1, 1), agent_direction=0)
    for camera in ("first_person", "first_person_narrow"):
        r = SceneRenderer(spec, camera=camera, resolution=128)
        try:
            frame = r.render(state, {"d1": False}).astype(int)
            cyan = ((frame[..., 1] > 120) & (frame[..., 2] > 120) & (frame[..., 0] < 90)).sum()
            assert cyan == 0, camera  # no wedge in the eye's own frame
        finally:
            r.close()
