from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from gridworld.render3d.cameras import (
    DIRECTION_YAW,
    EYE_HEIGHT,
    basis,
    camera_position,
    cell_center,
    fits,
    maze_corners,
    pose_for,
    project,
)

DIMS = [(8, 8), (10, 10), (14, 12), (14, 14)]
DELTA = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


def test_direction_yaw_mapping():
    assert DIRECTION_YAW == {0: 0.0, 1: -90.0, 2: 180.0, 3: 90.0}


@pytest.mark.parametrize("azimuth", [0.0, 90.0, -90.0, 180.0, 37.0])
@pytest.mark.parametrize("elevation", [-90.0, -55.0, -12.0, 0.0])
def test_basis_is_orthonormal(azimuth, elevation):
    f, r, u = basis(azimuth, elevation)
    for vec in (f, r, u):
        assert np.linalg.norm(vec) == pytest.approx(1.0)
    assert f @ r == pytest.approx(0.0, abs=1e-9)
    assert f @ u == pytest.approx(0.0, abs=1e-9)
    assert r @ u == pytest.approx(0.0, abs=1e-9)


def test_top_down_is_orthographic_north_up_and_fits():
    pose = pose_for("top_down", agent_cell=(1, 1), direction=0, maze_dims=(10, 6), wall_height=0.4)
    assert pose.orthographic and pose.elevation == -90.0
    north_west = project(pose, cell_center(0, 0))
    south_east = project(pose, cell_center(9, 5))
    assert north_west[0] < 0 < south_east[0]  # west is left
    assert north_west[1] > 0 > south_east[1]  # north is up
    assert fits(pose, maze_corners(10, 6, 0.4))


@pytest.mark.parametrize("direction", [0, 1, 2, 3])
def test_chase_faces_heading_so_forward_is_up_in_frame(direction):
    pose = pose_for("chase", agent_cell=(5, 5), direction=direction, maze_dims=(10, 10), wall_height=0.6)
    assert pose.azimuth == DIRECTION_YAW[direction] and not pose.orthographic
    dx, dy = DELTA[direction]
    here = project(pose, cell_center(5, 5))
    ahead = project(pose, cell_center(5 + 2 * dx, 5 + 2 * dy))
    assert ahead[1] > here[1]
    assert abs(ahead[0] - here[0]) < 0.05


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("preset", ["chase", "fixed_angled"])
def test_auto_fit_keeps_whole_maze_in_frame_and_is_tight(preset, dims):
    width, height = dims
    corners = maze_corners(width, height, 0.6)
    for cell in [(1, 1), (width - 2, 1), (1, height - 2), (width - 2, height - 2)]:
        tight_somewhere = False
        for direction in range(4):
            pose = pose_for(preset, agent_cell=cell, direction=direction, maze_dims=dims, wall_height=0.6)
            assert fits(pose, corners)
            tight_somewhere |= not fits(dataclasses.replace(pose, distance=pose.distance * 0.9), corners)
        # chase keeps one distance for all headings, so it is tight at its worst one
        assert tight_somewhere


@pytest.mark.parametrize("dims", DIMS)
def test_chase_zoom_and_aim_never_change_within_a_maze(dims):
    # Playtest: the view breathed on every step. Only a turn may move it.
    width, height = dims
    poses = [
        pose_for("chase", agent_cell=cell, direction=direction, maze_dims=dims, wall_height=0.6)
        for cell in [(1, 1), (width - 2, 1), (1, height - 2), (width // 2, height // 2)]
        for direction in range(4)
    ]
    assert len({(p.lookat, p.distance, p.elevation, p.fovy) for p in poses}) == 1
    assert {p.azimuth for p in poses} == set(DIRECTION_YAW.values())


def test_first_person_sits_at_agent_eye_facing_heading():
    pose = pose_for("first_person", agent_cell=(3, 4), direction=3, maze_dims=(8, 8), wall_height=1.0)
    cam = camera_position(pose)
    ax, ay, _ = cell_center(3, 4)
    assert math.dist(cam[:2], (ax, ay)) < 0.2
    assert cam[2] == pytest.approx(EYE_HEIGHT, abs=1e-6)
    assert pose.azimuth == 90.0 and pose.fovy == 110.0 and not pose.orthographic


def test_unknown_preset_or_direction_raises():
    with pytest.raises(ValueError, match="unknown camera preset"):
        pose_for("isometric", agent_cell=(1, 1), direction=0, maze_dims=(8, 8), wall_height=0.6)
    with pytest.raises(ValueError, match="agent_direction"):
        pose_for("chase", agent_cell=(1, 1), direction=7, maze_dims=(8, 8), wall_height=0.6)


# --- demo tilt: top_down (level 0) down to first person (last level) --------


def test_tilt_ladder_is_anchored_on_the_presets():
    from gridworld.render3d.cameras import TILT_LEVELS, tilt_pose, tilt_wall_height

    kwargs = dict(agent_cell=(3, 4), direction=1, maze_dims=(10, 8))
    anchors = {0: "top_down", 2: "chase", len(TILT_LEVELS) - 1: "first_person"}
    for level, preset in anchors.items():
        wall = tilt_wall_height(level)
        assert tilt_pose(level, **kwargs) == pose_for(preset, wall_height=wall, **kwargs)
    assert tilt_wall_height(0) == 0.4 and tilt_wall_height(len(TILT_LEVELS) - 1) == 1.0


def test_tilt_levels_descend_and_walls_rise_step_by_step():
    from gridworld.render3d.cameras import TILT_LEVELS, tilt_pose, tilt_wall_height

    levels = range(len(TILT_LEVELS))
    poses = [tilt_pose(n, agent_cell=(3, 4), direction=0, maze_dims=(10, 8)) for n in levels]
    elevations = [p.elevation for p in poses]
    walls = [tilt_wall_height(n) for n in levels]
    assert elevations == sorted(elevations) and len(set(elevations)) == len(elevations)
    assert walls == sorted(walls) and len(set(walls)) == len(walls)


@pytest.mark.parametrize("dims", DIMS)
def test_tilt_keeps_the_agent_in_view_and_the_camera_out_of_walls(dims):
    from gridworld.render3d.cameras import TILT_LEVELS, tilt_pose, tilt_wall_height

    width, height = dims
    for level in range(1, len(TILT_LEVELS) - 1):
        wall = tilt_wall_height(level)
        for cell in [(1, 1), (width - 2, height - 2), (width // 2, 1)]:
            for direction in range(4):
                pose = tilt_pose(level, agent_cell=cell, direction=direction, maze_dims=dims)
                u, v = project(pose, cell_center(*cell, 0.15))
                assert abs(u) < 1 and abs(v) < 1, (level, cell, direction)
                cam = camera_position(pose)
                inside_agent_cell = (
                    abs(cam[0] - cell_center(*cell)[0]) < 0.5 and abs(cam[1] - cell_center(*cell)[1]) < 0.5
                )
                assert cam[2] > wall or inside_agent_cell, (level, cell, direction)
