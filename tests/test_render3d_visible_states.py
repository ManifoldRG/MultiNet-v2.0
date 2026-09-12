"""Each mechanism state must be visibly distinct (the renderer-bug 0a9cfec
lesson as a test), and the renderer's pixels must match cameras.project."""

from __future__ import annotations

import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from gridworld.backends.base import GridState  # noqa: E402
from gridworld.render3d.cameras import cell_center, project, to_pixel  # noqa: E402
from gridworld.render3d.renderer import SceneRenderer  # noqa: E402
from gridworld.task_spec import TaskSpecification  # noqa: E402
from render3d_test_utils import MECHANISMS  # noqa: E402

RES = 256
MIN_CHANGED_PX = 20
CLOSED = {"d1": False}


@pytest.fixture(scope="module")
def spec():
    return TaskSpecification.from_dict(MECHANISMS)


def _state(**overrides) -> GridState:
    fields = dict(agent_position=(1, 2), agent_direction=0, key_positions={"k1": (2, 1)})
    fields.update(overrides)
    return GridState(**fields)


def _render(spec, camera, state, doors=CLOSED):
    renderer = SceneRenderer(spec, camera=camera, resolution=RES)
    try:
        frame = renderer.render(state, doors)
        return frame, renderer.last_pose, renderer.index.wall_height
    finally:
        renderer.close()


def _cell_region(pose, cell, wall_height):
    x, y = cell
    points = [(x + dx, -(y + dy), z) for dx in (0, 1) for dy in (0, 1) for z in (0.0, wall_height)]
    rows, cols = zip(*(to_pixel(project(pose, p), RES) for p in points))
    return (
        slice(max(0, int(min(rows))), min(RES, int(max(rows)) + 1)),
        slice(max(0, int(min(cols))), min(RES, int(max(cols)) + 1)),
    )


def _changed_px(a, b, region) -> int:
    return int(np.any(a[region] != b[region], axis=-1).sum())


def _cyan_px(frame) -> int:
    img = frame.astype(int)
    return int(((img[..., 1] > 120) & (img[..., 2] > 120) & (img[..., 0] < 90)).sum())


@pytest.mark.parametrize("camera", ["top_down", "chase"])
@pytest.mark.parametrize("mechanism", ["door", "switch", "gate", "key"])
def test_each_mechanism_state_is_visibly_distinct(spec, camera, mechanism):
    if mechanism == "door":
        a, b, cell = (_state(), {"d1": False}), (_state(), {"d1": True}), (4, 2)
    elif mechanism == "switch":
        a, b, cell = (_state(), CLOSED), (_state(active_switches={"s1"}), CLOSED), (2, 3)
    elif mechanism == "gate":
        a, b, cell = (_state(), CLOSED), (_state(open_gates={"g1"}), CLOSED), (6, 2)
    else:  # key picked up: it leaves the floor
        picked = _state(key_positions={}, agent_carrying="red", collected_keys={"k1"})
        a, b, cell = (_state(), CLOSED), (picked, CLOSED), (2, 1)
    frame_a, pose, wall_height = _render(spec, camera, *a)
    frame_b, _, _ = _render(spec, camera, *b)
    assert _changed_px(frame_a, frame_b, _cell_region(pose, cell, wall_height)) >= MIN_CHANGED_PX


def test_carried_key_shows_on_the_agent(spec):
    empty, pose, wall_height = _render(spec, "top_down", _state())
    carrying, _, _ = _render(spec, "top_down", _state(key_positions={}, agent_carrying="red"))
    assert _changed_px(empty, carrying, _cell_region(pose, (1, 2), wall_height)) >= MIN_CHANGED_PX


def test_agent_heading_is_visible_top_down(spec):
    east, pose, wall_height = _render(spec, "top_down", _state(agent_direction=0))
    south, _, _ = _render(spec, "top_down", _state(agent_direction=1))
    assert _changed_px(east, south, _cell_region(pose, (1, 2), wall_height)) >= MIN_CHANGED_PX


@pytest.mark.parametrize("camera", ["top_down", "chase", "fixed_angled"])
def test_goal_pad_lands_where_project_predicts(spec, camera):
    # Door and gate open, so nothing tall stands between the camera and the goal.
    frame, pose, _ = _render(spec, camera, _state(open_gates={"g1"}), {"d1": True})
    img = frame.astype(int)
    green = (img[..., 1] > img[..., 0] + 40) & (img[..., 1] > img[..., 2] + 40)
    rows, cols = np.nonzero(green)
    assert rows.size > 0
    want = to_pixel(project(pose, cell_center(7, 2, 0.024)), RES)
    assert math.dist((rows.mean(), cols.mean()), want) < 3.0


def test_agent_hidden_only_in_first_person(spec):
    top, _, _ = _render(spec, "top_down", _state())
    first, _, _ = _render(spec, "first_person", _state())
    assert _cyan_px(top) > 0
    assert _cyan_px(first) == 0


def test_render_is_deterministic_and_well_formed(spec):
    renderer = SceneRenderer(spec, camera="chase", resolution=RES)
    try:
        a = renderer.render(_state(), CLOSED)
        b = renderer.render(_state(), CLOSED)
    finally:
        renderer.close()
    assert a.shape == (RES, RES, 3) and a.dtype == np.uint8
    assert a.tobytes() == b.tobytes()
    assert a is not b


def test_set_camera_rebuilds_for_new_wall_height(spec):
    renderer = SceneRenderer(spec, camera="top_down", resolution=64)
    try:
        assert renderer.index.wall_height == pytest.approx(0.4)
        renderer.set_camera("first_person")
        assert renderer.camera == "first_person"
        assert renderer.index.wall_height == pytest.approx(1.4)
        assert renderer.render(_state(), CLOSED).shape == (64, 64, 3)
    finally:
        renderer.close()


def test_unknown_camera_raises(spec):
    with pytest.raises(ValueError, match="unknown camera preset"):
        SceneRenderer(spec, camera="isometric", resolution=64)
