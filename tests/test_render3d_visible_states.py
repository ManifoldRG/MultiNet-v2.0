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


ALL_CAMERAS = ["top_down", "chase", "fixed_angled", "first_person"]
# One mechanism flips per case; the key case uses a used-up key (not carried),
# so nothing but the key's own cell may change.
LOCAL_CHANGES = {
    "door": ((_state(), {"d1": False}), (_state(), {"d1": True}), (4, 2)),
    "switch": ((_state(), CLOSED), (_state(active_switches={"s1"}), CLOSED), (2, 3)),
    "gate": ((_state(), CLOSED), (_state(open_gates={"g1"}), CLOSED), (6, 2)),
    "key": ((_state(), CLOSED), (_state(key_positions={}, collected_keys={"k1"}), CLOSED), (2, 1)),
}


@pytest.mark.parametrize("camera", ALL_CAMERAS)
@pytest.mark.parametrize("mechanism", sorted(LOCAL_CHANGES))
def test_state_change_repaints_only_its_own_cell(spec, camera, mechanism):
    # A hidden part that is still drawn somewhere (e.g. a used key parked
    # below the world origin) shows up as a change outside the cell.
    a, b, cell = LOCAL_CHANGES[mechanism]
    frame_a, pose, wall_height = _render(spec, camera, *a)
    frame_b, _, _ = _render(spec, camera, *b)
    rows, cols = _cell_region(pose, cell, wall_height)
    outside = np.any(frame_a != frame_b, axis=-1)
    outside[max(0, rows.start - 2) : rows.stop + 2, max(0, cols.start - 2) : cols.stop + 2] = False
    assert int(outside.sum()) == 0


def _drawn_geoms(renderer) -> set[str]:
    scene = renderer._renderer.scene
    return {
        renderer.model.geom(int(scene.geoms[i].objid)).name
        for i in range(scene.ngeom)
        if scene.geoms[i].objtype == mujoco.mjtObj.mjOBJ_GEOM
    }


def _body_geoms(model, body: str) -> set[str]:
    body_id = model.body(body).id
    return {model.geom(g).name for g in range(model.ngeom) if model.geom_bodyid[g] == body_id}


@pytest.mark.parametrize("camera", ALL_CAMERAS)
@pytest.mark.parametrize("flipped", [False, True])
def test_hidden_parts_are_not_drawn(spec, camera, flipped):
    # Hidden parts must leave the draw list, not just move out of the way:
    # a part moved below the floor still shows past the floor's edge.
    if flipped:
        state = _state(
            key_positions={}, collected_keys={"k1"}, active_switches={"s1"}, open_gates={"g1"}
        )
        doors = {"d1": True}
    else:
        state, doors = _state(), CLOSED
    renderer = SceneRenderer(spec, camera=camera, resolution=64)
    try:
        renderer.render(state, doors)
        index = renderer.index
        hidden: set[str] = set().union(*index.carried.values())
        pairs = [
            (index.door_closed["d1"], index.door_open["d1"]),
            (index.switch_off["s1"], index.switch_on["s1"]),
            (index.gate_closed["g1"], index.gate_open["g1"]),
        ]
        for shown_when_unflipped, shown_when_flipped in pairs:
            hidden |= set(shown_when_unflipped if flipped else shown_when_flipped)
        if flipped:
            hidden |= _body_geoms(renderer.model, index.key_bodies["k1"])
        drawn = _drawn_geoms(renderer)
    finally:
        renderer.close()
    assert hidden and not (hidden & drawn)


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


@pytest.mark.parametrize("camera", ["top_down", "chase", "fixed_angled"])
@pytest.mark.parametrize("direction", [0, 1, 2, 3])
def test_agent_silhouette_points_where_it_faces(spec, camera, direction):
    # The agent reads like MiniGrid's triangle: along its facing, the front
    # half of its cyan shape is much narrower than the back half, in every
    # overhead-ish view (a disc has equal halves).
    frame, pose, _ = _render(spec, camera, _state(agent_position=(3, 2), agent_direction=direction))
    centre = np.array(to_pixel(project(pose, cell_center(3, 2, 0.15)), RES))
    yaw = math.radians({0: 0.0, 1: -90.0, 2: 180.0, 3: 90.0}[direction])
    ahead = np.array(
        to_pixel(project(pose, (3.5 + 0.3 * math.cos(yaw), -2.5 + 0.3 * math.sin(yaw), 0.15)), RES)
    )
    axis = (ahead - centre) / np.linalg.norm(ahead - centre)
    img = frame.astype(int)
    cyan = (img[..., 1] > 120) & (img[..., 2] > 120) & (img[..., 0] < 90)
    points = np.argwhere(cyan).astype(float)
    along, across = points @ axis, points @ np.array([-axis[1], axis[0]])
    front = along > (along.max() + along.min()) / 2
    width = lambda sel: across[sel].max() - across[sel].min()  # noqa: E731
    assert width(front) < 0.75 * width(~front)


@pytest.mark.parametrize("camera", ALL_CAMERAS)
def test_compass_is_drawn_only_on_views_that_turn(spec, camera):
    from gridworld.render3d.hud import COMPASS_CAMERAS, DISC, compass_box

    frame, _, _ = _render(spec, camera, _state())
    top, left, bottom, right = compass_box(RES)
    disc_px = int((frame[top:bottom, left:right] == np.array(DISC, np.uint8)).all(axis=-1).sum())
    assert (disc_px > 0) == (camera in COMPASS_CAMERAS)


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
