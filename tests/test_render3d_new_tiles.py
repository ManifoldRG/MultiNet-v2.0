"""PR #57 tiles (portals, kill cells, frozen tiles, rotating tiles) in the 3D
scene: accepted, placed, visibly distinct, and the rotator phase reaches the
frame."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from gridworld.backends.base import GridState
from gridworld.render3d import palette
from gridworld.render3d.cameras import project, to_pixel
from gridworld.render3d.scene import UnsupportedSpecError, build_scene, check_supported, wall_cells
from gridworld.task_spec import TaskSpecification
from render3d_test_utils import TILES, new_mechanism_mazes

RES = 256
MIN_CHANGED_PX = 20
NO_DOORS: dict[str, bool] = {}


def _spec(d=TILES):
    return TaskSpecification.from_dict(d)


# --- spec support -----------------------------------------------------------


def test_check_supported_accepts_every_new_tile():
    check_supported(_spec())


def test_check_supported_rejects_unknown_portal_colour():
    d = copy.deepcopy(TILES)
    d["mechanisms"]["teleporters"][0]["color"] = "chartreuse"
    with pytest.raises(UnsupportedSpecError, match="chartreuse"):
        check_supported(_spec(d))


def test_palette_knows_cyan():
    r, g, b, _a = palette.rgba("cyan")
    assert g > 0.7 and b > 0.7 and r < 0.3


def test_wall_cells_exclude_tile_cells():
    # MiniGrid's put_obj overwrites whatever is at the cell, so a tile cell is
    # never a wall even when the spec lists a wall there.
    d = copy.deepcopy(TILES)
    d["maze"]["walls"] = [[3, 1], [5, 1], [5, 3], [4, 2]]
    walls = wall_cells(_spec(d))
    for cell in ((3, 1), (7, 3), (3, 3), (7, 1), (5, 1), (5, 3), (4, 2), (6, 2)):
        assert cell not in walls
    assert (0, 0) in walls


@pytest.mark.parametrize("path", new_mechanism_mazes(), ids=lambda p: p.stem)
def test_wall_cells_match_minigrid_on_new_mechanism_mazes(path):
    pytest.importorskip("minigrid")
    from gridworld.backends.minigrid_backend import MiniGridBackend

    spec = TaskSpecification.from_json(str(path))
    if spec.mechanisms.blocks:
        pytest.skip("blocks are not renderable (deferred mechanism)")
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    backend.reset(seed=spec.seed)
    grid = backend.env.grid
    width, height = spec.maze.dimensions
    minigrid_walls = {
        (x, y)
        for x in range(width)
        for y in range(height)
        if grid.get(x, y) is not None and grid.get(x, y).type == "wall"
    }
    assert wall_cells(spec) == minigrid_walls


# --- scene geometry ---------------------------------------------------------


def _model(spec, wall_height=0.6):
    mujoco = pytest.importorskip("mujoco")
    xml, index = build_scene(spec, wall_height=wall_height, resolution=64)
    return mujoco.MjModel.from_xml_string(xml), index


def _top(model, geom) -> float:
    """World z of a geom's top face (cylinder half-height is size[1])."""
    import mujoco

    half = geom.size[1] if int(geom.type[0]) == int(mujoco.mjtGeom.mjGEOM_CYLINDER) else geom.size[2]
    return float(geom.pos[2] + half)


def test_scene_indexes_one_arrow_per_direction_per_rotator():
    model, index = _model(_spec())
    assert set(index.rotator_arrows) == {0, 1}  # spec order of rotating_tiles
    for arrows in index.rotator_arrows.values():
        assert len(arrows) == 4
        for names in arrows:
            assert names and all(model.geom(n).id >= 0 for n in names)


def _geoms(model, prefix):
    return [model.geom(i) for i in range(model.ngeom) if model.geom(i).name.startswith(prefix)]


def test_rotator_arrow_rides_on_a_raised_turntable():
    model, index = _model(_spec())
    arrow_top = max(_top(model, model.geom(n)) for n in index.rotator_arrows[0][0])
    assert arrow_top >= 0.10
    table = model.geom("rotator:0:table")
    assert _top(model, table) >= 0.05 and float(table.size[0]) >= 0.40


def test_portal_is_a_floor_ring_with_a_translucent_beacon_above_the_walls():
    from gridworld.render3d.cameras import DEFAULT_WALL_HEIGHT

    model, _ = _model(_spec())
    ring = model.geom("portal:tp:a:ring")
    assert 0.08 <= _top(model, ring) <= 0.25 and float(ring.rgba[3]) == 1.0
    beacon = model.geom("portal:tp:a:beacon")
    assert _top(model, beacon) >= DEFAULT_WALL_HEIGHT["first_person"] + 1.0
    assert 0.2 <= float(beacon.rgba[3]) <= 0.7
    assert float(beacon.size[0]) <= float(model.geom("portal:tp:a:dot").size[0]) + 1e-6  # core stays visible from above


def test_skull_floats_at_eye_level_with_sockets_on_every_approach():
    import mujoco

    from gridworld.render3d.cameras import EYE_HEIGHT

    model, _ = _model(_spec())
    cranium = model.geom("kill:0:cranium")
    assert int(cranium.type[0]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
    assert abs(float(cranium.pos[2]) - EYE_HEIGHT) <= 0.15
    sockets = _geoms(model, "kill:0:socket")
    assert len(sockets) == 4
    cx, cy = cranium.pos[0], cranium.pos[1]
    # one socket in each diagonal quadrant, so two face any cardinal approach
    quadrants = {(float(g.pos[0]) > cx, float(g.pos[1]) > cy) for g in sockets}
    assert len(quadrants) == 4
    assert model.geom("kill:0:disc").id >= 0  # the dark-red floor mark stays


def test_frozen_tile_is_a_low_snow_bank():
    import mujoco

    from gridworld.render3d.cameras import DEFAULT_WALL_HEIGHT

    model, _ = _model(_spec())
    bank = model.geom("frozen:0:bank")
    assert int(bank.type[0]) == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
    top = float(bank.pos[2] + bank.size[2])
    assert 0.10 <= top <= DEFAULT_WALL_HEIGHT["first_person"] * 0.2  # a drift, not a wall
    assert float(bank.size[0]) >= 0.40 and float(bank.size[1]) >= 0.40
    assert len(_geoms(model, "frozen:0:")) >= 2


def test_portal_pair_shares_its_colour_and_each_end_is_drawn():
    model, _ = _model(_spec())
    names = {model.geom(i).name for i in range(model.ngeom)}
    a = {n for n in names if n.startswith("portal:tc:a:")}
    b = {n for n in names if n.startswith("portal:tc:b:")}
    assert a and b and len(a) == len(b)
    cyan = palette.rgba("cyan")
    assert any(np.allclose(model.geom(n).rgba, cyan, atol=1e-3) for n in a)


# --- rotator phase reaches the renderer -----------------------------------------


def test_state_backend_default_has_no_rotators():
    from gridworld.backends.base import AbstractGridBackend

    class Stub(AbstractGridBackend):
        def configure(self, task_spec): ...
        def reset(self, seed=None): ...
        def step(self, action): ...
        def render(self): ...
        def get_mission_text(self): return ""
        def get_state(self): return GridState(agent_position=(0, 0), agent_direction=0)

    assert Stub().rotator_directions() == ()


def test_minigrid_backend_reports_live_rotator_directions():
    pytest.importorskip("minigrid")
    from gridworld.actions import MiniGridActions as A
    from gridworld.backends.minigrid_backend import MiniGridBackend

    backend = MiniGridBackend(render_mode="rgb_array")
    assert backend.rotator_directions() == ()
    backend.configure(_spec())
    backend.reset(seed=0)
    assert backend.rotator_directions() == (0, 3)
    backend.step(int(A.TURN_LEFT))  # every legal step spins every rotator
    assert backend.rotator_directions() == (1, 0)


def test_mujoco3d_frame_follows_the_rotator_phase():
    pytest.importorskip("mujoco")
    pytest.importorskip("minigrid")
    from gridworld.actions import MiniGridActions as A
    from gridworld.backends import get_backend

    backend = get_backend("mujoco3d", camera="top_down", resolution=RES)
    try:
        backend.configure(_spec())
        start, _state, _info = backend.reset(seed=0)
        turned, *_rest = backend.step(int(A.TURN_LEFT))
        assert backend.rotator_directions() == (1, 0)
        pose = backend._renderer.last_pose
        region = _cell_region(pose, (6, 2), backend._renderer.index.wall_height)
        assert _changed_px(start, turned, region) >= MIN_CHANGED_PX
    finally:
        backend.close()


# --- visible states -------------------------------------------------------------


def _state(**overrides) -> GridState:
    fields = dict(agent_position=(1, 2), agent_direction=0)
    fields.update(overrides)
    return GridState(**fields)


def _render(spec, camera, state, rotators=(0, 3)):
    from gridworld.render3d.renderer import SceneRenderer

    renderer = SceneRenderer(spec, camera=camera, resolution=RES)
    try:
        frame = renderer.render(state, NO_DOORS, rotators=rotators)
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


@pytest.mark.parametrize("camera", ["top_down", "chase"])
@pytest.mark.parametrize(
    "tile, cell",
    [("portal", (7, 3)), ("kill", (5, 1)), ("frozen", (5, 3)), ("rotator", (6, 2))],
)
def test_each_tile_differs_from_plain_floor(tile, cell, camera):
    pytest.importorskip("mujoco")
    spec = _spec()
    bare = copy.deepcopy(TILES)
    bare["mechanisms"] = {}
    bare["maze"]["walls"] = []
    with_tiles, pose, wall_height = _render(spec, camera, _state())
    floor_only, _, _ = _render(_spec(bare), camera, _state(), rotators=())
    assert _changed_px(with_tiles, floor_only, _cell_region(pose, cell, wall_height)) >= MIN_CHANGED_PX


@pytest.mark.parametrize("camera", ["top_down", "chase"])
def test_rotator_direction_change_is_visible(camera):
    pytest.importorskip("mujoco")
    spec = _spec()
    east, pose, wall_height = _render(spec, camera, _state(), rotators=(0, 3))
    south, _, _ = _render(spec, camera, _state(), rotators=(1, 3))
    region = _cell_region(pose, (4, 2), wall_height)
    assert _changed_px(east, south, region) >= MIN_CHANGED_PX
    # only that rotator's cell changed
    other = np.any(east != south, axis=-1)
    other[region] = False
    assert other.sum() == 0


def test_arrow_stays_readable_under_the_agent_top_down():
    """2D shrinks the agent so the tile's arrow shows; in 3D the full-size
    arrow must still change the cell when it turns under the wedge."""
    pytest.importorskip("mujoco")
    spec = _spec()
    agent_on = _state(agent_position=(4, 2))
    east, pose, wall_height = _render(spec, "top_down", agent_on, rotators=(0, 3))
    south, _, _ = _render(spec, "top_down", agent_on, rotators=(1, 3))
    assert _changed_px(east, south, _cell_region(pose, (4, 2), wall_height)) >= MIN_CHANGED_PX


def test_first_person_sees_the_tile_ahead():
    pytest.importorskip("mujoco")
    spec = _spec()
    # agent on (4,1) facing EAST looks straight at the kill cell (5,1)
    facing_kill, _, _ = _render(spec, "first_person", _state(agent_position=(4, 1), agent_direction=0))
    bare = copy.deepcopy(TILES)
    bare["mechanisms"] = {}
    bare["maze"]["walls"] = []
    facing_floor, _, _ = _render(_spec(bare), "first_person", _state(agent_position=(4, 1), agent_direction=0), rotators=())
    assert int(np.any(facing_kill != facing_floor, axis=-1).sum()) >= 200


@pytest.mark.parametrize("camera", ["top_down", "first_person"])
def test_arrow_lifts_above_the_agent_standing_on_it(camera):
    """Facing the same way as the arrow, the wedge would hide it; the arrow
    rises above the wedge so top-down draws it on top and the eye sees its tip."""
    pytest.importorskip("mujoco")
    spec = _spec()
    on_tile = _state(agent_position=(4, 2), agent_direction=0)
    east, pose, wall_height = _render(spec, camera, on_tile, rotators=(0, 3))
    west, _, _ = _render(spec, camera, on_tile, rotators=(2, 3))
    region = _cell_region(pose, (4, 2), wall_height) if camera == "top_down" else (slice(0, RES), slice(0, RES))
    assert _changed_px(east, west, region) >= MIN_CHANGED_PX


def test_arrow_sits_on_the_tile_once_the_agent_leaves():
    mujoco = pytest.importorskip("mujoco")
    from gridworld.render3d.renderer import SceneRenderer
    from gridworld.render3d.scene import ROTATOR_ARROW_LIFT

    renderer = SceneRenderer(_spec(), camera="top_down", resolution=64)
    try:
        arrow = mujoco.mj_name2id(renderer.model, mujoco.mjtObj.mjOBJ_GEOM, "rotator:0:arrow0")
        home = float(renderer.model.geom_pos[arrow][2])  # the compiler's mesh centre offset
        renderer.render(_state(agent_position=(4, 2)), NO_DOORS, rotators=(0, 3))
        assert renderer.model.geom_pos[arrow][2] == pytest.approx(home + ROTATOR_ARROW_LIFT)
        renderer.render(_state(agent_position=(3, 2)), NO_DOORS, rotators=(0, 3))
        assert renderer.model.geom_pos[arrow][2] == pytest.approx(home)
    finally:
        renderer.close()
