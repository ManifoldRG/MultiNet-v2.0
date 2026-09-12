from __future__ import annotations

import copy

import pytest

from gridworld.render3d.scene import (
    KEY_PARTS,
    UnsupportedSpecError,
    build_scene,
    check_supported,
    wall_cells,
)
from gridworld.task_spec import TaskSpecification
from render3d_test_utils import CORRIDOR, MECHANISMS, corpus_sample


def _spec(d):
    return TaskSpecification.from_dict(d)


@pytest.mark.parametrize(
    "patch, feature",
    [
        ({"mechanisms": {"blocks": [{"id": "b1", "position": [4, 1]}]}}, "blocks"),
        ({"mechanisms": {"hazards": [{"id": "h1", "position": [4, 1], "hazard_type": "lava"}]}}, "hazards"),
        (
            {"mechanisms": {"teleporters": [{"id": "t1", "position_a": [4, 1], "position_b": [5, 1]}]}},
            "teleporters",
        ),
        ({"goal": {"type": "collect_all", "target_ids": ["k1"]}}, "goal_type"),
        ({"rules": {"observability": "view_cone"}}, "observability"),
    ],
)
def test_check_supported_rejects_unrenderable_features(patch, feature):
    d = copy.deepcopy(CORRIDOR)
    d.update(patch)
    with pytest.raises(UnsupportedSpecError, match=feature):
        check_supported(_spec(d))


def test_check_supported_accepts_the_mechanism_room():
    check_supported(_spec(MECHANISMS))


def test_wall_cells_is_border_plus_spec_walls_minus_object_cells():
    spec = _spec(MECHANISMS)
    walls = wall_cells(spec)
    assert (0, 0) in walls and (8, 4) in walls  # border
    assert (4, 1) in walls and (4, 3) in walls  # spec walls
    assert (4, 2) not in walls  # the door cell
    assert (2, 2) not in walls  # floor


@pytest.mark.parametrize("path", corpus_sample(), ids=lambda p: p.stem)
def test_wall_cells_match_minigrid_grid(path):
    pytest.importorskip("minigrid")
    from gridworld.backends.minigrid_backend import MiniGridBackend

    spec = TaskSpecification.from_json(str(path))
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


def test_scene_compiles_and_index_names_exist():
    mujoco = pytest.importorskip("mujoco")
    spec = _spec(MECHANISMS)
    xml, index = build_scene(spec, wall_height=0.6, resolution=128)
    model = mujoco.MjModel.from_xml_string(xml)

    def geom_id(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    groups = (
        index.door_closed, index.door_open, index.switch_off,
        index.switch_on, index.gate_closed, index.gate_open, index.carried,
    )
    for mapping in groups:
        for names in mapping.values():
            assert names and all(geom_id(n) >= 0 for n in names), names
    for body in list(index.key_bodies.values()) + [index.agent_body]:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        assert body_id >= 0 and model.body_mocapid[body_id] >= 0  # mocap
    assert set(index.carried) == {"red"}
    assert index.carried["red"] == tuple(f"carried:red:{p}" for p in KEY_PARTS)
    width, height = spec.maze.dimensions
    n_walls = sum(1 for i in range(model.ngeom) if model.geom(i).name.startswith("wall:"))
    n_floor = sum(
        1 for i in range(model.ngeom)
        if model.geom(i).name.startswith("floor:") and model.geom(i).name != "floor:base"
    )
    assert n_walls == len(index.wall_cells)
    assert n_floor == width * height - len(index.wall_cells)


def test_door_slab_spans_the_corridor():
    mujoco = pytest.importorskip("mujoco")
    xml, _ = build_scene(_spec(CORRIDOR), wall_height=0.6, resolution=64)
    model = mujoco.MjModel.from_xml_string(xml)
    slab = model.geom("door:d1:slab")
    # Corridor runs east-west (walls north and south of the door), so the slab
    # is wide along y and thin along x.
    assert slab.size[1] > slab.size[0]
