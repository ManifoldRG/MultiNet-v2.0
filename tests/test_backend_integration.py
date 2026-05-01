import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.backends.multigrid_backend import MultiGridBackend
from gridworld.task_spec import TaskSpecification


def test_minigrid_respects_initial_switch_state():
    spec = TaskSpecification.from_dict({
        "task_id": "switch_initial_on",
        "seed": 5,
        "difficulty_tier": 2,
        "maze": {
            "dimensions": [8, 8],
            "walls": [],
            "start": [1, 1],
            "goal": [6, 6],
        },
        "mechanisms": {
            "switches": [
                {
                    "id": "s1",
                    "position": [2, 2],
                    "controls": ["g1"],
                    "switch_type": "toggle",
                    "initial_state": "on",
                }
            ],
            "gates": [{"id": "g1", "position": [4, 4], "initial_state": "closed"}],
        },
        "goal": {"type": "reach_position", "target": [6, 6]},
        "max_steps": 40,
    })

    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    _, state, _ = backend.reset(seed=5)

    assert "s1" in state.active_switches
    assert "g1" in state.open_gates


def test_switch_colors_are_preserved_across_backends():
    spec = TaskSpecification.from_dict({
        "task_id": "colored_switch",
        "seed": 11,
        "difficulty_tier": 2,
        "maze": {
            "dimensions": [8, 8],
            "walls": [],
            "start": [1, 1],
            "goal": [6, 6],
        },
        "mechanisms": {
            "switches": [
                {
                    "id": "s1",
                    "position": [2, 2],
                    "controls": ["g1"],
                    "color": "white",
                    "switch_type": "toggle",
                    "initial_state": "off",
                }
            ],
            "gates": [{"id": "g1", "position": [4, 4], "initial_state": "closed"}],
        },
        "goal": {"type": "reach_position", "target": [6, 6]},
        "max_steps": 40,
    })

    assert spec.mechanisms.switches[0].color == "white"
    assert spec.to_dict()["mechanisms"]["switches"][0]["color"] == "white"

    minigrid = MiniGridBackend(render_mode="rgb_array")
    minigrid.configure(spec)
    minigrid.reset(seed=11)
    minigrid_switch = minigrid.env.grid.get(2, 2)
    assert minigrid_switch.visual_color == "white"

    multigrid = MultiGridBackend(tiling="square", render_mode="rgb_array")
    multigrid.configure(spec)
    multigrid.reset(seed=11)
    assert multigrid.env.state.objects["s1"].color == "white"


def test_doors_and_gates_may_replace_wall_cells():
    spec = TaskSpecification.from_dict({
        "task_id": "barriers_replace_walls",
        "seed": 4,
        "difficulty_tier": 2,
        "maze": {
            "dimensions": [8, 8],
            "walls": [[3, 1], [4, 2]],
            "start": [1, 1],
            "goal": [6, 6],
        },
        "mechanisms": {
            "keys": [{"id": "k1", "position": [2, 1], "color": "red"}],
            "doors": [{"id": "d1", "position": [3, 1], "requires_key": "red"}],
            "switches": [{"id": "s1", "position": [2, 2], "controls": ["g1"]}],
            "gates": [{"id": "g1", "position": [4, 2], "initial_state": "closed"}],
        },
        "goal": {"type": "reach_position", "target": [6, 6]},
        "max_steps": 40,
    })

    assert spec.validate()[0] is True

    minigrid = MiniGridBackend(render_mode="rgb_array")
    minigrid.configure(spec)
    minigrid.reset(seed=4)
    assert minigrid.env.grid.get(3, 1).type == "door"
    assert minigrid.env.grid.get(4, 2).type == "door"

    multigrid = MultiGridBackend(tiling="square", render_mode="rgb_array")
    multigrid.configure(spec)
    multigrid.reset(seed=4)
    barrier_cells = {
        obj.cell_id
        for obj in multigrid.env.state.objects.values()
        if obj.id in {"d1", "g1"}
    }
    wall_cells = {
        obj.cell_id
        for obj in multigrid.env.state.objects.values()
        if obj.obj_type == "wall"
    }
    assert barrier_cells.isdisjoint(wall_cells)


def test_multigrid_backend_preserves_mechanism_types():
    spec = TaskSpecification.from_dict({
        "task_id": "multigrid_fidelity",
        "seed": 9,
        "difficulty_tier": 4,
        "maze": {
            "dimensions": [8, 8],
            "walls": [[3, 3]],
            "start": [1, 1],
            "goal": [6, 6],
        },
        "mechanisms": {
            "keys": [{"id": "k1", "position": [2, 1], "color": "red"}],
            "doors": [{"id": "d1", "position": [3, 1], "requires_key": "red"}],
            "switches": [
                {
                    "id": "s1",
                    "position": [2, 2],
                    "controls": ["g1"],
                    "switch_type": "toggle",
                    "initial_state": "off",
                }
            ],
            "gates": [{"id": "g1", "position": [4, 2], "initial_state": "closed"}],
            "blocks": [{"id": "b1", "position": [2, 3], "color": "grey"}],
            "hazards": [{"id": "h1", "position": [2, 4], "hazard_type": "lava"}],
            "teleporters": [
                {"id": "tp1", "position_a": [5, 1], "position_b": [6, 4], "bidirectional": True}
            ],
        },
        "goal": {"type": "reach_position", "target": [6, 6]},
        "max_steps": 60,
    })

    backend = MultiGridBackend(tiling="square", render_mode="rgb_array")
    backend.configure(spec)
    _, state, _ = backend.reset(seed=9)

    objects = backend.env.state.objects
    assert objects["k1"].obj_type == "key"
    assert objects["d1"].obj_type == "door"
    assert objects["s1"].obj_type == "switch"
    assert objects["g1"].obj_type == "gate"
    assert objects["h1"].obj_type == "hazard"
    assert objects["tp1_a"].obj_type == "teleporter"
    assert objects["tp1_b"].obj_type == "teleporter"
    assert "wall_3_3" in objects
    assert state.block_positions["b1"]


def test_multigrid_hex_adapter_preserves_grid_coordinates():
    spec = TaskSpecification.from_dict({
        "task_id": "multigrid_hex_coordinates",
        "seed": 3,
        "difficulty_tier": 1,
        "maze": {
            "dimensions": [8, 8],
            "walls": [],
            "start": [1, 1],
            "goal": [6, 6],
        },
        "mechanisms": {
            "keys": [{"id": "k1", "position": [6, 5], "color": "red"}],
        },
        "goal": {"type": "reach_position", "target": [6, 6]},
        "max_steps": 40,
    })

    backend = MultiGridBackend(tiling="hex", render_mode="rgb_array")
    backend.configure(spec)
    _, state, _ = backend.reset(seed=3)

    tiling = backend.env.tiling
    goal_cell = tiling.cells[backend.env.state.goal.target_cell_id]
    key_cell = tiling.cells[backend.env.state.objects["k1"].cell_id]
    agent_cell = tiling.cells[backend.env.state.agent.cell_id]

    assert (agent_cell.col, agent_cell.row) == (1, 1)
    assert (key_cell.col, key_cell.row) == (6, 5)
    assert (goal_cell.col, goal_cell.row) == (6, 6)
    assert state.agent_position == (1, 1)
