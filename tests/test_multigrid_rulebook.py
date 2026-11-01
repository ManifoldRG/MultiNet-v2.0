"""MultiGridBackend on the R1 rulebook (gridworld.world_model).

Square through MultiGrid must be MiniGridBackend step for step (state, reward,
info, frame), and a hex task must step exactly as the planner's ``apply``
says: the same parity PR #57 established between MiniGrid and BFS.
"""

from pathlib import Path

import numpy as np
import pytest

from gridworld.actions import MiniGridActions as A
from gridworld.backends import MiniGridBackend
from gridworld.backends.base import GridState
from gridworld.backends.multigrid_backend import MultiGridBackend
from gridworld.baselines import plan_bfs_path
from gridworld.task_spec import TaskSpecification
from gridworld.tiling import neighbor
from gridworld.world_model import TaskPlanningContext, apply, classify_mechanic, successors
from rulebook_test_utils import every_mechanic

_M10 = (
    Path(__file__).resolve().parents[1]
    / "ogbench/ogbench/procgen/maze_jsons/M10/18x18_dense_kr_sg_kb_portals_death_rotating_frozen_0.json"
)


def _route_labels(spec: TaskSpecification) -> list[str]:
    plan = plan_bfs_path(spec)
    assert plan.success
    return plan.action_labels


def _random_actions(n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    # Forward-heavy, so the walk travels instead of spinning in place.
    return [int(a) for a in rng.choice(7, size=n, p=[0.15, 0.15, 0.4, 0.1, 0.05, 0.1, 0.05])]


# --------------------------------------------------------------------------- #
# Square: MultiGrid == MiniGrid
# --------------------------------------------------------------------------- #
def _assert_backends_agree(spec: TaskSpecification, actions: list[int]) -> None:
    minigrid, multigrid = MiniGridBackend(), MultiGridBackend()
    minigrid.configure(spec)
    multigrid.configure(spec)
    assert multigrid.uses_rulebook
    frame_a, state_a, _ = minigrid.reset(seed=spec.seed)
    frame_b, state_b, _ = multigrid.reset(seed=spec.seed)
    assert state_a.to_dict() == state_b.to_dict()
    assert np.array_equal(frame_a, frame_b)
    for i, action in enumerate(actions):
        frame_a, reward_a, term_a, trunc_a, state_a, info_a = minigrid.step(action)
        frame_b, reward_b, term_b, trunc_b, state_b, info_b = multigrid.step(action)
        assert (reward_a, term_a, trunc_a, info_a) == (reward_b, term_b, trunc_b, info_b), i
        assert state_a.to_dict() == state_b.to_dict(), i
        assert np.array_equal(frame_a, frame_b), f"frames differ after step {i}"
        assert minigrid.rotator_directions() == multigrid.rotator_directions()
        assert minigrid.freeze_remaining() == multigrid.freeze_remaining()
        if term_a or trunc_a:
            break


@pytest.mark.parametrize("tiling", ["square", "hex"])
def test_the_route_crossesevery_mechanic(tiling):
    labels = _route_labels(every_mechanic(tiling))
    for label in ("pickup:kR", "open_door:DR", "freeze", "toggle:s1", "rotate"):
        assert label in labels, (tiling, label)


def test_square_bfs_route_matches_minigrid_step_for_step():
    spec = every_mechanic("square")
    _assert_backends_agree(spec, plan_bfs_path(spec).actions)


@pytest.mark.parametrize("seed", [0, 1])
def test_square_random_walk_matches_minigrid_step_for_step(seed):
    _assert_backends_agree(every_mechanic("square"), _random_actions(400, seed))


def test_square_death_reset_matches_minigrid():
    # Take the key, then walk back into the death pocket at (2, 2).
    actions = [A.MOVE_FORWARD, A.MOVE_FORWARD, A.PICKUP, A.TURN_LEFT, A.TURN_LEFT,
               A.MOVE_FORWARD, A.TURN_LEFT, A.MOVE_FORWARD, A.MOVE_FORWARD]
    _assert_backends_agree(every_mechanic("square"), [int(a) for a in actions])


@pytest.mark.skipif(not _M10.exists(), reason="ogbench submodule not checked out")
def test_square_m10_bfs_route_matches_minigrid():
    spec = TaskSpecification.from_json(str(_M10))
    _assert_backends_agree(spec, plan_bfs_path(spec).actions)


# --------------------------------------------------------------------------- #
# Hex: env step == planner apply
# --------------------------------------------------------------------------- #
def _hex_backend(spec: TaskSpecification) -> MultiGridBackend:
    backend = MultiGridBackend()
    backend.configure(spec)
    backend.reset(seed=spec.seed)
    return backend


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hex_random_walk_matches_planner_apply(seed):
    spec = every_mechanic("hex")
    backend = _hex_backend(spec)
    ctx = TaskPlanningContext(spec, drop_available=True)
    planned = ctx.initial_state()
    assert backend._read_planner_state() == planned
    mechanics = set()
    for i, action in enumerate(_random_actions(400, seed)):
        transition = next((t for t in successors(ctx, planned) if t.action == action), None)
        expected = apply(ctx, planned, action)
        _frame, reward, terminated, _trunc, state, info = backend.step(action)
        assert backend._read_planner_state() == expected, i
        if transition is not None:
            mechanic = classify_mechanic(ctx, planned, action, transition.label, expected)
            assert info.get("mechanic") == mechanic, i
            if mechanic:
                mechanics.add(mechanic["kind"])
        assert state.agent_position == expected.agent_pos
        assert state.agent_direction == expected.agent_dir
        assert state.rotator_dirs == expected.rotator_dirs
        assert state.freeze_remaining == expected.freeze_remaining
        assert terminated == (expected.agent_pos == ctx.goal)
        planned = expected
        if terminated:
            assert reward > 0 and state.goal_reached
            break
    assert mechanics, "the walk should meet at least one tile mechanic"


def test_hex_bfs_plan_solves_in_exactly_bfs_steps():
    spec = every_mechanic("hex")
    plan = plan_bfs_path(spec)
    assert plan.success
    backend = _hex_backend(spec)
    for i, action in enumerate(plan.actions):
        _frame, reward, terminated, truncated, state, _info = backend.step(action)
        assert state.agent_position == plan.positions[i + 1]
        assert terminated == (i == len(plan.actions) - 1)
    assert state.goal_reached and reward == pytest.approx(1 - 0.9 * len(plan.actions) / spec.max_steps)


# --------------------------------------------------------------------------- #
# Each mechanic on hex
# --------------------------------------------------------------------------- #
def _hex(walls=(), **mechanisms) -> TaskSpecification:
    return TaskSpecification.from_dict({
        "task_id": "hex_mechanic",
        "seed": 1,
        "difficulty_tier": 1,
        "maze": {"dimensions": [9, 9], "walls": [list(w) for w in walls], "start": [2, 3],
                 "goal": [7, 7], "tiling": "hex"},
        "mechanisms": mechanisms,
        "goal": {"type": "reach_position", "target": [7, 7]},
        "max_steps": 60,
    })


def test_hex_key_is_picked_up_in_place_and_opens_the_door_ahead():
    # (2, 3) is an odd row: east is (3, 3).
    backend = _hex_backend(_hex(
        keys=[{"id": "kB", "position": [2, 3], "color": "blue"}],
        doors=[{"id": "DB", "position": [3, 3], "requires_key": "blue", "initial_state": "locked"}],
    ))
    *_, state, info = backend.step(A.MOVE_FORWARD)
    assert state.agent_position == (2, 3) and info.get("invalid_action")
    *_, state, _ = backend.step(A.PICKUP)
    assert state.agent_carrying == "blue" and "kB" in state.collected_keys
    *_, state, _ = backend.step(A.TOGGLE)
    assert "DB" in state.open_doors and state.agent_carrying is None
    assert backend.door_states() == {"DB": True}
    *_, state, _ = backend.step(A.MOVE_FORWARD)
    assert state.agent_position == (3, 3)


def test_hex_switch_opens_the_gate():
    backend = _hex_backend(_hex(
        switches=[{"id": "s1", "position": [2, 3], "controls": ["g1"]}],
        gates=[{"id": "g1", "position": [3, 3]}],
    ))
    *_, state, info = backend.step(A.MOVE_FORWARD)
    assert state.agent_position == (2, 3) and info.get("invalid_action")
    *_, state, _ = backend.step(A.TOGGLE)
    assert state.active_switches == {"s1"} and state.open_gates == {"g1"}
    *_, state, _ = backend.step(A.MOVE_FORWARD)
    assert state.agent_position == (3, 3)


def test_hex_portal_lands_on_its_partner():
    backend = _hex_backend(_hex(teleporters=[{
        "id": "tp", "position_a": [3, 3], "position_b": [6, 6], "color": "purple",
    }]))
    *_, state, info = backend.step(A.MOVE_FORWARD)
    assert state.agent_position == (6, 6)
    assert info["mechanic"] == {"kind": "teleported", "from_xy": (3, 3), "to_xy": (6, 6), "color": "purple"}
    assert backend.env.state.objects["tp_a"].color == "purple"


def test_hex_ice_swallows_actions_while_turntables_keep_turning():
    backend = _hex_backend(_hex(
        frozen_tiles=[[3, 3]], freeze_steps=3,
        rotating_tiles=[[5, 5]], rotating_initial_directions=[0],
    ))
    *_, state, info = backend.step(A.MOVE_FORWARD)
    assert info["mechanic"] == {"kind": "froze", "cell_xy": (3, 3), "steps": 3}
    assert state.freeze_remaining == 3 and backend.freeze_remaining() == 3
    for remaining in (2, 1, 0):
        *_, state, info = backend.step(A.TURN_LEFT)
        assert state.agent_position == (3, 3) and state.agent_direction == 0
        assert info["mechanic"] == {"kind": "frozen", "remaining": remaining}
    assert state.rotator_dirs == (4,)  # one spin per action, four actions
    *_, state, _ = backend.step(A.MOVE_FORWARD)
    assert state.agent_position == (4, 3)


def test_hex_turntable_spins_through_six_directions_and_rides_its_arrow():
    # Agent at (2, 3) facing east steps onto the turntable at (3, 3).
    backend = _hex_backend(_hex(rotating_tiles=[[3, 3]], rotating_initial_directions=[5]))
    seen = [backend.rotator_directions()[0]]
    *_, state, info = backend.step(A.MOVE_FORWARD)
    assert info["mechanic"]["kind"] == "rotating" and state.agent_position == (3, 3)
    for _ in range(6):
        seen.append(backend.rotator_directions()[0])
        backend.step(A.TURN_LEFT)
    assert seen == [5, 0, 1, 2, 3, 4, 5]
    pointing = backend.rotator_directions()[0]
    *_, state, info = backend.step(A.MOVE_FORWARD)
    assert state.agent_position == neighbor("hex", (3, 3), pointing)
    assert info["mechanic"]["kind"] == "carried" and info["mechanic"]["direction"] == pointing


def test_hex_blocked_ride_stays_on_the_turntable():
    # Every neighbour of (3, 3) but the agent's own cell is a wall.
    walls = [neighbor("hex", (3, 3), d) for d in range(6) if neighbor("hex", (3, 3), d) != (2, 3)]
    backend = _hex_backend(_hex(walls=walls, rotating_tiles=[[3, 3]], rotating_initial_directions=[0]))
    backend.step(A.MOVE_FORWARD)  # the arrow turns to 1 (southeast, a wall)
    *_, state, info = backend.step(A.MOVE_FORWARD)
    assert state.agent_position == (3, 3)
    assert info["mechanic"] == {"kind": "rotating", "cell_xy": (3, 3), "pointing": 2, "stayed": True}


def test_hex_death_cell_resets_the_whole_maze():
    backend = _hex_backend(_hex(
        keys=[{"id": "kR", "position": [2, 3], "color": "red"}],
        doors=[{"id": "DR", "position": [5, 5], "requires_key": "red", "initial_state": "locked"}],
        death_portals=[[3, 3]],
    ))
    backend.step(A.PICKUP)
    *_, state, _ = backend.step(A.TURN_RIGHT)
    assert state.agent_carrying == "red" and state.agent_direction == 1
    backend.step(A.TURN_LEFT)
    *_, terminated, truncated, state, info = backend.step(A.MOVE_FORWARD)
    assert info["mechanic"]["kind"] == "died"
    assert (terminated, truncated) == (False, False)
    assert state.agent_position == (2, 3) and state.agent_direction == 0
    assert state.agent_carrying is None and state.key_positions == {"kR": (2, 3)}
    assert state.step_count == 4


# --------------------------------------------------------------------------- #
# Backend / spec agreement
# --------------------------------------------------------------------------- #
def test_backend_takes_the_spec_tiling_and_refuses_a_mismatch():
    hex_spec = every_mechanic("hex")
    assert MultiGridBackend()._configured is False
    backend = MultiGridBackend()
    backend.configure(hex_spec)
    assert backend.tiling_type == "hex" and backend.uses_rulebook
    with pytest.raises(ValueError, match="declares a hex tiling"):
        MultiGridBackend(tiling="square").configure(hex_spec)
    with pytest.raises(ValueError, match="declares a square tiling"):
        MultiGridBackend(tiling="hex").configure(every_mechanic("square"))


def test_legacy_tilings_refuse_rulebook_only_tasks():
    with pytest.raises(ValueError, match="native rules"):
        MultiGridBackend(tiling="triangle").configure(every_mechanic("hex"))
    with pytest.raises(ValueError, match="death_portals, frozen_tiles, rotating_tiles"):
        MultiGridBackend(tiling="triangle").configure(every_mechanic("square"))
    plain = every_mechanic("square", mechanisms={})
    backend = MultiGridBackend(tiling="triangle")
    backend.configure(plain)
    assert not backend.uses_rulebook


def test_grid_state_records_a_hex_tiling_and_keeps_square_snapshots_unchanged():
    backend = _hex_backend(every_mechanic("hex"))
    snapshot = backend.get_state().to_dict()
    assert snapshot["tiling"] == "hex"
    assert GridState.from_dict(snapshot).tiling == "hex"
    square = GridState(agent_position=(1, 1), agent_direction=0)
    assert "tiling" not in square.to_dict()
    assert GridState.from_dict(square.to_dict()).tiling == "square"
