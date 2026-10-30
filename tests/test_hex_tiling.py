"""Hex tiling: cell adjacency, the spec's maze.tiling, and moving and turning
under the R1 rulebook (gridworld.world_model) on six directions."""

import math

import pytest

from gridworld.actions import MiniGridActions
from gridworld.baselines import plan_bfs_path
from gridworld.task_spec import TaskSpecification
from gridworld.tiling import DIRECTION_NAMES, direction_between, direction_count, neighbor
from gridworld.world_model import TaskPlanningContext, apply
from multigrid.tilings.hex import DIRECTIONS as MULTIGRID_HEX_DIRECTIONS, HexTiling
from scorer.io import stable_hash

LEFT, RIGHT, FORWARD = MiniGridActions.TURN_LEFT, MiniGridActions.TURN_RIGHT, MiniGridActions.MOVE_FORWARD


def _spec(tiling="hex", start=(4, 4), walls=(), **overrides):
    d = {
        "task_id": "hex_test",
        "seed": 1,
        "difficulty_tier": 1,
        "maze": {
            "dimensions": [9, 9],
            "walls": [list(w) for w in walls],
            "start": list(start),
            "goal": [7, 7],
            "tiling": tiling,
        },
        "goal": {"type": "reach_position", "target": [7, 7]},
        "max_steps": 100,
    }
    d.update(overrides)
    return TaskSpecification.from_dict(d)


# --------------------------------------------------------------------------- #
# Adjacency
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("row", [3, 4], ids=["odd_row", "even_row"])
def test_hex_neighbours_match_the_multigrid_hex_graph(row):
    tiling = HexTiling()
    tiling.generate_graph(9, 9)
    by_xy = {(c.col, c.row): c for c in tiling.cells.values()}
    cell = by_xy[(4, row)]
    for direction, name in enumerate(MULTIGRID_HEX_DIRECTIONS):
        expected = tiling.cells[cell.neighbors[name]]
        assert neighbor("hex", (4, row), direction) == (expected.col, expected.row)
        assert name.upper() == DIRECTION_NAMES["hex"][direction]


def test_hex_direction_i_points_i_times_60_degrees_clockwise_from_east():
    tiling = HexTiling()
    tiling.generate_graph(9, 9)
    by_xy = {(c.col, c.row): c.position_hint for c in tiling.cells.values()}
    for start in ((4, 3), (4, 4)):
        x0, y0 = by_xy[start]
        for direction in range(6):
            x1, y1 = by_xy[neighbor("hex", start, direction)]
            angle = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 360  # y grows down
            assert angle == pytest.approx(60 * direction, abs=1e-6)


def test_square_directions_are_minigrids():
    assert [neighbor("square", (4, 4), d) for d in range(4)] == [(5, 4), (4, 5), (3, 4), (4, 3)]
    assert direction_count("square") == 4 and direction_count("hex") == 6
    assert direction_between("hex", (4, 3), (5, 4)) == 1  # odd row: southeast leans east
    assert direction_between("hex", (4, 4), (3, 5)) == 2  # even row: southwest leans west
    assert direction_between("hex", (4, 4), (6, 4)) is None


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #
def test_tiling_round_trips_and_square_stays_implicit():
    hex_spec = _spec()
    assert hex_spec.maze.tiling == "hex"
    assert hex_spec.to_dict()["maze"]["tiling"] == "hex"
    assert TaskSpecification.from_dict(hex_spec.to_dict()).maze.tiling == "hex"

    # Square stays implicit, so a spec that never named its tiling serializes
    # (and hashes: Stage 2 cache, run hash) exactly as before.
    legacy = _spec("square").to_dict()
    assert "tiling" not in legacy["maze"]
    assert stable_hash(TaskSpecification.from_dict(legacy).to_dict()) == stable_hash(legacy)
    assert stable_hash(legacy) != stable_hash(hex_spec.to_dict())


def test_validate_rejects_unknown_tiling_and_disagreeing_metadata():
    ok, errors = _spec("triangle").validate()
    assert not ok and any("Unknown maze tiling 'triangle'" in e for e in errors)
    ok, errors = _spec(metadata={"tiling": "square"}).validate()
    assert not ok and any("metadata.tiling 'square' disagrees" in e for e in errors)
    assert _spec(metadata={"tiling": "hex"}).validate()[0]


@pytest.mark.parametrize(
    "tiling,direction,valid",
    [("square", 3, True), ("square", 4, False), ("hex", 5, True), ("hex", 6, False), ("hex", -1, False)],
)
def test_rotating_directions_must_exist_on_the_tiling(tiling, direction, valid):
    spec = _spec(tiling, mechanisms={"rotating_tiles": [[5, 4]], "rotating_initial_directions": [direction]})
    ok, errors = spec.validate()
    assert ok is valid, errors
    if not valid:
        assert any(f"directions of a {tiling} tiling" in e for e in errors)


def test_minigrid_refuses_a_hex_maze():
    from gridworld.backends import MiniGridBackend
    from gridworld.task_parser import TaskParser

    with pytest.raises(ValueError, match="multigrid backend"):
        MiniGridBackend().configure(_spec())
    with pytest.raises(ValueError, match="multigrid backend"):
        TaskParser().parse(_spec())


def test_planner_refuses_an_unknown_tiling():
    with pytest.raises(ValueError, match="unknown tiling"):
        TaskPlanningContext(_spec("triangle"))


# --------------------------------------------------------------------------- #
# Moving and turning under the rulebook
# --------------------------------------------------------------------------- #
def test_turns_step_60_degrees_and_wrap_at_six():
    ctx = TaskPlanningContext(_spec())
    start = ctx.initial_state()
    assert start.agent_dir == 0  # east, as MiniGrid starts
    assert apply(ctx, start, LEFT).agent_dir == 5  # northeast
    state, seen = start, []
    for _ in range(6):
        state = apply(ctx, state, RIGHT)
        seen.append(state.agent_dir)
    assert seen == [1, 2, 3, 4, 5, 0]
    assert state == start


@pytest.mark.parametrize("start", [(4, 4), (4, 3)], ids=["even_row", "odd_row"])
def test_forward_moves_to_the_neighbour_in_each_facing(start):
    ctx = TaskPlanningContext(_spec(start=start))
    for direction in range(6):
        state = ctx.initial_state()
        for _ in range(direction):
            state = apply(ctx, state, RIGHT)
        moved = apply(ctx, state, FORWARD)
        assert moved.agent_pos == neighbor("hex", start, direction)
        assert moved.agent_dir == direction


def test_forward_into_a_wall_is_a_no_op():
    ctx = TaskPlanningContext(_spec(walls=[(5, 4)]))
    state = ctx.initial_state()
    assert apply(ctx, state, FORWARD) == state


def test_bfs_walks_hex_distance_with_one_turn():
    # (4, 4) -> (7, 7) is 5 hexes: two east and three southeast, so the
    # shortest plan turns right once.
    tiling = HexTiling()
    tiling.generate_graph(9, 9)
    ids = {(c.col, c.row): cid for cid, c in tiling.cells.items()}
    assert tiling.distance(ids[(4, 4)], ids[(7, 7)]) == 5
    path = plan_bfs_path(_spec())
    assert path.success
    assert path.positions[-1] == (7, 7)
    assert path.action_labels.count("move_forward") == 5
    assert path.action_labels.count("turn_right") == 1
    assert len(path.action_labels) == 6
