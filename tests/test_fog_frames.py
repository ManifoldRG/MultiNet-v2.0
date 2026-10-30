"""2D frames under partial observability (rules.observability).

fog_of_war draws each cell in view live, each explored cell out of view as it
looked when last seen (dimmed), and every other cell as the UNKNOWN tile;
view_cone keeps no memory. Full observability never reaches the mask, so its
frames are pinned byte-for-byte to the pre-mask renderer.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from collections import deque
from pathlib import Path

import minigrid
import numpy as np
import pytest
from minigrid.core.grid import Grid

from gridworld.actions import MiniGridActions as A
from gridworld.backends import get_backend
from gridworld.baselines import plan_bfs_path
from gridworld.custom_env import (
    REMEMBERED_DIM,
    UNKNOWN_BASE,
    UNKNOWN_HATCH,
    CustomMiniGridEnv,
    RotatingTile,
    unknown_tile,
)
from gridworld.task_spec import TaskSpecification
from gridworld.world_model import successors
from interface.progress_watchdog import ProgressStallWatchdog

_REPO = Path(__file__).resolve().parents[1]
_MANIFEST = json.loads((_REPO / "gridworld/fixtures/manifest.mechanics_15.json").read_text())
_SOURCES = {t["task_id"]: _REPO / t["source"] for t in _MANIFEST["tasks"]}
T = 32  # MiniGrid's tile size: the frames the pipeline sends

PORTALS = "mech15_M7_18x18_dense_kr_sg_kb_portals_0"
DEATH_FROZEN = "mech15_M8_18x18_dense_kr_sg_kb_death_frozen_0"
ROTATING = "mech15_M7_18x18_dense_kr_sg_kb_rotating_0"


def _load(task_id: str) -> TaskSpecification:
    return TaskSpecification.from_json(str(_SOURCES[task_id]))


def _with_obs(spec: TaskSpecification, mode: str, view_size: int = 7) -> TaskSpecification:
    rules = dataclasses.replace(spec.rules, observability=mode, view_size=view_size)
    return dataclasses.replace(spec, rules=rules)


def _backend(spec: TaskSpecification):
    backend = get_backend("minigrid")
    backend.configure(spec)
    frame, state, _ = backend.reset(seed=spec.seed)
    return backend, frame, state


def _tile(frame: np.ndarray, x: int, y: int) -> np.ndarray:
    return frame[y * T:(y + 1) * T, x * T:(x + 1) * T]


def _drawn(obj) -> np.ndarray:
    """A cell's tile as Grid.render draws it (its float tile cast to uint8)."""
    return Grid.render_tile(obj, tile_size=T).astype(np.uint8)


def _dimmed(obj) -> np.ndarray:
    return (_drawn(obj) * REMEMBERED_DIM).round().astype(np.uint8)


def _is_unknown(frame, x, y) -> bool:
    return np.array_equal(_tile(frame, x, y), unknown_tile(T))


def _check_frame(fog_frame, full_frame, env: CustomMiniGridEnv) -> None:
    """Every tile is live (in view), remembered (fog_of_war) or unknown."""
    visible = env.get_visible_cells() | {tuple(env.agent_pos)}
    for y in range(env.height):
        for x in range(env.width):
            got = _tile(fog_frame, x, y)
            if (x, y) in visible:
                # live state, the agent included, exactly as the full frame draws it
                assert np.array_equal(got, _tile(full_frame, x, y)), (x, y)
            elif env.observability == "fog_of_war" and (x, y) in env.remembered_cells:
                assert np.array_equal(got, _dimmed(env.remembered_cells[(x, y)])), (x, y)
            else:
                assert np.array_equal(got, unknown_tile(T)), (x, y)


# --------------------------------------------------------------------------- #
# Full observability: byte-for-byte the frames R2 ran on
# --------------------------------------------------------------------------- #
# sha256 over every full-observability frame of each mechanics-15 maze's BFS
# path followed by a 120-step seeded random walk (turns, moves, pickups,
# toggles; it dies, freezes, rides rotators and teleports), recorded on main
# 4b8b54d before the mask existed, with minigrid 3.1.0.
_FULL_FRAME_DIGESTS = {
    "mech15_M7_18x18_dense_kr_sg_kb_portals_0": "eae98355abfc2dcd6fa122ba5bc3845185f870b5acddd086102f8413a06210d8",
    "mech15_M7_18x18_dense_kr_sg_kb_death_0": "95ff7bb0eea5423e7688358b0bb613eec9d144f9b4476d38d4c8684cd7be8286",
    "mech15_M7_18x18_dense_kr_sg_kb_rotating_0": "cabbf6fddb6256175a80ced33b41a1883c3502fb942e52db78d39d42408f7ac2",
    "mech15_M7_18x18_dense_kr_sg_kb_frozen_0": "022bf850e2af5e23d5faa99af7b33f0500840b26256b93632e7ccb030c3bdd30",
    "mech15_M8_18x18_dense_kr_sg_kb_portals_rotating_0": "a8ce121e4f74664cc75efcd3563181fab7308c9347d06c71d53d48c713fb126e",
    "mech15_M8_18x18_dense_kr_sg_kb_portals_death_0": "fb38fcb16496cde2ffaaf15189c25f08486e8be695b092939aa55ea903e33e70",
    "mech15_M8_18x18_dense_kr_sg_kb_death_rotating_0": "3c1dc2facd28f998bd33dc01672b5bad4d2a6e043cc77ffdbae37d8dd565adc5",
    "mech15_M8_18x18_dense_kr_sg_kb_portals_frozen_0": "d5b425d915235a86db8d409c2d9300faf6488272f2653600f0627a1a8b60d076",
    "mech15_M8_18x18_dense_kr_sg_kb_frozen_rotating_0": "a293e86e947c161168c9cd9a5f53968c2bbb0083e811756b48236d30d2ef207d",
    "mech15_M8_18x18_dense_kr_sg_kb_death_frozen_0": "0e0db02a8c55af626720cad0657496c349f1b4537518db636780444e3e66c82d",
    "mech15_M9_18x18_dense_kr_sg_kb_portals_death_rotating_0": "72cdefc6edb56c9dbb1de153cfe77d6ba075f02586d8bc98748f248e2356f2a4",
    "mech15_M9_18x18_dense_kr_sg_kb_portals_rotating_frozen_0": "0992b03842669878984bca60cb8226c40288e606e7b132c3f5f95cdaa3833a57",
    "mech15_M9_18x18_dense_kr_sg_kb_portals_death_frozen_0": "19f04f5d1fa4b1272c9691b36de99768713435b307b5c02516ebb4af01c2b573",
    "mech15_M9_18x18_dense_kr_sg_kb_death_rotating_frozen_0": "2502e4b4804e8ab92d50b883b3e82be3b0d6cca3a1b8eefa1e50ed2624796c71",
    "mech15_M10_18x18_dense_kr_sg_kb_portals_death_rotating_frozen_0": "5a7f6c52f29e97aca8a1903644a26ce8213afc41cae262f506ad690ea907f51e",
}


def _full_rollout_digest(spec: TaskSpecification) -> str:
    h = hashlib.sha256()

    def feed(frame):
        h.update(repr(frame.shape).encode())
        h.update(frame.tobytes())

    backend, frame, _ = _backend(spec)
    feed(frame)
    for action in plan_bfs_path(spec).actions:
        frame, *_ = backend.step(int(action))
        feed(frame)
    rng = random.Random(0)
    backend.reset(seed=spec.seed)
    for _ in range(120):
        frame, *_ = backend.step(rng.choice([0, 1, 2, 2, 2, 3, 5]))
        feed(frame)
    return h.hexdigest()


@pytest.mark.parametrize("task_id", sorted(_FULL_FRAME_DIGESTS))
def test_full_observability_frames_are_byte_identical_to_the_pre_mask_renderer(task_id):
    if minigrid.__version__ != "3.1.0":
        pytest.skip(f"digests pinned with minigrid 3.1.0, found {minigrid.__version__}")
    spec = _load(task_id)
    assert spec.rules.observability == "full"
    assert _full_rollout_digest(spec) == _FULL_FRAME_DIGESTS[task_id]


def test_full_observability_never_reaches_the_mask(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("full observability reached the fog mask")

    monkeypatch.setattr(CustomMiniGridEnv, "_mask_unseen", boom)
    spec = _load("mech15_M10_18x18_dense_kr_sg_kb_portals_death_rotating_frozen_0")
    backend, *_ = _backend(spec)
    for action in plan_bfs_path(spec).actions:
        backend.step(int(action))
    assert backend.get_state().goal_reached


# --------------------------------------------------------------------------- #
# fog_of_war: nothing unseen leaks, memory is last-seen, the agent is drawn
# --------------------------------------------------------------------------- #
def test_unknown_tile_is_neither_floor_nor_wall():
    tile = unknown_tile(T)
    colours = {tuple(c) for c in tile.reshape(-1, 3)}
    assert colours == {tuple(UNKNOWN_BASE), tuple(UNKNOWN_HATCH)}
    assert (0, 0, 0) not in colours  # MiniGrid floor
    assert (100, 100, 100) not in colours  # MiniGrid wall and grid lines


def test_fog_reset_hides_every_object_the_agent_has_not_seen():
    spec = _load(PORTALS)
    backend, frame, state = _backend(_with_obs(spec, "fog_of_war"))
    env = backend.env
    _, full_frame, _ = _backend(spec)
    _check_frame(frame, full_frame, env)
    mech = spec.mechanisms
    hidden = [spec.goal.target.to_tuple()]
    hidden += [k.position.to_tuple() for k in mech.keys]
    hidden += [d.position.to_tuple() for d in mech.doors]
    hidden += [s.position.to_tuple() for s in mech.switches]
    hidden += [g.position.to_tuple() for g in mech.gates]
    for cell in hidden:
        assert cell not in env.explored_cells and _is_unknown(frame, *cell), cell
    # No pixel of the goal or of either key anywhere in the frame.
    pixels = {tuple(c) for c in frame.reshape(-1, 3)}
    assert not pixels & {(0, 255, 0), (0, 0, 255)}
    # The state the watchdog reads is the set the frame shows.
    assert state.explored_cells == env.explored_cells == set(env.remembered_cells)


def test_fog_frames_along_a_solve_are_live_remembered_or_unknown():
    """Lockstep with a full-observability twin: in-view tiles (and the agent)
    match it pixel for pixel, the explored set only grows, and it is exactly
    the set of cells the frame does not hide."""
    spec = _load(PORTALS)
    fog, frame, state = _backend(_with_obs(spec, "fog_of_war"))
    full, full_frame, _ = _backend(spec)
    sizes = [len(state.explored_cells)]
    for action in plan_bfs_path(spec).actions:
        frame, *_, state, _info = fog.step(int(action))
        full_frame, *_ = full.step(int(action))
        _check_frame(frame, full_frame, fog.env)
        assert state.explored_cells == fog.env.explored_cells == set(fog.env.remembered_cells)
        sizes.append(len(state.explored_cells))
    assert state.goal_reached
    assert sizes == sorted(sizes) and sizes[-1] > 2 * sizes[0]


def test_remembered_rotators_keep_the_direction_last_seen():
    spec = _load(ROTATING)
    fog, *_ = _backend(_with_obs(spec, "fog_of_war"))
    full, *_ = _backend(spec)
    stale = 0
    for action in plan_bfs_path(spec).actions:
        frame, *_ = fog.step(int(action))
        full_frame, *_ = full.step(int(action))
        env = fog.env
        out_of_view = set(env.remembered_cells) - env.get_visible_cells() - {tuple(env.agent_pos)}
        for cell in out_of_view:
            seen = env.remembered_cells[cell]
            live = env.grid.get(*cell)
            if isinstance(live, RotatingTile) and seen.direction != live.direction:
                stale += 1
                assert np.array_equal(_tile(frame, *cell), _dimmed(seen))
                assert not np.array_equal(_tile(frame, *cell), _dimmed(live))
                # the full frame shows where it points now
                assert np.array_equal(_tile(full_frame, *cell), _drawn(live))
    assert stale > 0  # rotators spin every step, so this path sees stale memories


GATE_ROOM = {
    "task_id": "fog_gate_room",
    "seed": 0,
    "difficulty_tier": 1,
    "maze": {"dimensions": [7, 7], "walls": [], "start": [1, 1], "goal": [5, 5]},
    "mechanisms": {
        "switches": [{"id": "s1", "position": [1, 3], "controls": ["g1"], "color": "yellow",
                      "switch_type": "toggle", "initial_state": "off"}],
        "gates": [{"id": "g1", "position": [5, 1], "color": "grey", "initial_state": "closed"}],
    },
    "rules": {"observability": "fog_of_war", "view_size": 7},
    "goal": {"type": "reach_position", "target": [5, 5]},
    "max_steps": 50,
}


def test_a_gate_opened_out_of_view_stays_drawn_closed_until_seen_again():
    backend, frame, _ = _backend(TaskSpecification.from_dict(GATE_ROOM))
    env = backend.env
    gate = (5, 1)
    assert gate in env.get_visible_cells() and not env.gates["g1"].is_open
    closed = _drawn(env.gates["g1"])
    assert np.array_equal(_tile(frame, *gate), closed)

    for action in (A.TURN_RIGHT, A.MOVE_FORWARD, A.MOVE_FORWARD):  # face south, onto the switch
        frame, *_ = backend.step(int(action))
    frame, *_ = backend.step(int(A.TOGGLE))
    assert env.gates["g1"].is_open and gate not in env.get_visible_cells()
    assert np.array_equal(_tile(frame, *gate), (closed * REMEMBERED_DIM).round().astype(np.uint8))

    frame, *_ = backend.step(int(A.TURN_LEFT))  # face east: the gate is in view again
    assert gate in env.get_visible_cells()
    assert np.array_equal(_tile(frame, *gate), _drawn(env.gates["g1"]))
    assert not np.array_equal(_tile(frame, *gate), closed)


# --------------------------------------------------------------------------- #
# view_cone and the view size
# --------------------------------------------------------------------------- #
def test_view_cone_draws_only_what_is_in_view_now():
    spec = _load(PORTALS)
    cone, frame, state = _backend(_with_obs(spec, "view_cone"))
    full, full_frame, _ = _backend(spec)
    for action in plan_bfs_path(spec).actions[:45]:
        frame, *_, state, _info = cone.step(int(action))
        full_frame, *_ = full.step(int(action))
        _check_frame(frame, full_frame, cone.env)
    env = cone.env
    in_view = env.get_visible_cells() | {tuple(env.agent_pos)}
    # The explored set (stall-watchdog progress) still grows; the frame forgets.
    assert len(state.explored_cells) > len(in_view)
    for cell in state.explored_cells - in_view:
        assert _is_unknown(frame, *cell)


def _open_room(view_size: int) -> TaskSpecification:
    return TaskSpecification.from_dict({
        "task_id": "fog_open_room", "seed": 0, "difficulty_tier": 1,
        "maze": {"dimensions": [15, 15], "walls": [], "start": [1, 7], "goal": [13, 13]},
        "mechanisms": {},
        "rules": {"observability": "fog_of_war", "view_size": view_size},
        "goal": {"type": "reach_position", "target": [13, 13]},
        "max_steps": 50,
    })


@pytest.mark.parametrize("view_size", [3, 5, 7, 9])
def test_the_view_reaches_view_size_minus_one_ahead_and_half_to_each_side(view_size):
    """MiniGrid's view: a view_size square ahead of the agent (its own row at
    the back), nothing behind. In an open room nothing occludes it."""
    backend, frame, _ = _backend(_open_room(view_size))
    half = view_size // 2
    expected = {
        (x, y)
        for x in range(1, 1 + view_size)
        for y in range(7 - half, 7 + half + 1)
    }
    assert backend.env.get_visible_cells() == expected
    for y in range(15):
        for x in range(15):
            assert _is_unknown(frame, x, y) == ((x, y) not in expected), (x, y)


# --------------------------------------------------------------------------- #
# Death resets, the stall watchdog and the planner
# --------------------------------------------------------------------------- #
def _actions_into_death(backend) -> list[int]:
    """Shortest action sequence (over the rulebook) whose last action enters a
    death cell."""
    env = backend.env
    ctx = env._planning_context()
    start = env._read_planner_state(ctx)
    queue, seen = deque([(start, [])]), {start}
    while queue:
        state, actions = queue.popleft()
        for t in successors(ctx, state):
            if t.label == "kill_reset":
                return actions + [t.action]
            if t.next_state not in seen:
                seen.add(t.next_state)
                queue.append((t.next_state, actions + [t.action]))
    raise AssertionError("no death cell reachable")


def test_a_death_reset_keeps_the_explored_memory_and_last_seen_state():
    """The world resets, the episode does not: the agent keeps what it saw,
    as its frame history does. The red door it opened before dying is locked
    again, but out of view it is still drawn as last seen: open."""
    spec = _load(DEATH_FROZEN)
    plan = plan_bfs_path(spec).actions
    backend, *_ = _backend(_with_obs(spec, "fog_of_war"))
    for action in plan[:88]:  # past the red door
        backend.step(int(action))
    assert backend.door_states()["DR"] is True
    *walk, fatal = _actions_into_death(backend)
    for action in walk:
        backend.step(int(action))
    explored_before = set(backend.env.explored_cells)

    frame, *_, state, info = backend.step(int(fatal))
    env = backend.env
    assert info["mechanic"]["kind"] == "died"
    assert tuple(state.agent_position) == spec.maze.start.to_tuple()
    assert backend.door_states()["DR"] is False  # the world reset
    assert state.explored_cells >= explored_before  # the memory did not
    assert state.explored_cells == env.explored_cells == set(env.remembered_cells)
    door = spec.mechanisms.doors[0]
    assert door.id == "DR"
    cell = door.position.to_tuple()
    assert cell not in env.get_visible_cells()
    seen = env.remembered_cells[cell]
    assert seen.is_open
    assert np.array_equal(_tile(frame, *cell), _dimmed(seen))
    assert not np.array_equal(_tile(frame, *cell), _dimmed(env.grid.get(*cell)))
    for x, y in explored_before:
        assert not _is_unknown(frame, x, y)


def test_turning_to_look_is_progress_under_fog_and_a_stall_under_full():
    """The stall watchdog counts newly seen cells as progress only under
    partial observability, where the frame really does show more."""
    spec = _load(PORTALS)
    stalls = {}
    for mode in ("full", "fog_of_war"):
        backend, _frame, state = _backend(_with_obs(spec, mode))
        watchdog = ProgressStallWatchdog(30, state)
        counts = []
        for _ in range(6):
            *_, state, _info = backend.step(int(A.TURN_LEFT))
            watchdog.observe(state)
            counts.append(watchdog.stall_count)
        stalls[mode] = counts
    assert stalls["full"] == [1, 2, 3, 4, 5, 6]
    # north, west and south each reveal cells; the fourth turn faces east again
    assert stalls["fog_of_war"] == [0, 0, 0, 1, 2, 3]


@pytest.mark.parametrize("task_id", [
    "mech15_M7_18x18_dense_kr_sg_kb_portals_0",
    "mech15_M7_18x18_dense_kr_sg_kb_death_0",
    "mech15_M8_18x18_dense_kr_sg_kb_death_rotating_0",
    "mech15_M8_18x18_dense_kr_sg_kb_frozen_rotating_0",
])
def test_fog_changes_the_frame_not_the_dynamics(task_id):
    """The planner-graded BFS optimum still applies: the full-observability
    plan solves the fogged maze in the same steps along the same states."""
    spec = _load(task_id)
    plan = plan_bfs_path(spec)
    fog, *_ = _backend(_with_obs(spec, "fog_of_war"))
    full, *_ = _backend(spec)
    for action in plan.actions:
        *_, s_fog, _ = fog.step(int(action))
        *_, s_full, _ = full.step(int(action))
        assert (s_fog.agent_position, s_fog.agent_direction, s_fog.agent_carrying) == (
            s_full.agent_position, s_full.agent_direction, s_full.agent_carrying)
        assert (s_fog.open_doors, s_fog.open_gates, s_fog.rotator_dirs) == (
            s_full.open_doors, s_full.open_gates, s_full.rotator_dirs)
    assert s_fog.goal_reached and s_fog.step_count == len(plan.actions) == s_full.step_count
