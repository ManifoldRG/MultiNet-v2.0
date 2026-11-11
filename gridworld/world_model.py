"""R1 gridworld rulebook: pure (state, action) -> state.

BFS searches this. MiniGrid ``step`` applies this then writes the result
back onto the live env. The validator uses the same graph. Nothing here
holds a MiniGrid instance.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from math import isfinite, lcm
from typing import Callable, Iterable

from .actions import MiniGridActions
from .task_spec import TaskSpecification


DIRECTION_VECTORS: tuple[tuple[int, int], ...] = (
    (1, 0),   # right
    (0, 1),   # down
    (-1, 0),  # left
    (0, -1),  # up
)


@dataclass(frozen=True)
class PlannerState:
    """Hashable world state used as a BFS node and as ``step``'s snapshot."""

    agent_pos: tuple[int, int]
    agent_dir: int
    carrying_key: str | None
    collected_keys: frozenset[str]
    active_switches: frozenset[str]
    used_switches: frozenset[str]
    open_gates: frozenset[str]
    open_doors: frozenset[str]
    key_positions: frozenset[tuple[str, int, int]] = frozenset()
    freeze_remaining: int = 0
    rotator_dirs: tuple[int, ...] = ()
    tick: int = 0
    sealed: frozenset[tuple[int, int]] = frozenset()
    vanish_at: frozenset[tuple[int, int, int]] = frozenset()
    door_timers: frozenset[tuple[str, int]] = frozenset()
    switch_timers: frozenset[tuple[str, int]] = frozenset()
    shut_doors: frozenset[str] = frozenset()
    shut_switches: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Transition:
    """One legal action in the R1 graph."""

    action: int
    label: str
    next_state: PlannerState


class TaskPlanningContext:
    """Fast lookup tables derived from a ``TaskSpecification``."""

    def __init__(self, spec: TaskSpecification, *, drop_available: bool = False):
        self.spec = spec
        self.drop_available = drop_available
        self.width, self.height = spec.maze.dimensions
        self.goal = spec.resolved_goal()
        self.start = spec.maze.start.to_tuple()
        self.key_consumption = spec.rules.key_consumption
        self.disappear_after = spec.rules.disappear_after
        self.shut_after = spec.rules.shut_after

        self.walls = {(wall.x, wall.y) for wall in spec.maze.walls}
        for x in range(self.width):
            self.walls.add((x, 0))
            self.walls.add((x, self.height - 1))
        for y in range(self.height):
            self.walls.add((0, y))
            self.walls.add((self.width - 1, y))

        self.keys_by_pos = {
            key.position.to_tuple(): {"id": key.id, "color": key.color}
            for key in spec.mechanisms.keys
        }
        self.keys_by_id = {
            key.id: {"position": key.position.to_tuple(), "color": key.color}
            for key in spec.mechanisms.keys
        }
        self.doors_by_pos = {
            door.position.to_tuple(): {
                "id": door.id,
                "color": door.requires_key,
                "locked": door.initial_state == "locked",
            }
            for door in spec.mechanisms.doors
        }
        self.switches_by_pos = {
            switch.position.to_tuple(): {
                "id": switch.id,
                "controls": tuple(switch.controls),
                "switch_type": switch.switch_type,
                "initial_state": switch.initial_state,
            }
            for switch in spec.mechanisms.switches
        }
        self.gates_by_pos = {
            gate.position.to_tuple(): {
                "id": gate.id,
                "open": gate.initial_state == "open",
            }
            for gate in spec.mechanisms.gates
        }
        self.blocks = {block.position.to_tuple() for block in spec.mechanisms.blocks}
        self.hazards = {hazard.position.to_tuple() for hazard in spec.mechanisms.hazards}
        self.kill_cells = {cell.to_tuple() for cell in spec.mechanisms.kill_cells}
        self.frozen_tiles = {cell.to_tuple() for cell in spec.mechanisms.frozen_tiles}
        self.freeze_steps = spec.mechanisms.freeze_steps
        self.rotating_index = {
            cell.to_tuple(): i for i, cell in enumerate(spec.mechanisms.rotating_tiles)
        }
        self.portal_at: dict[tuple[int, int], tuple[tuple[tuple[int, int], ...], int, bool]] = {}
        for teleporter in spec.mechanisms.teleporters:
            ends = tuple(pos.to_tuple() for pos in teleporter.ends())
            for index, pos in enumerate(ends):
                self.portal_at[pos] = (ends, index, teleporter.bidirectional)
        self.phased = [
            (
                frozenset(cell.to_tuple() for cell in patch.cells),
                patch.open_steps,
                patch.closed_steps,
                patch.offset,
            )
            for patch in spec.mechanisms.phased_patches
        ]
        self.phased_cells = frozenset().union(*(cells for cells, *_ in self.phased)) if self.phased else frozenset()
        self.phase_period = 1
        for _, open_steps, closed_steps, _ in self.phased:
            self.phase_period = lcm(self.phase_period, open_steps + closed_steps)

        self.initial_open_doors = frozenset(
            door["id"] for door in self.doors_by_pos.values() if not door["locked"]
        )
        self.initial_active_switches = frozenset(
            switch["id"]
            for switch in self.switches_by_pos.values()
            if switch["initial_state"] == "on"
        )
        self.initial_used_switches = frozenset(
            switch["id"]
            for switch in self.switches_by_pos.values()
            if switch["initial_state"] == "on" and switch["switch_type"] == "one_shot"
        )

    def initial_state(self) -> PlannerState:
        """Return the world state at the beginning of the episode."""
        return PlannerState(
            agent_pos=self.start,
            agent_dir=0,
            carrying_key=None,
            collected_keys=frozenset(),
            active_switches=self.initial_active_switches,
            used_switches=self.initial_used_switches,
            open_gates=self.recompute_open_gates(self.initial_active_switches),
            open_doors=self.initial_open_doors,
            key_positions=frozenset(
                (key.id, key.position.x, key.position.y)
                for key in self.spec.mechanisms.keys
            ),
            rotator_dirs=tuple(self.spec.mechanisms.rotating_initial_directions),
        )

    def recompute_open_gates(self, active_switches: frozenset[str]) -> frozenset[str]:
        """An active switch flips each gate it controls from its initial state."""
        flipped = {
            gate_id
            for switch in self.switches_by_pos.values()
            if switch["id"] in active_switches
            for gate_id in switch["controls"]
        }
        return frozenset(
            gate["id"]
            for gate in self.gates_by_pos.values()
            if gate["open"] != (gate["id"] in flipped)
        )


def apply(ctx: TaskPlanningContext, state: PlannerState, action: int) -> PlannerState:
    """Return the next world state for ``action``. Illegal actions still spin rotators."""
    action = int(action)
    for transition in successors(ctx, state):
        if transition.action == action:
            return transition.next_state
    settled = _settle(state)
    clocks = isfinite(ctx.disappear_after) or isfinite(ctx.shut_after) or bool(ctx.phased)
    if clocks:
        return _tick(ctx, state, settled)
    return settled


def successors(ctx: TaskPlanningContext, state: PlannerState) -> Iterable[Transition]:
    """Generate legal R1 actions from a world state."""
    clocks = isfinite(ctx.disappear_after) or isfinite(ctx.shut_after) or bool(ctx.phased)
    fresh = ctx.initial_state()
    for transition in _raw_successors(ctx, state):
        if not clocks or transition.next_state == fresh:
            yield transition
        else:
            yield Transition(
                transition.action,
                transition.label,
                _tick(ctx, state, transition.next_state),
            )


def _raw_successors(ctx: TaskPlanningContext, state: PlannerState) -> Iterable[Transition]:
    if state.freeze_remaining > 0:
        thawed = replace(
            state,
            freeze_remaining=state.freeze_remaining - 1,
            rotator_dirs=_spin(state.rotator_dirs),
        )
        for action in MiniGridActions:
            yield Transition(int(action), "freeze", thawed)
        return

    if state.agent_pos in ctx.rotating_index:
        used = {int(MiniGridActions.MOVE_FORWARD)}
        for t in _idle_successors(ctx, state):
            if t.action == int(MiniGridActions.MOVE_FORWARD):
                continue
            used.add(t.action)
            ns = t.next_state if t.label == "kill_reset" else _settle(t.next_state)
            yield Transition(t.action, t.label, ns)
        waited = _settle(state)
        for action in MiniGridActions:
            if int(action) not in used:
                yield Transition(int(action), "rotate_wait", waited)
        yield Transition(int(MiniGridActions.MOVE_FORWARD), "rotate", _ride(ctx, state))
        return

    for t in _idle_successors(ctx, state):
        ns = t.next_state if t.label == "kill_reset" else _settle(t.next_state)
        yield Transition(t.action, t.label, ns)


def _idle_successors(ctx: TaskPlanningContext, state: PlannerState) -> Iterable[Transition]:
    yield Transition(
        action=int(MiniGridActions.TURN_LEFT),
        label="turn_left",
        next_state=replace(state, agent_dir=(state.agent_dir - 1) % len(DIRECTION_VECTORS)),
    )
    yield Transition(
        action=int(MiniGridActions.TURN_RIGHT),
        label="turn_right",
        next_state=replace(state, agent_dir=(state.agent_dir + 1) % len(DIRECTION_VECTORS)),
    )

    front = _front_pos(state)
    key_id = _key_id_at(state, state.agent_pos)
    if key_id is not None and state.carrying_key is None:
        yield Transition(
            action=int(MiniGridActions.PICKUP),
            label=f"pickup:{key_id}",
            next_state=replace(
                state,
                carrying_key=key_id,
                collected_keys=state.collected_keys | {key_id},
                key_positions=_without_key(state.key_positions, key_id),
            ),
        )

    if ctx.drop_available and state.carrying_key is not None and _can_drop_here(ctx, state):
        key_id = state.carrying_key
        x, y = state.agent_pos
        yield Transition(
            action=int(MiniGridActions.DROP),
            label=f"drop:{key_id}",
            next_state=replace(
                state,
                carrying_key=None,
                collected_keys=state.collected_keys - {key_id},
                key_positions=state.key_positions | {(key_id, x, y)},
            ),
        )

    switch = ctx.switches_by_pos.get(state.agent_pos)
    if switch and switch["switch_type"] != "hold":
        toggled = _apply_switch(ctx, state, switch)
        if toggled is not None:
            yield Transition(
                action=int(MiniGridActions.TOGGLE),
                label=f"toggle:{switch['id']}",
                next_state=toggled,
            )

    if switch is None:
        door = ctx.doors_by_pos.get(front)
        if door and door["id"] not in state.open_doors and door["id"] not in state.shut_doors and state.carrying_key is not None:
            held_color = ctx.keys_by_id[state.carrying_key]["color"]
            if held_color == door["color"]:
                yield Transition(
                    action=int(MiniGridActions.TOGGLE),
                    label=f"open_door:{door['id']}",
                    next_state=replace(
                        state,
                        carrying_key=None if ctx.key_consumption else state.carrying_key,
                        open_doors=state.open_doors | {door["id"]},
                    ),
                )

    yield from _forward_successor(ctx, state, front)


def shortest_plan(
    ctx: TaskPlanningContext,
    start: PlannerState,
    is_goal: Callable[[PlannerState], bool],
    max_states: int = 500_000,
) -> tuple[list[int], PlannerState | None, int]:
    """Run BFS over ``successors`` and return the first shortest plan."""
    if is_goal(start):
        return [], start, 1

    queue = deque([start])
    parent: dict[PlannerState, tuple[PlannerState, int]] = {}
    visited = {start}

    while queue:
        if len(visited) >= max_states:
            return [], None, len(visited)
        state = queue.popleft()
        for transition in successors(ctx, state):
            if transition.next_state in visited:
                continue
            visited.add(transition.next_state)
            parent[transition.next_state] = (state, transition.action)
            if is_goal(transition.next_state):
                return (
                    _reconstruct_actions(parent, transition.next_state),
                    transition.next_state,
                    len(visited),
                )
            queue.append(transition.next_state)

    return [], None, len(visited)


def goal_still_reachable(ctx: TaskPlanningContext, state: PlannerState) -> bool:
    """Flood-fill from the agent to the goal. A closed door counts as open while
    a key of its colour still exists, a closed gate while one of its switches can
    still be used, and a portal pad leads to every end of its portal."""
    colors = {ctx.keys_by_id[key_id]["color"] for key_id, _, _ in state.key_positions}
    if state.carrying_key is not None:
        colors.add(ctx.keys_by_id[state.carrying_key]["color"])

    def passable(pos: tuple[int, int]) -> bool:
        if pos in ctx.walls or pos in state.sealed or pos in ctx.hazards or pos in ctx.blocks or pos in ctx.kill_cells:
            return False
        door = ctx.doors_by_pos.get(pos)
        if door and door["id"] not in state.open_doors:
            return door["id"] not in state.shut_doors and door["color"] in colors
        gate = ctx.gates_by_pos.get(pos)
        if gate and gate["id"] not in state.open_gates:
            return any(
                gate["id"] in switch["controls"]
                and switch["id"] not in state.shut_switches
                and switch["id"] not in state.used_switches
                for switch in ctx.switches_by_pos.values()
            )
        return True

    queue = deque([state.agent_pos])
    seen = {state.agent_pos}
    while queue:
        x, y = queue.popleft()
        if (x, y) == ctx.goal:
            return True
        nxt = [(x + dx, y + dy) for dx, dy in DIRECTION_VECTORS]
        if (x, y) in ctx.portal_at:
            nxt += ctx.portal_at[(x, y)][0]
        for pos in nxt:
            if pos not in seen and passable(pos):
                seen.add(pos)
                queue.append(pos)
    return False


def _front_pos(state: PlannerState) -> tuple[int, int]:
    dx, dy = DIRECTION_VECTORS[state.agent_dir]
    x, y = state.agent_pos
    return x + dx, y + dy


def _key_id_at(state: PlannerState, pos: tuple[int, int]) -> str | None:
    for key_id, x, y in state.key_positions:
        if (x, y) == pos:
            return key_id
    return None


def _without_key(
    key_positions: frozenset[tuple[str, int, int]], key_id: str
) -> frozenset[tuple[str, int, int]]:
    return frozenset(item for item in key_positions if item[0] != key_id)


def _can_drop_here(ctx: TaskPlanningContext, state: PlannerState) -> bool:
    """DROP needs an empty floor cell (same rule as the old MiniGrid handler)."""
    pos = state.agent_pos
    if pos in ctx.walls or pos in ctx.hazards or pos in ctx.blocks:
        return False
    if pos in ctx.switches_by_pos or pos in ctx.doors_by_pos or pos in ctx.gates_by_pos:
        return False
    if pos in ctx.portal_at or pos in ctx.kill_cells or pos in ctx.frozen_tiles or pos in ctx.rotating_index or pos == ctx.goal:
        return False
    return _key_id_at(state, pos) is None


def _apply_switch(
    ctx: TaskPlanningContext,
    state: PlannerState,
    switch: dict,
) -> PlannerState | None:
    switch_id = switch["id"]
    if switch_id in state.shut_switches:
        return None
    active = set(state.active_switches)
    used = set(state.used_switches)
    switch_type = switch["switch_type"]

    if switch_type == "one_shot":
        if switch_id in used:
            return None
        used.add(switch_id)
        active.add(switch_id)
    elif switch_type == "hold":
        active.add(switch_id)
    else:
        if switch_id in active:
            active.remove(switch_id)
        else:
            active.add(switch_id)

    active_fs = frozenset(active)
    return replace(
        state,
        active_switches=active_fs,
        used_switches=frozenset(used),
        open_gates=ctx.recompute_open_gates(active_fs),
    )


def _forward_successor(
    ctx: TaskPlanningContext,
    state: PlannerState,
    front: tuple[int, int],
) -> Iterable[Transition]:
    if _blocked(ctx, state, front):
        return

    next_pos = _landing(ctx, front, state.agent_dir)
    if next_pos in ctx.kill_cells:
        yield Transition(
            action=int(MiniGridActions.MOVE_FORWARD),
            label="kill_reset",
            next_state=ctx.initial_state(),
        )
        return

    yield Transition(
        action=int(MiniGridActions.MOVE_FORWARD),
        label="move_forward",
        next_state=_moved(ctx, state, next_pos),
    )


def _blocked(ctx: TaskPlanningContext, state: PlannerState, pos: tuple[int, int]) -> bool:
    return (
        pos in ctx.walls
        or pos in state.sealed
        or pos in ctx.hazards
        or pos in ctx.blocks
        or _has_closed_door(ctx, state, pos)
        or _has_closed_gate(ctx, state, pos)
        or _phase_closed(ctx, state.tick, pos)
    )


def _moved(ctx: TaskPlanningContext, state: PlannerState, next_pos: tuple[int, int]) -> PlannerState:
    active_switches = _active_switches_after_move(ctx, state, next_pos)
    return replace(
        state,
        agent_pos=next_pos,
        active_switches=active_switches,
        open_gates=ctx.recompute_open_gates(active_switches),
        freeze_remaining=ctx.freeze_steps if next_pos in ctx.frozen_tiles else 0,
    )


def _spin(dirs: tuple[int, ...]) -> tuple[int, ...]:
    return tuple((d + 1) % len(DIRECTION_VECTORS) for d in dirs)


def _settle(state: PlannerState) -> PlannerState:
    return replace(state, rotator_dirs=_spin(state.rotator_dirs))


def _ride_target(ctx: TaskPlanningContext, state: PlannerState) -> tuple[int, int] | None:
    """The cell a turntable ride from ``state`` enters, or None if the arrow is blocked."""
    dx, dy = DIRECTION_VECTORS[state.rotator_dirs[ctx.rotating_index[state.agent_pos]]]
    dest = (state.agent_pos[0] + dx, state.agent_pos[1] + dy)
    if _blocked(ctx, state, dest):
        return None
    return dest


def _landing(ctx: TaskPlanningContext, entered: tuple[int, int], direction: int) -> tuple[int, int]:
    """Where the agent ends up after entering ``entered``. A portal pad sends it on."""
    info = ctx.portal_at.get(entered)
    if info is None:
        return entered
    ends, index, bidirectional = info
    if len(ends) == 2:
        if index == 0:
            return ends[1]
        return ends[0] if bidirectional else entered
    x, y = ends[index]
    dx, dy = DIRECTION_VECTORS[direction]
    for other in (index - 1, index + 1):
        if 0 <= other < len(ends):
            ox, oy = ends[other]
            if (ox - x) * dx + (oy - y) * dy > 0:
                return ox, oy
    return entered


def _ride(ctx: TaskPlanningContext, state: PlannerState) -> PlannerState:
    dest = _ride_target(ctx, state)
    if dest is None:
        return _settle(state)
    direction = state.rotator_dirs[ctx.rotating_index[state.agent_pos]]
    next_pos = _landing(ctx, dest, direction)
    if next_pos in ctx.kill_cells:
        return ctx.initial_state()
    return _settle(_moved(ctx, state, next_pos))


def classify_mechanic(
    ctx: TaskPlanningContext,
    before: PlannerState,
    action: int,
    transition_label: str,
    after: PlannerState,
) -> dict | None:
    """Name the tile mechanic a transition triggered, for feedback and summaries.

    Returns one of, else None:

    * ``{"kind": "teleported", "from_xy", "to_xy", "color"}`` -- entered a portal
      pad (by MOVE_FORWARD or a turntable ride) and landed on its partner;
    * ``{"kind": "died", "cell_xy", "start_xy", "start_dir"}`` -- entered a death
      tile (``cell_xy`` is the tile actually entered, also via a portal or a
      ride) and the whole maze reset;
    * ``{"kind": "froze", "cell_xy", "steps"}`` -- the move onto an ice tile;
    * ``{"kind": "frozen", "remaining"}`` -- an action swallowed while frozen;
    * ``{"kind": "carried", "from_xy", "to_xy", "direction"}`` -- a turntable
      ride moved the agent (``direction`` is the arrow it rode);
    * ``{"kind": "rotating", "cell_xy", "pointing", "stayed"}`` -- the move onto
      a turntable (``stayed`` False), or a ride its arrow blocked (True);
      ``pointing`` is the arrow after this step's spin.

    The first that applies wins: died, teleported, froze, carried, rotating.
    Positions are (x, y); directions are 0..3 (EAST, SOUTH, WEST, NORTH).
    ``action`` is accepted for symmetry with ``successors``; the label and the
    two states decide everything.
    """
    if transition_label == "freeze":
        return {"kind": "frozen", "remaining": after.freeze_remaining}
    riding = transition_label == "rotate"
    if transition_label in ("move_forward", "kill_reset"):
        entered = _front_pos(before)
    elif riding:
        entered = _ride_target(ctx, before)
        if entered is None:
            return _rotating_label(ctx, after, before.agent_pos, stayed=True)
    else:
        return None

    if riding:
        direction = before.rotator_dirs[ctx.rotating_index[before.agent_pos]]
    else:
        direction = before.agent_dir
    landed = _landing(ctx, entered, direction)
    if landed in ctx.kill_cells:
        return {
            "kind": "died",
            "cell_xy": landed,
            "start_xy": after.agent_pos,
            "start_dir": after.agent_dir,
        }
    if landed != entered:
        return {
            "kind": "teleported",
            "from_xy": entered,
            "to_xy": landed,
            "color": _teleporter_color(ctx, entered),
        }
    if landed in ctx.frozen_tiles and after.freeze_remaining > 0:
        return {"kind": "froze", "cell_xy": landed, "steps": after.freeze_remaining}
    if riding:
        return {
            "kind": "carried",
            "from_xy": before.agent_pos,
            "to_xy": landed,
            "direction": before.rotator_dirs[ctx.rotating_index[before.agent_pos]],
        }
    if landed in ctx.rotating_index:
        return _rotating_label(ctx, after, landed, stayed=False)
    return None


def _rotating_label(
    ctx: TaskPlanningContext,
    after: PlannerState,
    cell: tuple[int, int],
    *,
    stayed: bool,
) -> dict:
    return {
        "kind": "rotating",
        "cell_xy": cell,
        "pointing": after.rotator_dirs[ctx.rotating_index[cell]],
        "stayed": stayed,
    }


def _teleporter_color(ctx: TaskPlanningContext, pad: tuple[int, int]) -> str:
    for teleporter in ctx.spec.mechanisms.teleporters:
        if pad in {pos.to_tuple() for pos in teleporter.ends()}:
            return teleporter.color
    raise KeyError(f"no teleporter at {pad}")


def _phase_closed(ctx: TaskPlanningContext, tick: int, pos: tuple[int, int]) -> bool:
    for cells, open_steps, closed_steps, offset in ctx.phased:
        if pos in cells and (tick + offset) % (open_steps + closed_steps) >= open_steps:
            return True
    return False


def _can_vanish(ctx: TaskPlanningContext, state: PlannerState, pos: tuple[int, int]) -> bool:
    if pos == ctx.goal or pos in ctx.walls or pos in ctx.phased_cells:
        return False
    if pos in ctx.doors_by_pos or pos in ctx.gates_by_pos or pos in ctx.switches_by_pos:
        return False
    if pos in ctx.portal_at or pos in ctx.kill_cells or pos in ctx.frozen_tiles or pos in ctx.rotating_index:
        return False
    if pos in ctx.hazards or pos in ctx.blocks or _key_id_at(state, pos) is not None:
        return False
    return True


def _tick(ctx: TaskPlanningContext, prev: PlannerState, nxt: PlannerState) -> PlannerState:
    sealed = set(nxt.sealed)
    vanish = {(x, y): left for x, y, left in nxt.vanish_at}
    if isfinite(ctx.disappear_after) and nxt.agent_pos != prev.agent_pos:
        pos = nxt.agent_pos
        if _can_vanish(ctx, nxt, pos) and pos not in sealed and pos not in vanish:
            vanish[pos] = int(ctx.disappear_after)
    armed = set()
    for (x, y), left in list(vanish.items()):
        just = (
            (x, y) == nxt.agent_pos
            and (x, y) not in {(px, py) for px, py, _ in prev.vanish_at}
            and (x, y) not in prev.sealed
        )
        if just:
            armed.add((x, y, left))
            continue
        left -= 1
        if left <= 0:
            sealed.add((x, y))
        else:
            armed.add((x, y, left))

    door_timers = dict(nxt.door_timers)
    switch_timers = dict(nxt.switch_timers)
    shut_doors = set(nxt.shut_doors)
    shut_switches = set(nxt.shut_switches)
    open_doors = set(nxt.open_doors)
    active = set(nxt.active_switches)
    if isfinite(ctx.shut_after):
        for door_id in nxt.open_doors - prev.open_doors:
            if door_id not in shut_doors:
                door_timers[door_id] = int(ctx.shut_after)
        for switch_id in nxt.active_switches - prev.active_switches:
            if switch_id not in shut_switches:
                switch_timers[switch_id] = int(ctx.shut_after)
        for door_id, left in list(door_timers.items()):
            if door_id in nxt.open_doors - prev.open_doors:
                continue
            left -= 1
            if left <= 0:
                door_timers.pop(door_id)
                open_doors.discard(door_id)
                shut_doors.add(door_id)
            else:
                door_timers[door_id] = left
        for switch_id, left in list(switch_timers.items()):
            if switch_id in nxt.active_switches - prev.active_switches:
                continue
            left -= 1
            if left <= 0:
                switch_timers.pop(switch_id)
                active.discard(switch_id)
                shut_switches.add(switch_id)
            else:
                switch_timers[switch_id] = left

    return replace(
        nxt,
        tick=(prev.tick + 1) % ctx.phase_period if ctx.phased else prev.tick,
        sealed=frozenset(sealed),
        vanish_at=frozenset(armed),
        door_timers=frozenset(door_timers.items()),
        switch_timers=frozenset(switch_timers.items()),
        shut_doors=frozenset(shut_doors),
        shut_switches=frozenset(shut_switches),
        open_doors=frozenset(open_doors),
        active_switches=frozenset(active),
        open_gates=ctx.recompute_open_gates(frozenset(active)),
    )


def _active_switches_after_move(
    ctx: TaskPlanningContext,
    state: PlannerState,
    next_pos: tuple[int, int],
) -> frozenset[str]:
    active = set(state.active_switches)
    for pos, switch in ctx.switches_by_pos.items():
        if switch["switch_type"] != "hold":
            continue
        if pos == next_pos:
            active.add(switch["id"])
        else:
            active.discard(switch["id"])
    return frozenset(active)


def _has_closed_door(
    ctx: TaskPlanningContext,
    state: PlannerState,
    pos: tuple[int, int],
) -> bool:
    door = ctx.doors_by_pos.get(pos)
    return door is not None and door["id"] not in state.open_doors


def _has_closed_gate(
    ctx: TaskPlanningContext,
    state: PlannerState,
    pos: tuple[int, int],
) -> bool:
    gate = ctx.gates_by_pos.get(pos)
    return gate is not None and gate["id"] not in state.open_gates


def _reconstruct_actions(
    parent: dict[PlannerState, tuple[PlannerState, int]],
    state: PlannerState,
) -> list[int]:
    actions = []
    while state in parent:
        state, action = parent[state]
        actions.append(action)
    actions.reverse()
    return actions


# Names tests and baselines historically imported from this module's old home.
_successors = successors
_shortest_plan = shortest_plan
