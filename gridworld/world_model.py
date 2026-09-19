"""R1 gridworld rulebook: pure (state, action) -> state.

BFS searches this. MiniGrid ``step`` applies this then writes the result
back onto the live env. The validator uses the same graph. Nothing here
holds a MiniGrid instance.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
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
        self.teleporters = {}
        for teleporter in spec.mechanisms.teleporters:
            pos_a = teleporter.position_a.to_tuple()
            pos_b = teleporter.position_b.to_tuple()
            self.teleporters[pos_a] = pos_b
            if teleporter.bidirectional:
                self.teleporters[pos_b] = pos_a

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
        )

    def recompute_open_gates(self, active_switches: frozenset[str]) -> frozenset[str]:
        """Open gates controlled by active switches, plus gates initially open."""
        open_gates = {
            gate["id"] for gate in self.gates_by_pos.values() if gate["open"]
        }
        for switch in self.switches_by_pos.values():
            if switch["id"] in active_switches:
                open_gates.update(switch["controls"])
        return frozenset(open_gates)


def apply(ctx: TaskPlanningContext, state: PlannerState, action: int) -> PlannerState:
    """Return the next world state for ``action``. Illegal actions are no-ops."""
    action = int(action)
    for transition in successors(ctx, state):
        if transition.action == action:
            return transition.next_state
    return state


def successors(ctx: TaskPlanningContext, state: PlannerState) -> Iterable[Transition]:
    """Generate legal R1 actions from a world state."""
    if state.freeze_remaining > 0:
        thawed = replace(state, freeze_remaining=state.freeze_remaining - 1)
        for action in MiniGridActions:
            yield Transition(int(action), "freeze", thawed)
        return

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
        if door and door["id"] not in state.open_doors and state.carrying_key is not None:
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
) -> tuple[list[int], PlannerState | None, int]:
    """Run BFS over ``successors`` and return the first shortest plan."""
    if is_goal(start):
        return [], start, 1

    queue = deque([start])
    parent: dict[PlannerState, tuple[PlannerState, int]] = {}
    visited = {start}

    while queue:
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
    if pos in ctx.teleporters or pos in ctx.kill_cells or pos in ctx.frozen_tiles or pos == ctx.goal:
        return False
    return _key_id_at(state, pos) is None


def _apply_switch(
    ctx: TaskPlanningContext,
    state: PlannerState,
    switch: dict,
) -> PlannerState | None:
    switch_id = switch["id"]
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
    if (
        front in ctx.walls
        or front in ctx.hazards
        or front in ctx.blocks
        or _has_closed_door(ctx, state, front)
        or _has_closed_gate(ctx, state, front)
    ):
        return

    next_pos = ctx.teleporters.get(front, front)
    if next_pos in ctx.kill_cells:
        yield Transition(
            action=int(MiniGridActions.MOVE_FORWARD),
            label="kill_reset",
            next_state=ctx.initial_state(),
        )
        return

    active_switches = _active_switches_after_move(ctx, state, next_pos)
    freeze_remaining = ctx.freeze_steps if next_pos in ctx.frozen_tiles else 0
    yield Transition(
        action=int(MiniGridActions.MOVE_FORWARD),
        label="move_forward",
        next_state=replace(
            state,
            agent_pos=next_pos,
            active_switches=active_switches,
            open_gates=ctx.recompute_open_gates(active_switches),
            freeze_remaining=freeze_remaining,
        ),
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
