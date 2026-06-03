"""Deterministic gridworld baseline agents.

The baselines in this module use the same ``ModelInterface`` path as VLM
adapters, so they can be run by ``EvaluationHarness`` and ``run_eval.py``.
They plan over the actual MiniGrid action space: turn, move, pickup, toggle,
drop, and done.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable

from model_interface import ModelInput, ModelInterface, ModelOutput

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
    """Compact state used by the baseline planners."""

    agent_pos: tuple[int, int]
    agent_dir: int
    carrying_key: str | None
    collected_keys: frozenset[str]
    active_switches: frozenset[str]
    used_switches: frozenset[str]
    open_gates: frozenset[str]
    open_doors: frozenset[str]


@dataclass(frozen=True)
class Transition:
    """One executable action in the planner search graph."""

    action: int
    label: str
    next_state: PlannerState


@dataclass(frozen=True)
class PlannedPath:
    """Planner output with replayed positions for scorer/reporting artifacts."""

    success: bool
    actions: list[int]
    action_labels: list[str]
    positions: list[tuple[int, int]]
    states_explored: int = 0


class TaskPlanningContext:
    """Fast lookup tables derived from a ``TaskSpecification``."""

    def __init__(self, spec: TaskSpecification):
        self.spec = spec
        self.width, self.height = spec.maze.dimensions
        self.goal = spec.maze.goal.to_tuple()
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
        """Return the planner state at the beginning of the episode."""
        return PlannerState(
            agent_pos=self.start,
            agent_dir=0,
            carrying_key=None,
            collected_keys=frozenset(),
            active_switches=self.initial_active_switches,
            used_switches=self.initial_used_switches,
            open_gates=self.recompute_open_gates(self.initial_active_switches),
            open_doors=self.initial_open_doors,
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


def _front_pos(state: PlannerState) -> tuple[int, int]:
    dx, dy = DIRECTION_VECTORS[state.agent_dir]
    x, y = state.agent_pos
    return x + dx, y + dy


def _apply_switch(
    ctx: TaskPlanningContext,
    state: PlannerState,
    switch: dict,
) -> PlannerState | None:
    """Return the state after toggling a switch, or ``None`` if unavailable."""
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
    return PlannerState(
        agent_pos=state.agent_pos,
        agent_dir=state.agent_dir,
        carrying_key=state.carrying_key,
        collected_keys=state.collected_keys,
        active_switches=active_fs,
        used_switches=frozenset(used),
        open_gates=ctx.recompute_open_gates(active_fs),
        open_doors=state.open_doors,
    )


def _successors(ctx: TaskPlanningContext, state: PlannerState) -> Iterable[Transition]:
    """Generate valid MiniGrid actions from a planner state."""
    yield Transition(
        action=int(MiniGridActions.TURN_LEFT),
        label="turn_left",
        next_state=PlannerState(
            agent_pos=state.agent_pos,
            agent_dir=(state.agent_dir - 1) % len(DIRECTION_VECTORS),
            carrying_key=state.carrying_key,
            collected_keys=state.collected_keys,
            active_switches=state.active_switches,
            used_switches=state.used_switches,
            open_gates=state.open_gates,
            open_doors=state.open_doors,
        ),
    )
    yield Transition(
        action=int(MiniGridActions.TURN_RIGHT),
        label="turn_right",
        next_state=PlannerState(
            agent_pos=state.agent_pos,
            agent_dir=(state.agent_dir + 1) % len(DIRECTION_VECTORS),
            carrying_key=state.carrying_key,
            collected_keys=state.collected_keys,
            active_switches=state.active_switches,
            used_switches=state.used_switches,
            open_gates=state.open_gates,
            open_doors=state.open_doors,
        ),
    )

    front = _front_pos(state)
    key = ctx.keys_by_pos.get(front)
    if key and key["id"] not in state.collected_keys and state.carrying_key is None:
        yield Transition(
            action=int(MiniGridActions.PICKUP),
            label=f"pickup:{key['id']}",
            next_state=PlannerState(
                agent_pos=state.agent_pos,
                agent_dir=state.agent_dir,
                carrying_key=key["id"],
                collected_keys=state.collected_keys | {key["id"]},
                active_switches=state.active_switches,
                used_switches=state.used_switches,
                open_gates=state.open_gates,
                open_doors=state.open_doors,
            ),
        )

    door = ctx.doors_by_pos.get(front)
    if door and door["id"] not in state.open_doors and state.carrying_key is not None:
        held_color = ctx.keys_by_id[state.carrying_key]["color"]
        if held_color == door["color"]:
            yield Transition(
                action=int(MiniGridActions.TOGGLE),
                label=f"open_door:{door['id']}",
                next_state=PlannerState(
                    agent_pos=state.agent_pos,
                    agent_dir=state.agent_dir,
                    carrying_key=None if ctx.key_consumption else state.carrying_key,
                    collected_keys=state.collected_keys,
                    active_switches=state.active_switches,
                    used_switches=state.used_switches,
                    open_gates=state.open_gates,
                    open_doors=state.open_doors | {door["id"]},
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

    yield from _forward_successor(ctx, state, front)


def _forward_successor(
    ctx: TaskPlanningContext,
    state: PlannerState,
    front: tuple[int, int],
) -> Iterable[Transition]:
    """Yield a forward move if the target cell is passable."""
    if (
        front in ctx.walls
        or front in ctx.hazards
        or front in ctx.blocks
        or _has_uncollected_key(ctx, state, front)
        or _has_closed_door(ctx, state, front)
        or _has_closed_gate(ctx, state, front)
    ):
        return

    next_pos = ctx.teleporters.get(front, front)
    active_switches = _active_switches_after_move(ctx, state, next_pos)
    yield Transition(
        action=int(MiniGridActions.MOVE_FORWARD),
        label="move_forward",
        next_state=PlannerState(
            agent_pos=next_pos,
            agent_dir=state.agent_dir,
            carrying_key=state.carrying_key,
            collected_keys=state.collected_keys,
            active_switches=active_switches,
            used_switches=state.used_switches,
            open_gates=ctx.recompute_open_gates(active_switches),
            open_doors=state.open_doors,
        ),
    )


def _active_switches_after_move(
    ctx: TaskPlanningContext,
    state: PlannerState,
    next_pos: tuple[int, int],
) -> frozenset[str]:
    """Apply hold-switch semantics after a forward movement."""
    active = set(state.active_switches)
    for pos, switch in ctx.switches_by_pos.items():
        if switch["switch_type"] != "hold":
            continue
        if pos == next_pos:
            active.add(switch["id"])
        else:
            active.discard(switch["id"])
    return frozenset(active)


def _has_uncollected_key(
    ctx: TaskPlanningContext,
    state: PlannerState,
    pos: tuple[int, int],
) -> bool:
    key = ctx.keys_by_pos.get(pos)
    return key is not None and key["id"] not in state.collected_keys


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


def _shortest_plan(
    ctx: TaskPlanningContext,
    start: PlannerState,
    is_goal: Callable[[PlannerState], bool],
) -> tuple[list[int], PlannerState | None, int]:
    """Run BFS over executable actions and return the first shortest plan."""
    if is_goal(start):
        return [], start, 1

    queue = deque([start])
    parent: dict[PlannerState, tuple[PlannerState, int]] = {}
    visited = {start}

    while queue:
        state = queue.popleft()
        for transition in _successors(ctx, state):
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


def _shortest_plan_to_interaction(
    ctx: TaskPlanningContext,
    start: PlannerState,
) -> tuple[list[int], PlannerState | None]:
    """Find the nearest useful key, door, switch, or goal interaction."""
    queue = deque([start])
    parent: dict[PlannerState, tuple[PlannerState, int]] = {}
    visited = {start}

    while queue:
        state = queue.popleft()
        for transition in _successors(ctx, state):
            if transition.next_state in visited:
                continue
            visited.add(transition.next_state)
            parent[transition.next_state] = (state, transition.action)
            if _is_useful_interaction(ctx, state, transition):
                return _reconstruct_actions(parent, transition.next_state), transition.next_state
            queue.append(transition.next_state)

    return [], None


def _is_useful_interaction(
    ctx: TaskPlanningContext,
    state: PlannerState,
    transition: Transition,
) -> bool:
    """Identify the next local objective for the greedy baseline."""
    if transition.next_state.agent_pos == ctx.goal:
        return True
    if transition.label.startswith("open_door:"):
        return True
    if transition.label.startswith("pickup:"):
        key_id = transition.label.split(":", 1)[1]
        key_color = ctx.keys_by_id[key_id]["color"]
        return any(
            door["color"] == key_color and door["id"] not in state.open_doors
            for door in ctx.doors_by_pos.values()
        )
    if transition.label.startswith("toggle:"):
        switch_id = transition.label.split(":", 1)[1]
        before = set(state.open_gates)
        after = set(transition.next_state.open_gates)
        return switch_id in transition.next_state.active_switches and bool(after - before)
    return False


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


def _bfs_actions(spec: TaskSpecification) -> list[int]:
    actions, _ = _bfs_actions_with_stats(spec)
    return actions


def _bfs_actions_with_stats(spec: TaskSpecification) -> tuple[list[int], int]:
    ctx = TaskPlanningContext(spec)
    actions, _, states_explored = _shortest_plan(
        ctx,
        ctx.initial_state(),
        lambda st: st.agent_pos == ctx.goal,
    )
    return actions, states_explored


def _greedy_actions(spec: TaskSpecification) -> list[int]:
    ctx = TaskPlanningContext(spec)
    state = ctx.initial_state()
    actions: list[int] = []

    for _ in range(spec.max_steps):
        if state.agent_pos == ctx.goal:
            break
        chunk, next_state = _shortest_plan_to_interaction(ctx, state)
        if next_state is None:
            chunk, next_state, _ = _shortest_plan(
                ctx,
                state,
                lambda st: st.agent_pos == ctx.goal,
            )
        if next_state is None or not chunk:
            break
        actions.extend(chunk)
        state = next_state

    return actions


def trace_planned_actions(spec: TaskSpecification, actions: list[int]) -> PlannedPath:
    """Replay planner actions through the planner graph without running a backend."""
    ctx = TaskPlanningContext(spec)
    state = ctx.initial_state()
    positions = [state.agent_pos]
    executed_actions: list[int] = []
    labels: list[str] = []

    for action in actions:
        if action == int(MiniGridActions.DONE):
            break
        executed_actions.append(action)
        transition = next(
            (candidate for candidate in _successors(ctx, state) if candidate.action == action),
            None,
        )
        if transition is None:
            labels.append(f"invalid:{action}")
            return PlannedPath(
                success=False,
                actions=executed_actions,
                action_labels=labels,
                positions=positions,
            )
        labels.append(transition.label)
        state = transition.next_state
        positions.append(state.agent_pos)

    return PlannedPath(
        success=state.agent_pos == ctx.goal,
        actions=executed_actions,
        action_labels=labels,
        positions=positions,
    )


def plan_bfs_actions(spec: TaskSpecification) -> list[int]:
    """Return the deterministic BFS baseline action plan."""
    return _bfs_actions(spec)


def plan_greedy_actions(spec: TaskSpecification) -> list[int]:
    """Return the deterministic greedy baseline action plan."""
    return _greedy_actions(spec)


def plan_bfs_path(spec: TaskSpecification) -> PlannedPath:
    """Return the BFS baseline plan plus replayed positions."""
    actions, states_explored = _bfs_actions_with_stats(spec)
    path = trace_planned_actions(spec, actions)
    return PlannedPath(
        success=path.success,
        actions=path.actions,
        action_labels=path.action_labels,
        positions=path.positions,
        states_explored=states_explored,
    )


def plan_greedy_path(spec: TaskSpecification) -> PlannedPath:
    """Return the greedy baseline plan plus replayed positions."""
    return trace_planned_actions(spec, plan_greedy_actions(spec))


class PlannedBaselineModel(ModelInterface):
    """Base class for deterministic baselines that precompute an action plan."""

    baseline_name = "planned"

    def __init__(self):
        self._task_id: str | None = None
        self._actions: list[int] = []
        self._cursor = 0

    @property
    def model_name(self) -> str:
        return self.baseline_name

    def predict(self, input: ModelInput) -> ModelOutput:
        if input.task_spec is None:
            raise ValueError(f"{self.model_name} baseline requires ModelInput.task_spec")

        task_id = getattr(input.task_spec, "task_id", "unknown")
        if task_id != self._task_id:
            self._task_id = task_id
            self._actions = self._build_plan(input.task_spec)
            self._cursor = 0

        if self._cursor >= len(self._actions):
            action = int(MiniGridActions.DONE)
        else:
            action = self._actions[self._cursor]
            self._cursor += 1

        return ModelOutput(
            action=action,
            confidence=1.0,
            reasoning=f"{self.model_name} planned action {self._cursor}/{len(self._actions)}",
        )

    def _build_plan(self, spec: TaskSpecification) -> list[int]:
        raise NotImplementedError


class BFSModelInterface(PlannedBaselineModel):
    """Shortest-path baseline over the executable gridworld action space."""

    baseline_name = "bfs"

    def _build_plan(self, spec: TaskSpecification) -> list[int]:
        return _bfs_actions(spec)


class GreedyModelInterface(PlannedBaselineModel):
    """Greedy baseline that moves to the nearest useful objective first."""

    baseline_name = "greedy"

    def _build_plan(self, spec: TaskSpecification) -> list[int]:
        return _greedy_actions(spec)
