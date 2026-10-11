"""Deterministic gridworld baseline agents.

Search (BFS / greedy) over the R1 rulebook in ``world_model``. Physics live
there; this module only plans.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from model_interface import ModelInput, ModelInterface, ModelOutput

from .actions import MiniGridActions
from .task_spec import TaskSpecification
from .world_model import (
    PlannerState,
    TaskPlanningContext,
    Transition,
    apply,
    shortest_plan,
    successors,
)

# Re-exports: tests and scorer imported these from baselines.
_successors = successors
_shortest_plan = shortest_plan
_apply = apply


@dataclass(frozen=True)
class PlannedPath:
    """Planner output with replayed positions for scorer/reporting artifacts."""

    success: bool
    actions: list[int]
    action_labels: list[str]
    positions: list[tuple[int, int]]
    states_explored: int = 0


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
        for transition in successors(ctx, state):
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


def _bfs_actions_with_stats(
    spec: TaskSpecification, *, drop_available: bool = False
) -> tuple[list[int], int]:
    ctx = TaskPlanningContext(spec, drop_available=drop_available)
    actions, _, states_explored = shortest_plan(
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
            chunk, next_state, _ = shortest_plan(
                ctx,
                state,
                lambda st: st.agent_pos == ctx.goal,
            )
        if next_state is None or not chunk:
            break
        actions.extend(chunk)
        state = next_state

    return actions


def trace_planned_actions(
    spec: TaskSpecification, actions: list[int], *, drop_available: bool = False
) -> PlannedPath:
    """Replay planner actions through the R1 graph without running a backend."""
    ctx = TaskPlanningContext(spec, drop_available=drop_available)
    state = ctx.initial_state()
    positions = [state.agent_pos]
    executed_actions: list[int] = []
    labels: list[str] = []

    for action in actions:
        if action == int(MiniGridActions.DONE):
            break
        executed_actions.append(action)
        transition = next(
            (candidate for candidate in successors(ctx, state) if candidate.action == action),
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


def plan_bfs_path(spec: TaskSpecification, *, drop_available: bool = False) -> PlannedPath:
    """Return the BFS baseline plan plus replayed positions."""
    actions, states_explored = _bfs_actions_with_stats(spec, drop_available=drop_available)
    path = trace_planned_actions(spec, actions, drop_available=drop_available)
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
