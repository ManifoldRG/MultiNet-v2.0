"""BFS test helper for experimental maze JSON specs.

This module preserves the legacy ``solve``/``find_all_paths`` API used by the
maze tests while delegating planning to the gridworld baseline BFS solver.
"""

from __future__ import annotations

from gridworld.actions import MiniGridActions
from gridworld.baselines import TaskPlanningContext, _shortest_plan, _successors
from gridworld.task_spec import TaskSpecification


def _to_task_spec(spec: dict | TaskSpecification) -> TaskSpecification:
    if isinstance(spec, TaskSpecification):
        return spec
    return TaskSpecification.from_dict(spec)


def _interaction_label(label: str) -> str | None:
    if label.startswith("pickup:"):
        return label
    if label.startswith("open_door:"):
        return f"open:{label.split(':', 1)[1]}"
    if label.startswith("toggle:"):
        return label
    return None


def solve(spec):
    """Return a shortest path result for a maze JSON spec."""
    task_spec = _to_task_spec(spec)
    ctx = TaskPlanningContext(task_spec)
    start_state = ctx.initial_state()
    actions, final_state = _shortest_plan(
        ctx,
        start_state,
        lambda state: state.agent_pos == ctx.goal,
    )

    if final_state is None:
        return {
            "is_solvable": False,
            "path": [],
            "interactions": [],
            "optimal_cost": None,
        }

    state = start_state
    path = [state.agent_pos]
    interactions = []
    include_pickup_positions = ctx.goal in ctx.doors_by_pos

    for action in actions:
        transition = next(
            candidate
            for candidate in _successors(ctx, state)
            if candidate.action == action
        )
        label = _interaction_label(transition.label)
        if label is not None:
            interactions.append(label)
            if include_pickup_positions and transition.label.startswith("pickup:"):
                key_id = transition.label.split(":", 1)[1]
                key_pos = ctx.keys_by_id[key_id]["position"]
                if path[-1] != key_pos:
                    path.append(key_pos)
        state = transition.next_state
        if action == int(MiniGridActions.MOVE_FORWARD):
            path.append(state.agent_pos)

    return {
        "is_solvable": True,
        "path": path,
        "interactions": interactions,
        "optimal_cost": len(path) - 1,
    }


def find_all_paths(spec):
    """Return the BFS solver path in the legacy list-of-paths test-helper shape."""
    result = solve(spec)
    if not result["is_solvable"]:
        return []
    return [result["path"]]
