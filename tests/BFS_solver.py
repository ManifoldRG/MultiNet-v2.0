"""BFS test helper for experimental maze JSON specs.

This module preserves the legacy ``solve``/``find_all_paths`` API used by the
maze tests while delegating planning to the gridworld baseline BFS solver.
"""

from __future__ import annotations

from gridworld.baselines import plan_bfs_path
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


def _movement_path(positions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path = []
    for position in positions:
        if not path or path[-1] != position:
            path.append(position)
    return path


def solve(spec):
    """Return a shortest path result for a maze JSON spec."""
    task_spec = _to_task_spec(spec)
    planned = plan_bfs_path(task_spec)

    if not planned.success:
        return {
            "is_solvable": False,
            "path": [],
            "interactions": [],
            "optimal_cost": None,
        }

    path = _movement_path(planned.positions)
    interactions = [
        interaction
        for label in planned.action_labels
        if (interaction := _interaction_label(label)) is not None
    ]

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
