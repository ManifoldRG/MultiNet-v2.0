"""Canonical solver integration for scorer artifacts."""

from __future__ import annotations

from typing import Any

from gridworld.baselines import PlannedPath, plan_bfs_path, plan_greedy_path
from gridworld.task_spec import TaskSpecification

from .artifacts import CanonicalPathReport
from .config import SCORER_VERSION
from .io import stable_hash


def _path_payload(path) -> dict[str, Any]:
    return {
        "success": path.success,
        "actions": list(path.action_labels),
        "positions": [list(pos) for pos in path.positions],
        "steps": len(path.action_labels),
    }


def require_scorable_spec(spec: TaskSpecification) -> None:
    """Reject malformed tasks before canonical planners inspect their coordinates."""
    schema_valid, schema_errors = spec.validate()
    if not schema_valid:
        detail = "; ".join(schema_errors)
        raise ValueError(f"Task {spec.task_id!r} failed schema validation: {detail}")


def compute_canonical_paths(
    spec: TaskSpecification,
    bfs_path: PlannedPath | None = None,
    greedy_path: PlannedPath | None = None,
) -> CanonicalPathReport:
    """Emit canonical BFS and greedy traces using the merged baseline solvers."""
    require_scorable_spec(spec)
    if bfs_path is None:
        bfs_path = plan_bfs_path(spec)
    if greedy_path is None:
        greedy_path = plan_greedy_path(spec)

    if bfs_path.success:
        message = (
            f"Solution found in {len(bfs_path.action_labels)} steps "
            f"({bfs_path.states_explored} states explored)"
        )
    elif bfs_path.states_explored:
        message = (
            "No solution found "
            f"({bfs_path.states_explored} states explored, all reachable states checked)"
        )
    else:
        message = "No solution found"

    inputs_hash = stable_hash({"task": spec.to_dict(), "scorer_version": SCORER_VERSION})

    return CanonicalPathReport(
        task_id=spec.task_id,
        success=bfs_path.success,
        actions=list(bfs_path.action_labels),
        positions=list(bfs_path.positions),
        optimal_steps=len(bfs_path.action_labels) if bfs_path.success else 0,
        states_explored=bfs_path.states_explored,
        message=message,
        greedy=_path_payload(greedy_path),
        inputs_hash=inputs_hash,
    )


def compute_greedy_solvability(
    spec: TaskSpecification,
    greedy_path: PlannedPath | None = None,
) -> float:
    """Return 1 when the merged greedy planner solves the task, else 0."""
    if greedy_path is None:
        greedy_path = plan_greedy_path(spec)
    return 1.0 if greedy_path.success else 0.0
