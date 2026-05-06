"""Build mazegen solver plan for a task JSON maze and replay it in :class:`~nlu_benchmark.env.GridWorldEnv` with PNG traces (same outputs as ``smoke_bfs``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from automatic_maze_generation.mazegen.solver import solve_maze
from nlu_benchmark.env import FACING_ORDER, FACING_TO_DELTA
from nlu_benchmark.loader import (
    grid_state_to_maze_instance,
    load_maze_from_dict,
    task_dict_shrink_dimensions_minus_two,
)
from nlu_benchmark.smoke_trace import trace_prepare, trace_reset, trace_step, trace_write_text_artifacts


def path_to_actions(path: list[tuple[int, int]], start_facing: str = "NORTH") -> list[str]:
    if not path or len(path) < 2:
        return ["DONE"]
    facing = start_facing
    actions: list[str] = []
    for (r, c), (nr, nc) in zip(path, path[1:]):
        dr, dc = nr - r, nc - c
        target = next((f for f, d in FACING_TO_DELTA.items() if d == (dr, dc)), None)
        if target is None:
            continue
        cur_idx = FACING_ORDER.index(facing)
        tgt_idx = FACING_ORDER.index(target)
        diff = (tgt_idx - cur_idx) % 4
        if diff == 1:
            actions.append("TURN_RIGHT")
        elif diff == 2:
            actions.extend(["TURN_RIGHT", "TURN_RIGHT"])
        elif diff == 3:
            actions.append("TURN_LEFT")
        actions.append("MOVE_FORWARD")
        facing = target
    actions.append("DONE")
    return actions


def xy_path_to_rc(path_xy: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """mazegen 0-based ``(x, y)`` (y south from north edge) → NLU ``(row, column)`` with row southward."""
    return [(y + 1, x + 1) for (x, y) in path_xy]


def inject_pickups(actions: list[str], env: Any, state: Any) -> list[str]:
    """NLU env needs explicit PICKUP; solver assumes pickup-on-entry."""
    out: list[str] = []
    sim_state = state
    for a in actions:
        has_key_here = any(tuple(k["position"]) == sim_state.agent_pos for k in sim_state.keys)
        if has_key_here and a != "PICKUP":
            out.append("PICKUP")
            sim_state, _ = env.step("PICKUP")
        out.append(a)
        sim_state, _ = env.step(a)
    return out


def write_png_trace_for_maze_json(maze_path: Path, out_dir: Path) -> dict[str, Any]:
    """
    Solve ``maze_path``, replay the plan in the NLU env, write ``step_*.png``, ``run_log.txt``, ``plan.txt`` under ``out_dir``.

    Applies :func:`~nlu_benchmark.loader.task_dict_shrink_dimensions_minus_two` to the task JSON first
    (same convention as benchmark maze smoke).

    Returns a dict with keys ``ok`` (bool), ``optimal_cost``, ``success``, ``steps_used``, ``out_dir``,
    and on failure ``reason`` (str) or ``solver_result``.
    """
    trace_prepare(out_dir)
    raw: dict[str, Any] = task_dict_shrink_dimensions_minus_two(
        json.loads(maze_path.read_text(encoding="utf-8"))
    )
    env_plan = load_maze_from_dict(raw)
    plan_state = env_plan.reset()
    maze_inst = grid_state_to_maze_instance(plan_state)
    solver_result = solve_maze(maze_inst)
    if not solver_result.get("is_solvable"):
        return {
            "ok": False,
            "out_dir": out_dir,
            "solver_result": solver_result,
            "reason": "solver reported unsolvable",
        }

    path_rc = xy_path_to_rc(solver_result.get("path", []))
    planned_actions = path_to_actions(path_rc, start_facing="NORTH")
    executable_actions = inject_pickups(planned_actions, env_plan, plan_state)

    env = load_maze_from_dict(raw)
    state = env.reset()
    lines = trace_reset(out_dir, state)

    for step, action in enumerate(executable_actions, start=1):
        before = state.agent_pos
        state, event = trace_step(out_dir, lines, step, action, env, position_before=before)
        if event.type == "DONE":
            break

    trace_write_text_artifacts(out_dir, lines, executable_actions)
    success = state.agent_pos == state.goal
    return {
        "ok": True,
        "out_dir": out_dir,
        "optimal_cost": solver_result.get("optimal_cost"),
        "success": success,
        "steps_used": state.step_count,
    }
