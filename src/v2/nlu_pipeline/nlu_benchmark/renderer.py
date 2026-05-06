"""Maze text split: **initial** layout (system) vs **current** situation (user turn)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from nlu_benchmark.env import GridState

_RENDER_DATASET_MOD = None


def _render_dataset_module():
    """Load ``automatic_maze_generation/render_dataset.py`` without requiring ``v2`` on ``PYTHONPATH``."""
    global _RENDER_DATASET_MOD
    if _RENDER_DATASET_MOD is None:
        path = Path(__file__).resolve().parents[2] / "automatic_maze_generation" / "render_dataset.py"
        name = "_multinet_automatic_maze_generation_render_dataset"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load maze renderer from {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _RENDER_DATASET_MOD = mod
    return _RENDER_DATASET_MOD


def _internal_pos_to_json_list(pos: tuple[int, int]) -> list[int]:
    """Env ``(row, column)`` → JSON ``[x, y]`` = ``[column, row]`` (standard Cartesian order)."""
    row, col = pos
    return [col, row]


def _mechanism_dict_for_payload(item: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    if "position" in out:
        out["position"] = _internal_pos_to_json_list(tuple(out["position"]))
    return out


def _grid_state_to_maze_payload(state: GridState, *, task_id: str = "") -> dict:
    """JSON-shaped maze dict for ``render_maze_payload`` / ``render_maze_payload_bytes``."""
    out: dict[str, Any] = {
        "maze": {
            # Task JSON ``[x, y]`` = ``[column, row]`` (``dimensions`` are ``[rows, cols]``).
            "dimensions": [state.rows, state.cols],
            "walls": [_internal_pos_to_json_list(w) for w in sorted(state.walls)],
            "start": _internal_pos_to_json_list(state.start),
            "goal": _internal_pos_to_json_list(state.goal),
        },
        "mechanisms": {
            "keys": [_mechanism_dict_for_payload(k) for k in state.keys],
            "doors": [_mechanism_dict_for_payload(d) for d in state.doors],
            "switches": [_mechanism_dict_for_payload(s) for s in state.switches],
            "gates": [_mechanism_dict_for_payload(g) for g in state.gates],
        },
    }
    if task_id:
        out["task_id"] = task_id
    return out


def _static_layout_lines(state: GridState) -> list[str]:
    wall_str = ", ".join(f"({row},{col})" for row, col in sorted(state.walls)) or "none"
    return [
        f"The world is a {state.rows} by {state.cols} grid.",
        "Coordinates: JSON lists use ``[x, y]`` (east, south) from the **top-left** corner ``(1, 1)``;"
        " tuples in this text use ``(row, column)`` matching env state (row southward, column east)."
        " So ``x`` = column index, ``y`` = row index (e.g. goal ``[2, 12]`` is the cell ``(12, 2)``).",
        f"The start is at {state.start}.",
        f"The goal is at {state.goal}.",
        f"The following cells are walls: {wall_str}.",
    ]


def _mechanism_lines(state: GridState) -> list[str]:
    parts: list[str] = []
    for key in state.keys:
        row, col = key["position"]
        parts.append(f"There is a {key['color']} key at ({row},{col}).")

    for door in state.doors:
        row, col = door["position"]
        parts.append(
            f"There is a locked {door['requires_key']} door at ({row},{col})."
            f" It requires the {door['requires_key']} key to open."
        )

    for switch in state.switches:
        row, col = switch["position"]
        controls = ", ".join(switch.get("controls", []))
        on_off = "on" if switch.get("on") else "off"
        parts.append(
            f"There is a {switch.get('switch_type', 'toggle')} switch at ({row},{col}) (currently {on_off})."
            f" It controls: {controls}."
        )

    for gate in state.gates:
        row, col = gate["position"]
        cur = gate.get("state", gate.get("initial_state", "closed"))
        parts.append(
            f"There is a gate ({gate['id']}) at ({row},{col})."
            f" It is currently {cur} (initially {gate.get('initial_state', 'closed')})."
        )
    return parts


def render_initial_maze_text(state: GridState) -> str:
    """Episode layout for the **system** prompt. Pass ``state`` from ``env.reset()``."""
    return "\n".join(_static_layout_lines(state) + _mechanism_lines(state))


def render_user_observation_text(state: GridState) -> str:
    """**Current** state for the **user** turn (text or image+text modes)."""
    inv = ", ".join(state.inventory) if state.inventory else "empty"
    head = [
        "Current situation (this step):",
        f"The goal is at {state.goal}.",
        f"You are at {state.agent_pos} facing {state.facing}.",
        "Environment steps used so far: "
        f"{state.step_count} (max {state.max_steps} before timeout).",
        f"Your inventory: {inv}.",
        "",
        "Map contents as of this step (keys on the ground, doors, switches, gates):",
    ]
    mech = _mechanism_lines(state)
    if mech:
        head.extend(mech)
    else:
        head.append("(No keys on the ground, doors, switches, or gates in the current state description.)")
    return "\n".join(head)


def render_maze_image_png_bytes(state: GridState, *, task_id: str = "") -> bytes:
    """Render the current ``GridState`` to a PNG (same style as ``render_dataset.render_maze_payload``).

    ``task_id`` is only for the optional figure title (smoke replay uses the JSON id; LLM observations
    default to empty so the title does not change ``tight_layout`` / margins).
    """
    mod = _render_dataset_module()
    payload = _grid_state_to_maze_payload(state, task_id=task_id)
    row, col = state.agent_pos
    return mod.render_maze_payload_bytes(
        payload,
        dpi=150,
        agent_pos=(col, row),
        facing=state.facing,
    )


def render_task_json_with_solver_path_png(
    task_data: dict[str, Any],
    solver_path_xy: list[tuple[int, int]],
    output_path: Path,
) -> None:
    """
    One static figure like ``automatic_maze_generation/render_dataset.py`` / ``main()``:
    maze + mechanisms + semi-transparent optimal route.

    ``solver_path_xy`` is ``solve_maze(...)["path"]`` (mazegen 0-based ``(x, y)``; ``x`` = column index, ``y`` = row index).
    """
    optimal_path_cells = [[x + 1, y + 1] for (x, y) in solver_path_xy]
    payload = {
        **task_data,
        "validation": {
            **task_data.get("validation", {}),
            "optimal_path": optimal_path_cells,
        },
    }
    mod = _render_dataset_module()
    mod.render_maze_payload(payload, output_path)
