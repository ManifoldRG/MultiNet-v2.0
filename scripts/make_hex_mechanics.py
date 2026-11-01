"""Convert mechanics-15 mazes to the hex tiling, for the hex smoke run.

Each square maze is re-read as pointy-top hexes in the odd-r offset layout:
cell (x, y) becomes column x of row y, so the walls, the red key -> red door ->
yellow switch -> gate -> blue key -> blue door chain and every mechanic cell
stay where they were, and the only change is how cells connect. Square
neighbours stay neighbours (east/west in the row; north/south become the
diagonal into the next row), and each cell gains two diagonal neighbours, so
"vertical" corridors zigzag and some corners open up. A rotating tile keeps
the cell its arrow first points at (its arrow then turns through six
directions instead of four).

The script refuses a conversion the planner cannot solve, whose BFS falls
outside R2's range, or that frees a key or switch the square maze needs (a
corner opening that bypasses a door or gate), then writes the mazes and
their manifest:

    python -m scripts.make_hex_mechanics            # write mazes + manifest
    python -m scripts.make_hex_mechanics --check    # verify the committed ones
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gridworld.baselines import plan_bfs_path  # noqa: E402
from gridworld.task_spec import TaskSpecification  # noqa: E402
from gridworld.task_validator import TaskValidator  # noqa: E402
from gridworld.tiling import direction_between, neighbor  # noqa: E402

_FIXTURES = _REPO_ROOT / "gridworld" / "fixtures"
SQUARE_MANIFEST = _FIXTURES / "manifest.mechanics_15.json"
HEX_MANIFEST = _FIXTURES / "manifest.mechanics_15_hex.json"
OUT_DIR = _REPO_ROOT / "mazes" / "mechanics_hex"

# One maze per mechanic, plus the one with all four (M10).
DEFAULT_TASKS = (
    "mech15_M7_18x18_dense_kr_sg_kb_portals_0",
    "mech15_M7_18x18_dense_kr_sg_kb_death_0",
    "mech15_M7_18x18_dense_kr_sg_kb_rotating_0",
    "mech15_M7_18x18_dense_kr_sg_kb_frozen_0",
    "mech15_M10_18x18_dense_kr_sg_kb_portals_death_rotating_frozen_0",
)
# R2's mechanics-15 BFS lengths run 49-161 egocentric steps.
BFS_RANGE = (49, 161)


def hex_rotating_direction(pos: tuple[int, int], square_direction: int) -> int:
    """The hex direction from ``pos`` to the cell a square arrow points at."""
    target = neighbor("square", pos, square_direction)
    direction = direction_between("hex", pos, target)
    assert direction is not None, "square neighbours are hex neighbours in odd-r"
    return direction


def convert_to_hex(payload: dict[str, Any], source: str) -> dict[str, Any]:
    """The hex maze for a square mechanics maze payload (not validated)."""
    out = copy.deepcopy(payload)
    out["task_id"] = f"hex_{payload['task_id']}"
    out["description"] = (
        f"Hex conversion of {payload['task_id']}: the same walls, red key-door, "
        "switch-gate, blue key-door chain and mechanic cells, read as pointy-top "
        "hexes (odd-r offset: cell (x, y) is column x of row y, odd rows half a "
        "cell east). Rotating arrows keep the cell they first point at."
    )
    out["maze"]["tiling"] = "hex"
    mech = out.get("mechanisms", {})
    if mech.get("rotating_tiles"):
        mech["rotating_initial_directions"] = [
            hex_rotating_direction(tuple(cell), int(direction))
            for cell, direction in zip(mech["rotating_tiles"], mech["rotating_initial_directions"])
        ]
    metadata = dict(out.get("metadata") or {})
    metadata["tiling"] = "hex"
    metadata["converted_from"] = source
    out["metadata"] = metadata
    return out


def _needed(spec: TaskSpecification) -> set[str]:
    """Keys and switches the maze cannot be solved without."""
    unnecessary = {
        line.split()[1] for line in TaskValidator(spec).validate_mechanism_necessity()
    }
    ids = {k.id for k in spec.mechanisms.keys} | {s.id for s in spec.mechanisms.switches}
    return ids - unnecessary


def check_conversion(square: TaskSpecification, hexed: TaskSpecification) -> tuple[dict[str, Any], list[str]]:
    """BFS lengths and on-path chain of both, plus every reason to refuse."""
    errors: list[str] = []
    ok, messages = hexed.validate()
    if not ok:
        errors.append(f"validate() failed: {messages}")
        return {}, errors
    square_bfs, hex_bfs = plan_bfs_path(square), plan_bfs_path(hexed)
    if not hex_bfs.success:
        errors.append("the planner cannot solve the hex maze")
        return {}, errors
    steps = len(hex_bfs.action_labels)
    if not BFS_RANGE[0] <= steps <= BFS_RANGE[1]:
        errors.append(f"hex BFS {steps} steps is outside R2's range {BFS_RANGE}")
    freed = _needed(square) - _needed(hexed)
    if freed:
        errors.append(f"hex adjacency bypasses {sorted(freed)}, which the square maze needs")
    report = {
        "square_optimal_steps": len(square_bfs.action_labels),
        "optimal_steps": steps,
        "chain": [label for label in hex_bfs.action_labels if ":" in label],
        "square_chain": [label for label in square_bfs.action_labels if ":" in label],
    }
    return report, errors


def build(task_ids: tuple[str, ...] = DEFAULT_TASKS) -> tuple[dict[str, dict], dict[str, Any], list[str]]:
    """(hex payload per file name, hex manifest, errors) from the square manifest."""
    square_manifest = json.loads(SQUARE_MANIFEST.read_text(encoding="utf-8"))
    rows = {row["task_id"]: row for row in square_manifest["tasks"]}
    mazes: dict[str, dict] = {}
    hex_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for task_id in task_ids:
        row = rows[task_id]
        payload = json.loads((_REPO_ROOT / row["source"]).read_text(encoding="utf-8"))
        hexed = convert_to_hex(payload, row["source"])
        report, problems = check_conversion(
            TaskSpecification.from_dict(payload), TaskSpecification.from_dict(hexed)
        )
        errors.extend(f"{task_id}: {problem}" for problem in problems)
        if problems:
            continue
        name = f"{hexed['task_id']}.json"
        mazes[name] = hexed
        hex_rows.append({
            "task_id": task_id.replace("mech15_", "mech15hex_", 1),
            "experiment": "mechanics_15_hex",
            "condition": row["condition"],
            "variant": f"hex_{row['variant']}",
            "source": (OUT_DIR / name).relative_to(_REPO_ROOT).as_posix(),
            "square_task_id": task_id,
            "maze_family": row["maze_family"],
            "maze_category": row["maze_category"],
            "rank": len(hex_rows) + 1,
            "optimal_steps": report["optimal_steps"],
            "square_optimal_steps": report["square_optimal_steps"],
            "new_mechanisms": row["new_mechanisms"],
            "new_mechanism_count": row["new_mechanism_count"],
            "topology": row["topology"],
            "grid": row["grid"],
            "tiling": "hex",
        })
    manifest = {
        "description": (
            "Mechanics-15 on hex: five of the 15 ogbench mechanics mazes (one per mechanic, "
            "and M10 with all four) converted to pointy-top hexes by scripts/make_hex_mechanics.py, "
            "same walls, chain and mechanic cells. Run with a multigrid render block "
            "(run_config.mechanics_15_hex_2d.json); the minigrid and 3D backends refuse hex mazes."
        ),
        "selection": {
            "source_set": "manifest.mechanics_15.json (ogbench M7-M10)",
            "maze_count": len(hex_rows),
            "conversion": "odd-r offset: square cell (x, y) is hex column x of row y",
            "optimal_steps": "executable-action BFS on the hex maze (gridworld.baselines.plan_bfs_path), the runtime's 3x step-cap base",
            "square_optimal_steps": "the same on the square original",
            "checks": "planner-solvable, BFS within R2's 49-161 range, and no key or switch the square maze needs is bypassed",
        },
        "tasks": hex_rows,
    }
    return mazes, manifest, errors


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=1) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert mechanics-15 mazes to the hex tiling.")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS),
                        help="mechanics-15 task ids to convert (default: the five of the hex smoke run)")
    parser.add_argument("--check", action="store_true",
                        help="write nothing; fail if the committed mazes or manifest differ")
    args = parser.parse_args(argv)

    mazes, manifest, errors = build(tuple(args.tasks))
    if errors:
        print("Hex conversion FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1
    files = {OUT_DIR / name: _dump(payload) for name, payload in mazes.items()}
    files[HEX_MANIFEST] = _dump(manifest)
    for row in manifest["tasks"]:
        print(f"{row['task_id']}: BFS {row['optimal_steps']} (square {row['square_optimal_steps']})")
    if args.check:
        stale = [p for p, text in files.items() if not p.exists() or p.read_text(encoding="utf-8") != text]
        for path in stale:
            print(f"stale: {path.relative_to(_REPO_ROOT)}")
        return 1 if stale else 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, text in files.items():
        path.write_text(text, encoding="utf-8")
    print(f"wrote {len(mazes)} mazes to {OUT_DIR.relative_to(_REPO_ROOT)} and {HEX_MANIFEST.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
