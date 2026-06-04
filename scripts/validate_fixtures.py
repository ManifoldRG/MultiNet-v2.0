"""Validate pipeline fixtures and derive test-2 route discriminators.

Read-only with respect to task files; with ``--write`` it caches the computed
``route_short_cells`` / ``route_long_cells`` back into the manifest so the
runtime ``path_choice`` metric has unambiguous per-route cell sets.

Checks:
  * every fixture passes ``TaskSpecification.validate()`` and BFS-solves;
  * test-2 rows: both the short route and the route forced by walling off
    ``route_block`` are solvable and visit distinct cells;
  * test-3 rows: members sharing a ``pair_id`` have identical maze topology
    (dimensions + walls) and equal BFS optimal step counts.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from gridworld.baselines import plan_bfs_path
from gridworld.task_spec import TaskSpecification

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(source: str, manifest_path: Path) -> Path:
    candidate = Path(source)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for base in (manifest_path.parent, _REPO_ROOT):
        resolved = (base / source).resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"Task source not found: {source}")


def _load_spec(path: Path) -> TaskSpecification:
    return TaskSpecification.from_json(str(path))


def _spec_with_extra_wall(spec: TaskSpecification, cell: list[int]) -> TaskSpecification:
    data = spec.to_dict()
    walls = [list(w) for w in data["maze"].get("walls", [])]
    if list(cell) not in walls:
        walls.append(list(cell))
    data["maze"]["walls"] = walls
    return TaskSpecification.from_dict(data)


def _validate_one(row: dict[str, Any], manifest_path: Path) -> list[str]:
    errors: list[str] = []
    source = _resolve(row["source"], manifest_path)
    spec = _load_spec(source)
    ok, messages = spec.validate()
    if not ok:
        errors.append(f"{row['task_id']}: validate() failed: {messages}")
        return errors
    bfs = plan_bfs_path(spec)
    if not bfs.success:
        errors.append(f"{row['task_id']}: BFS could not solve the task")
    return errors


def _derive_test2_routes(row: dict[str, Any], manifest_path: Path) -> list[str]:
    errors: list[str] = []
    source = _resolve(row["source"], manifest_path)
    spec = _load_spec(source)

    short = plan_bfs_path(spec)
    if not short.success:
        return [f"{row['task_id']}: short route unsolvable"]

    block = row.get("route_block")
    if block is None:
        interior = [p for p in short.positions[1:-1]]
        if not interior:
            return [f"{row['task_id']}: no interior cell to block; set route_block explicitly"]
        block = list(interior[len(interior) // 2])

    long = plan_bfs_path(_spec_with_extra_wall(spec, block))
    if not long.success:
        errors.append(
            f"{row['task_id']}: blocking {block} leaves no alternate route (pick a different route_block)"
        )
        return errors

    short_cells = {tuple(p) for p in short.positions}
    long_cells = {tuple(p) for p in long.positions}
    short_only = sorted(short_cells - long_cells)
    long_only = sorted(long_cells - short_cells)
    if not short_only or not long_only:
        errors.append(f"{row['task_id']}: routes do not diverge enough to discriminate path_choice")

    row["route_block"] = list(block)
    row["route_short_cells"] = [list(c) for c in short_only]
    row["route_long_cells"] = [list(c) for c in long_only]
    return errors


def _check_test3_pairs(rows: list[dict[str, Any]], manifest_path: Path) -> list[str]:
    errors: list[str] = []
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("experiment") == "test3" and row.get("pair_id"):
            pairs[row["pair_id"]].append(row)

    for pair_id, members in pairs.items():
        if len(members) < 2:
            errors.append(f"pair {pair_id}: needs >= 2 members, found {len(members)}")
            continue
        specs = [_load_spec(_resolve(m["source"], manifest_path)) for m in members]
        dims = {tuple(s.maze.dimensions) for s in specs}
        walls = {frozenset((w.x, w.y) for w in s.maze.walls) for s in specs}
        if len(dims) != 1:
            errors.append(f"pair {pair_id}: maze dimensions differ: {dims}")
        if len(walls) != 1:
            errors.append(f"pair {pair_id}: wall layouts differ across members")
        optimal = []
        for member, spec in zip(members, specs):
            bfs = plan_bfs_path(spec)
            if not bfs.success:
                errors.append(f"{member['task_id']}: BFS could not solve the task")
            else:
                optimal.append(len(bfs.action_labels))
        if len(set(optimal)) > 1:
            errors.append(
                f"pair {pair_id}: BFS optimal step counts differ {optimal} "
                "(test3 requires equal path length within a pair)"
            )
    return errors


def validate_manifest(manifest_path: Path) -> tuple[dict[str, Any], list[str]]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = data["tasks"] if isinstance(data, dict) else data

    errors: list[str] = []
    for row in rows:
        errors.extend(_validate_one(row, manifest_path))
        if row.get("experiment") == "test2":
            errors.extend(_derive_test2_routes(row, manifest_path))
    errors.extend(_check_test3_pairs(rows, manifest_path))
    return data, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate pipeline fixtures and derive test-2 routes.")
    parser.add_argument(
        "--manifest", default=str(_REPO_ROOT / "gridworld" / "fixtures" / "manifest.json")
    )
    parser.add_argument("--write", action="store_true", help="Persist derived route cells to the manifest.")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    data, errors = validate_manifest(manifest_path)

    if errors:
        print("Fixture validation FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1

    if args.write:
        manifest_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"Validated OK; route discriminators written to {manifest_path}")
    else:
        print("Validated OK (use --write to cache test-2 route discriminators).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
