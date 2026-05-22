"""
For each ``nlu_benchmark/benchmark_mazes/**/*.json``:

0. Apply :func:`~nlu_benchmark.loader.task_dict_shrink_dimensions_minus_two` (labeled 10×10 → 8×8 grid; coordinates unchanged).
1. Run mazegen :func:`~automatic_maze_generation.mazegen.solver.solve_maze` (no ``validate_maze``).
2. Write **one** PNG (same style as ``automatic_maze_generation/render_dataset.py``) under
   ``smoke_tests/results/benchmark_solver/<category>/<stem>.png``:
   if solvable, maze + mechanisms + overlaid optimal path; if not solvable, maze + mechanisms only (no path).
3. Write ``smoke_tests/results/benchmark_solver/benchmark_mazes_metadata.csv`` with columns:
   ``rel_path``, ``chain_pattern``, ``is_solvable``, ``optimal_cost``, ``optimal_path``, ``n_interactions``, ``error``.
   ``optimal_path`` is a JSON list of ``[x, y]`` cells in mazegen 0-based coordinates (column, row),
   only filled when ``is_solvable``; otherwise empty.

Run from repo root::

    PYTHONPATH=src/v2 python3 src/v2/nlu_pipeline/nlu_benchmark/smoke_tests/smoke_benchmark_mazes.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from automatic_maze_generation.mazegen.solver import solve_maze  # noqa: E402
from nlu_benchmark.loader import maze_instance_from_task_dict, task_dict_shrink_dimensions_minus_two  # noqa: E402
from nlu_benchmark.renderer import render_task_json_with_solver_path_png  # noqa: E402

_SCRIPT_DIR = Path(__file__).resolve().parent
_RESULTS_ROOT = _SCRIPT_DIR / "results"
_BENCHMARK_SOLVER_DIR = _RESULTS_ROOT / "benchmark_solver"
_CSV_PATH = _BENCHMARK_SOLVER_DIR / "benchmark_mazes_metadata.csv"

_CSV_FIELDNAMES = [
    "rel_path",
    "chain_pattern",
    "is_solvable",
    "optimal_cost",
    "optimal_path",
    "n_interactions",
    "error",
]


def _fill_solver_columns(row: dict[str, object], result: dict) -> None:
    solvable = bool(result.get("is_solvable"))
    inter = result.get("interactions") or []
    cost = result.get("optimal_cost")
    row["is_solvable"] = solvable
    row["optimal_cost"] = "" if cost is None else cost
    row["n_interactions"] = len(inter)
    if solvable:
        pts = result.get("path") or []
        as_lists = [[int(x), int(y)] for x, y in pts]
        row["optimal_path"] = json.dumps(as_lists, ensure_ascii=False)
    else:
        row["optimal_path"] = ""


def main() -> None:
    base = ROOT / "nlu_benchmark" / "benchmark_mazes"
    if not base.is_dir():
        print(f"error: {base} not found (add benchmark JSONs there)", file=sys.stderr)
        sys.exit(2)

    paths = sorted(base.glob("**/*.json"))
    if not paths:
        print(f"warning: no JSON files under {base}", file=sys.stderr)
        sys.exit(0)

    _RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    _BENCHMARK_SOLVER_DIR.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict[str, object]] = []

    failed = 0
    failures: list[tuple[Path, str]] = []
    for path in paths:
        rel = path.relative_to(base)
        row: dict[str, object] = {
            "rel_path": str(rel),
            "chain_pattern": "",
            "is_solvable": "",
            "optimal_cost": "",
            "optimal_path": "",
            "n_interactions": "",
            "error": "",
        }

        try:
            text = path.read_text(encoding="utf-8")
            raw = json.loads(text)
            data = task_dict_shrink_dimensions_minus_two(raw)
            row["chain_pattern"] = (raw.get("metadata") or {}).get("chain_pattern", "")
        except Exception as e:
            failed += 1
            msg = str(e)
            row["error"] = msg
            failures.append((rel, msg))
            print(f"FAIL {rel}: {msg}", file=sys.stderr, flush=True)
            csv_rows.append(row)
            continue

        try:
            result = solve_maze(maze_instance_from_task_dict(data))
            _fill_solver_columns(row, result)
            solvable = bool(result.get("is_solvable"))
            path_pts = (result.get("path") or []) if solvable else []
            out_png = _BENCHMARK_SOLVER_DIR / rel.parent / f"{path.stem}.png"
            out_png.parent.mkdir(parents=True, exist_ok=True)
            render_task_json_with_solver_path_png(data, path_pts, out_png)
            if solvable:
                print(f"ok {rel} cost={result.get('optimal_cost')} png={out_png}")
            else:
                print(f"ok {rel} not solvable png={out_png} (no path)")
        except Exception as e:
            failed += 1
            msg = str(e)
            row["error"] = msg
            failures.append((rel, msg))
            print(f"FAIL {rel}: {msg}", file=sys.stderr, flush=True)
        csv_rows.append(row)

    with _CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(csv_rows)

    print(f"\n{len(paths)} files, {failed} failed")
    if failures:
        print("Failed files:", file=sys.stderr)
        for rel, msg in failures:
            print(f"  - {rel}: {msg}", file=sys.stderr)
    print(f"Outputs under {_BENCHMARK_SOLVER_DIR}")
    print(f"CSV {_CSV_PATH}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
