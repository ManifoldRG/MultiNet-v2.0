"""
For each ``nlu_benchmark/benchmark_mazes/**/*.json``:

0. Apply :func:`~nlu_benchmark.loader.task_dict_shrink_dimensions_minus_two` (labeled 10×10 → 8×8 grid; coordinates unchanged).
1. Run mazegen :func:`~automatic_maze_generation.mazegen.solver.solve_maze` (no ``validate_maze``).
2. If solvable, write **one** PNG (same style as ``automatic_maze_generation/render_dataset.py``):
   maze + mechanisms + overlaid optimal path, under
   ``smoke_tests/results/benchmark_png/<category>/<stem>.png``.

Run from repo root::

    PYTHONPATH=src/v2 python3 src/v2/nlu_pipeline/nlu_benchmark/smoke_tests/smoke_benchmark_mazes.py
"""

from __future__ import annotations

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
_PNG_ROOT = _SCRIPT_DIR / "results" / "benchmark_png"


def main() -> None:
    base = ROOT / "nlu_benchmark" / "benchmark_mazes"
    if not base.is_dir():
        print(f"error: {base} not found (add benchmark JSONs there)", file=sys.stderr)
        sys.exit(2)

    paths = sorted(base.glob("**/*.json"))
    if not paths:
        print(f"warning: no JSON files under {base}", file=sys.stderr)
        sys.exit(0)

    failed = 0
    failures: list[tuple[Path, str]] = []
    for path in paths:
        rel = path.relative_to(base)
        try:
            text = path.read_text(encoding="utf-8")
            data = task_dict_shrink_dimensions_minus_two(json.loads(text))
            result = solve_maze(maze_instance_from_task_dict(data))
            if not result.get("is_solvable"):
                failed += 1
                msg = "not solvable"
                failures.append((rel, msg))
                print(f"FAIL {rel}: {msg}", flush=True)
                continue
            out_png = _PNG_ROOT / rel.parent / f"{path.stem}.png"
            out_png.parent.mkdir(parents=True, exist_ok=True)
            render_task_json_with_solver_path_png(data, result.get("path", []), out_png)
            print(f"ok {rel} cost={result.get('optimal_cost')} png={out_png}")
        except Exception as e:
            failed += 1
            msg = str(e)
            failures.append((rel, msg))
            print(f"FAIL {rel}: {msg}", file=sys.stderr, flush=True)

    print(f"\n{len(paths)} files, {failed} failed")
    if failures:
        print("Failed files:", file=sys.stderr)
        for rel, msg in failures:
            print(f"  - {rel}: {msg}", file=sys.stderr)
    print(f"PNGs under {_PNG_ROOT}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
