from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
V2_ROOT = Path(__file__).resolve().parents[3]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from nlu_benchmark.smoke_tests.solver_plan_trace import write_png_trace_for_maze_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test: mazegen solver plan replayed in NLU env (PNG trace under results/smoke_*_bfs/).")
    parser.add_argument("--maze", default="V04_single_key.json", help="Maze JSON filename under sample mazes/")
    parser.add_argument("--tag", default="", help="Optional output tag suffix.")
    args = parser.parse_args()

    maze_path = ROOT / "nlu_benchmark" / "sample mazes" / args.maze
    maze_stem = Path(args.maze).stem
    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = Path(__file__).resolve().parent / "results" / f"smoke_{maze_stem}_bfs{suffix}"

    r = write_png_trace_for_maze_json(maze_path, out_dir)
    if not r["ok"]:
        print("Solver reported unsolvable maze.")
        return
    print(f"\nsuccess={r['success']}")
    print(f"steps_used={r['steps_used']}")
    print(f"out={r['out_dir']}")


if __name__ == "__main__":
    main()
