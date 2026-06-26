"""Render OGBench maze JSONs through all local grid backends."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from scripts.render_maze_jsons import (
    find_maze_files,
    output_path_for as minigrid_output_path_for,
    render_maze as render_minigrid_maze,
)
from scripts.render_multigrid_maze_jsons import (
    output_path_for as multigrid_output_path_for,
    render_maze as render_multigrid_maze,
)


def _clean_path(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render OGBench maze JSONs recursively through MiniGrid and "
            "MultiGrid, preserving input subdirectories."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("mazes/ogbench"),
        help="Directory containing maze JSON files, possibly nested.",
    )
    parser.add_argument(
        "--minigrid-output-dir",
        type=Path,
        default=Path("mazes/maze_image"),
        help="Directory for MiniGrid PNGs.",
    )
    parser.add_argument(
        "--multigrid-output-dir",
        type=Path,
        default=Path("mazes/multigrid_image"),
        help="Directory for MultiGrid tiling-specific PNG folders.",
    )
    parser.add_argument(
        "--tilings",
        nargs="+",
        default=["square", "hex"],
        help="MultiGrid tilings to render.",
    )
    parser.add_argument(
        "--skip-minigrid",
        action="store_true",
        help="Do not render MiniGrid images.",
    )
    parser.add_argument(
        "--skip-multigrid",
        action="store_true",
        help="Do not render MultiGrid images.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove output directories before rendering.",
    )
    args = parser.parse_args()

    maze_files = find_maze_files(args.input_dir)
    if not maze_files:
        print(f"No JSON files found in: {args.input_dir}")
        return 1

    if args.clean:
        if not args.skip_minigrid:
            _clean_path(args.minigrid_output_dir)
        if not args.skip_multigrid:
            for tiling in args.tilings:
                _clean_path(args.multigrid_output_dir / tiling)

    failures: list[tuple[str, Path, str]] = []

    if not args.skip_minigrid:
        for json_path in maze_files:
            output_path = minigrid_output_path_for(
                args.input_dir, args.minigrid_output_dir, json_path
            )
            try:
                render_minigrid_maze(json_path, output_path)
            except Exception as exc:  # noqa: BLE001
                failures.append(("minigrid", json_path, str(exc)))
                print(f"FAIL: minigrid: {json_path.relative_to(args.input_dir)} ({exc})")
                continue
            print(f"OK: minigrid: {json_path.relative_to(args.input_dir)} -> {output_path}")

    if not args.skip_multigrid:
        for tiling in args.tilings:
            for json_path in maze_files:
                output_path = multigrid_output_path_for(
                    args.input_dir, args.multigrid_output_dir, tiling, json_path
                )
                try:
                    render_multigrid_maze(json_path, output_path, tiling)
                except Exception as exc:  # noqa: BLE001
                    failures.append((f"multigrid/{tiling}", json_path, str(exc)))
                    print(
                        f"FAIL: multigrid/{tiling}: "
                        f"{json_path.relative_to(args.input_dir)} ({exc})"
                    )
                    continue
                print(
                    f"OK: multigrid/{tiling}: "
                    f"{json_path.relative_to(args.input_dir)} -> {output_path}"
                )

    backends = 0
    if not args.skip_minigrid:
        backends += 1
    if not args.skip_multigrid:
        backends += len(args.tilings)
    total = len(maze_files) * backends
    rendered = total - len(failures)
    print(f"Rendered {rendered}/{total} images from {len(maze_files)} maze JSON files")

    if failures:
        print("Failed files:")
        for backend, path, error in failures:
            print(f"  - {backend}/{path.relative_to(args.input_dir)}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
