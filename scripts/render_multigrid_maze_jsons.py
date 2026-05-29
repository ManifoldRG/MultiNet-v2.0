"""Render maze JSON files through the local MultiGrid backend."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from gridworld.backends.multigrid_backend import MultiGridBackend
from gridworld.task_spec import TaskSpecification


def find_maze_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*.json") if path.is_file())


def output_path_for(input_dir: Path, output_dir: Path, tiling: str, json_path: Path) -> Path:
    return output_dir / tiling / json_path.relative_to(input_dir).with_suffix(".png")


def render_maze(json_path: Path, output_path: Path, tiling: str) -> None:
    spec = TaskSpecification.from_json(str(json_path))
    backend = MultiGridBackend(tiling=tiling, render_mode="rgb_array")
    backend.configure(spec)
    backend.reset(seed=spec.seed)
    image = backend.render()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render all maze JSON files with gridworld.backends.MultiGridBackend."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("mazes/ogbench"),
        help="Directory containing maze JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mazes/multigrid_image"),
        help="Directory where tiling-specific PNG folders will be written.",
    )
    parser.add_argument(
        "--tilings",
        nargs="+",
        default=["square", "hex"],
        help="MultiGrid tilings to render.",
    )
    args = parser.parse_args()

    maze_files = find_maze_files(args.input_dir)
    if not maze_files:
        print(f"No JSON files found in: {args.input_dir}")
        return 1

    failures: list[tuple[str, Path, str]] = []
    for tiling in args.tilings:
        for json_path in maze_files:
            output_path = output_path_for(args.input_dir, args.output_dir, tiling, json_path)
            try:
                render_maze(json_path, output_path, tiling)
            except Exception as exc:  # noqa: BLE001
                failures.append((tiling, json_path, str(exc)))
                print(f"FAIL: {tiling}: {json_path.relative_to(args.input_dir)} ({exc})")
                continue
            print(f"OK: {tiling}: {json_path.relative_to(args.input_dir)} -> {output_path}")

    total = len(maze_files) * len(args.tilings)
    rendered = total - len(failures)
    print(f"Rendered {rendered}/{total} MultiGrid mazes to {args.output_dir}")
    if failures:
        print("Failed files:")
        for tiling, path, error in failures:
            print(f"  - {tiling}/{path.name}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
