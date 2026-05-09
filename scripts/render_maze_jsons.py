"""Render maze JSON files through the local MiniGrid backend."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.task_spec import TaskSpecification


def find_maze_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*.json") if path.is_file())


def output_path_for(input_dir: Path, output_dir: Path, json_path: Path) -> Path:
    return output_dir / json_path.relative_to(input_dir).with_suffix(".png")


def render_maze(json_path: Path, output_path: Path) -> None:
    spec = TaskSpecification.from_json(str(json_path))
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    image, _, _ = backend.reset(seed=spec.seed)

    if backend.env is not None and backend.env.highlight:
        raise RuntimeError(f"MiniGrid highlight overlay is enabled for {json_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render all maze JSON files with gridworld.backends.MiniGridBackend."
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
        default=Path("mazes/maze_image"),
        help="Directory where PNG images will be written.",
    )
    args = parser.parse_args()

    maze_files = find_maze_files(args.input_dir)
    if not maze_files:
        print(f"No JSON files found in: {args.input_dir}")
        return 1

    failures: list[tuple[Path, str]] = []
    for json_path in maze_files:
        output_path = output_path_for(args.input_dir, args.output_dir, json_path)
        try:
            render_maze(json_path, output_path)
        except Exception as exc:  # noqa: BLE001
            failures.append((json_path, str(exc)))
            print(f"FAIL: {json_path.relative_to(args.input_dir)} ({exc})")
            continue
        print(f"OK: {json_path.relative_to(args.input_dir)} -> {output_path}")

    rendered = len(maze_files) - len(failures)
    print(f"Rendered {rendered}/{len(maze_files)} mazes to {args.output_dir}")
    if failures:
        print("Failed files:")
        for path, error in failures:
            print(f"  - {path.name}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
