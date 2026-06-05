"""Generate a text preview of prompt experiment condition variants."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from prompting_experiments import CONDITION_SETS
from prompting_experiments.prompt_templates import feedback as feedback_templates


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    lines: list[str] = []
    image_count = 0
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            lines.append(block.get("text", ""))
        elif block.get("type") == "image_url":
            image_count += 1
            lines.append(f"[image block {image_count}]")
    return "\n".join(part for part in lines if part)


def _missing_dependency_message(exc: ModuleNotFoundError) -> str:
    return (
        f"Missing dependency: {exc.name}. Install the project dependencies in this environment, "
        "for example: python3 -m pip install -e '.[dev]'"
    )


def _prompt_preview(config, maze_path: Path, max_steps: int) -> tuple[str, str]:
    try:
        from interface.loader import load_task
        from interface.runner import build_runner
    except ModuleNotFoundError as exc:
        raise SystemExit(_missing_dependency_message(exc)) from exc

    backend, spec = load_task(maze_path)
    spec.max_steps = max_steps
    runner = build_runner(config, backend, spec)
    runner.last_rgb, state, _reset_info = backend.reset(seed=spec.seed)

    system_prompt = runner.prompt.build_system_prompt()

    user_message = runner._build_message(state, feedback_templates.INITIAL_FEEDBACK, [])
    return system_prompt, _content_to_text(user_message.get("content"))


def build_preview(maze_path: Path, max_steps: int) -> str:
    chunks = [
        "Prompt Experiment Preview",
        f"Maze: {maze_path}",
        f"Max steps: {max_steps}",
        "",
    ]

    for idx, condition in enumerate(CONDITION_SETS.values(), start=1):
        chunks.extend(
            [
                "=" * 88,
                f"condition set {idx}: {condition.name}",
                "=" * 88,
            ]
        )
        for variant_name, variant in condition.variants.items():
            chunks.extend(
                [
                    f"variant name: {variant_name}",
                    f"description: {variant.description}",
                    "prompts:",
                ]
            )
            if not variant.implemented:
                chunks.extend(
                    [
                        "Status: not implemented in ExperimentConfig",
                        "-" * 88,
                    ]
                )
                continue

            try:
                config = variant.build_config()
            except ModuleNotFoundError as exc:
                raise SystemExit(_missing_dependency_message(exc)) from exc
            system_prompt, user_prompt = _prompt_preview(config, maze_path, max_steps)
            chunks.extend(
                [
                    "[system prompt]",
                    system_prompt,
                    "",
                    "[user prompt]",
                    user_prompt,
                    "-" * 88,
                ]
            )

    return "\n".join(chunks).rstrip() + "\n"


def _default_maze_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "mazes" / "validation_10" / name


def main() -> None:
    parser = argparse.ArgumentParser(description="Write prompt experiment previews to prompts.txt.")
    parser.add_argument("--maze", default="V01_empty_room.json")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "prompts.txt",
    )
    args = parser.parse_args()

    maze_path = Path(args.maze)
    if not maze_path.is_file():
        maze_path = _default_maze_path(args.maze)

    preview = build_preview(maze_path, args.max_steps)
    args.output.write_text(preview, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
