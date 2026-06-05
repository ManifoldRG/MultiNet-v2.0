"""Generate a text preview of prompt experiment condition variants."""

from __future__ import annotations

import argparse
import random
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


def _rollout_preview_steps(
    runner,
    state,
    steps: int,
    seed: int,
) -> tuple[Any, str, list[dict]]:
    from interface.actions_map import nlu_action_to_int
    from interface.coords import agent_facing, agent_row_col
    from interface.episode_log import state_snapshot
    from interface.feedback import format_step_feedback
    from interface.parser import ACTION_ORDER

    rng = random.Random(seed)
    actions = [action for action in ACTION_ORDER if action != "DONE"]
    last_feedback = feedback_templates.INITIAL_FEEDBACK
    transcript: list[dict] = []

    for step_index in range(1, steps + 1):
        action = rng.choice(actions)
        position_before = agent_row_col(state)
        facing_before = agent_facing(state)
        state_before = state_snapshot(state)
        decision_frame_rgb = runner.last_rgb
        prev_state = state

        runner.last_rgb, reward, terminated, truncated, state, info = runner.backend.step(
            nlu_action_to_int(action)
        )
        step_detail, event_type = format_step_feedback(
            action, prev_state, state, reward, terminated, runner.task_spec
        )
        last_feedback = step_detail
        transcript.append(
            {
                "kind": "step",
                "step_index": step_index,
                "query_index": 0,
                "action_queue_index": 0,
                "env_step_count": state.step_count,
                "action": action,
                "event_type": event_type,
                "feedback": step_detail,
                "prompt_feedback": last_feedback,
                "facing_before": facing_before,
                "facing_after": agent_facing(state),
                "position_before": list(position_before),
                "position_after": list(agent_row_col(state)),
                "state_before": state_before,
                "state_after": state_snapshot(state),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "backend_info": info,
                "actions_remaining_after": [],
                "consecutive_failures_after": 0,
                "_decision_frame_rgb": decision_frame_rgb,
                "_post_step_rgb": runner.last_rgb,
            }
        )
        if terminated or truncated:
            break

    return state, last_feedback, transcript


def _prompt_preview(
    config,
    maze_path: Path,
    max_steps: int,
    preview_steps: int,
    rollout_seed: int,
) -> tuple[str, str]:
    try:
        from interface.loader import load_task
        from interface.runner import build_runner
    except ModuleNotFoundError as exc:
        raise SystemExit(_missing_dependency_message(exc)) from exc

    backend, spec = load_task(maze_path)
    spec.max_steps = max_steps
    runner = build_runner(config, backend, spec)
    runner.last_rgb, state, _reset_info = backend.reset(seed=spec.seed)
    state, last_feedback, transcript = _rollout_preview_steps(
        runner,
        state,
        preview_steps,
        rollout_seed,
    )
    system_prompt, user_message = runner.build_prompt_message(
        state,
        last_feedback,
        transcript,
    )
    return system_prompt, _content_to_text(user_message.get("content"))


def build_preview(
    maze_path: Path,
    max_steps: int,
    preview_steps: int,
    rollout_seed: int,
) -> str:
    chunks = [
        "Prompt Experiment Preview",
        f"Maze: {maze_path}",
        f"Max steps: {max_steps}",
        f"Preview prompt state: after {preview_steps} random steps (seed: {rollout_seed})",
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
            system_prompt, user_prompt = _prompt_preview(
                config,
                maze_path,
                max_steps,
                preview_steps,
                rollout_seed,
            )
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
    parser.add_argument("--preview-steps", type=int, default=3)
    parser.add_argument("--rollout-seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "prompts.txt",
    )
    args = parser.parse_args()

    maze_path = Path(args.maze)
    if not maze_path.is_file():
        maze_path = _default_maze_path(args.maze)

    preview = build_preview(
        maze_path,
        args.max_steps,
        args.preview_steps,
        args.rollout_seed,
    )
    args.output.write_text(preview, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
