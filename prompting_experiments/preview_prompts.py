"""Generate a text preview of prompt experiment condition variants."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from prompting_experiments import CONDITION_SETS
from prompting_experiments.prompt_templates import feedback as feedback_templates

_ONE_SHOT_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "mazes" / "one_shot_example"
_ONE_SHOT_MAZE_PATH = (
    _ONE_SHOT_EXAMPLE_DIR / "one_shot_example_14x14_dense_kr_sg_kb_2.json"
)
_ONE_SHOT_SOLUTION_PATH = _ONE_SHOT_EXAMPLE_DIR / "one_shot_example_solution.json"


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
    *,
    move_only: bool = False,
) -> tuple[Any, str, list[dict]]:
    from interface.actions_map import nlu_action_to_int
    from interface.coords import agent_facing, agent_row_col
    from interface.episode_log import state_snapshot
    from interface.feedback import format_step_feedback
    from interface.parser import ACTION_ORDER

    rng = random.Random(seed)
    _nav_actions = {"TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD"}
    actions = [
        action for action in ACTION_ORDER
        if action != "DONE" and (not move_only or action in _nav_actions)
    ]
    last_feedback = feedback_templates.INITIAL_FEEDBACK
    transcript: list[dict] = []

    for step_index in range(1, steps + 1):
        action = rng.choice(actions)
        position_before = agent_row_col(state)
        facing_before = agent_facing(state)
        state_before = state_snapshot(state)
        decision_frame_rgb = runner.last_rgb
        prev_state = state

        # Handle DONE token (no environment step) and map NLU action token to backend-specific integer.
        if action == 'DONE':
            break
        try:
            from gridworld.backends.multigrid_backend import MultiGridBackend
            from multigrid.agent import Action as MGAction
        except Exception:
            MultiGridBackend = None
            MGAction = None

        if MultiGridBackend is not None and isinstance(runner.backend, MultiGridBackend):
            act = MGAction.FORWARD if action == 'MOVE_FORWARD' else getattr(MGAction, action)
            # Call the underlying MultiGrid environment directly with the
            # MultiGrid action index. `MultiGridBackend.step` expects a
            # MiniGrid-style action index and would remap the value, so
            # use `env.step` to execute the native MultiGrid action.
            runner.last_rgb, reward, terminated, truncated, info = runner.backend.env.step(int(act))
            # Maintain backend step bookkeeping and rebuild GridState so
            # GridState fields (including goal_reached) are correct.
            runner.backend._step_count += 1
            state = runner.backend._build_grid_state()
            state.terminated = terminated
            state.truncated = truncated
            state.reward = reward
            state.step_count = runner.backend._step_count
        else:
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


def _solution_preview_steps(runner, state, actions: list[str]) -> tuple[Any, str, list[dict]]:
    from interface.actions_map import nlu_action_to_int
    from interface.coords import agent_facing, agent_row_col
    from interface.episode_log import state_snapshot
    from interface.feedback import format_step_feedback

    last_feedback = feedback_templates.INITIAL_FEEDBACK
    transcript: list[dict] = []

    for step_index, action in enumerate(actions, start=1):
        position_before = agent_row_col(state)
        facing_before = agent_facing(state)
        state_before = state_snapshot(state)
        decision_frame_rgb = runner.last_rgb
        prev_state = state

        # Handle DONE token (no environment step) and map NLU action token to backend-specific integer.
        if action == 'DONE':
            break
        try:
            from gridworld.backends.multigrid_backend import MultiGridBackend
            from multigrid.agent import Action as MGAction
        except Exception:
            MultiGridBackend = None
            MGAction = None

        if MultiGridBackend is not None and isinstance(runner.backend, MultiGridBackend):
            act = MGAction.FORWARD if action == 'MOVE_FORWARD' else getattr(MGAction, action)
            # Use the native MultiGrid env.step so the action index is treated
            # as a MultiGrid action. `MultiGridBackend.step` would remap the
            # integer assuming a MiniGrid action space, which is not desired
            # when we already have a native MultiGrid `Action` value.
            runner.last_rgb, reward, terminated, truncated, info = runner.backend.env.step(int(act))
            runner.backend._step_count += 1
            state = runner.backend._build_grid_state()
            state.terminated = terminated
            state.truncated = truncated
            state.reward = reward
            state.step_count = runner.backend._step_count
        else:
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
                "action_queue_index": step_index - 1,
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
                "actions_remaining_after": actions[step_index:],
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
    *,
    move_only: bool = False,
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
        move_only=move_only,
    )
    system_prompt, user_message = runner.build_prompt_message(
        state,
        last_feedback,
        transcript,
    )
    return system_prompt, _content_to_text(user_message.get("content"))


def _one_shot_text_summary_preview(config) -> tuple[int, str, str]:
    try:
        from interface.loader import load_task
        from interface.runner import build_runner
    except ModuleNotFoundError as exc:
        raise SystemExit(_missing_dependency_message(exc)) from exc

    solution = json.loads(_ONE_SHOT_SOLUTION_PATH.read_text(encoding="utf-8"))
    actions = solution["actions"]
    # Use the MultiGrid backend for the one-shot example so the replay
    # executes the full mechanism semantics (keys, doors, switches, gates).
    from gridworld.task_spec import TaskSpecification
    from gridworld.backends.multigrid_backend import MultiGridBackend

    spec = TaskSpecification.from_json(str(_ONE_SHOT_MAZE_PATH))
    spec.max_steps = len(actions)
    backend = MultiGridBackend(tiling='square', render_mode='state_dict')
    backend.configure(spec)
    runner = build_runner(config, backend, spec)
    runner.last_rgb, state, _reset_info = backend.reset(seed=spec.seed)
    state, last_feedback, transcript = _solution_preview_steps(runner, state, actions)
    system_prompt, user_message = runner.build_prompt_message(
        state,
        last_feedback,
        transcript,
    )
    # Ensure the replay actually reached the goal in the live environment
    # Prefer calling `check_goal()` if the state is a native MultiGrid state,
    # otherwise fall back to the backend-agnostic GridState `goal_reached`.
    try:
        check = getattr(state, "check_goal", None)
        if callable(check):
            reached = state.check_goal()
        else:
            reached = getattr(state, "goal_reached", False)
    except Exception:
        reached = False
    if not reached:
        raise RuntimeError("Replayed one-shot solution did not reach the goal when executed in the environment")
    return len(transcript), system_prompt, _content_to_text(user_message.get("content"))


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
            steps_for_variant = variant.preview_steps if variant.preview_steps is not None else preview_steps
            seed_for_variant = variant.preview_rollout_seed if variant.preview_rollout_seed is not None else rollout_seed
            system_prompt, user_prompt = _prompt_preview(
                config,
                maze_path,
                max_steps,
                steps_for_variant,
                seed_for_variant,
                move_only=variant.preview_move_only,
            )
            chunks.extend(
                [
                    f"preview steps: {steps_for_variant}  rollout seed: {seed_for_variant}",
                    "[system prompt]",
                    system_prompt,
                    "",
                    "[user prompt]",
                    user_prompt,
                    "-" * 88,
                ]
            )
            if condition.name == "Context window" and variant_name in (
                "text_summary",
                "text_summary_and_last3",
            ):
                solution_steps, system_prompt, user_prompt = _one_shot_text_summary_preview(
                    config
                )
                chunks.extend(
                    [
                        f"additional {variant_name} example: one_shot_example maze after one_shot_example_solution",
                        f"maze: {_ONE_SHOT_MAZE_PATH}",
                        f"solution steps replayed: {solution_steps}",
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
