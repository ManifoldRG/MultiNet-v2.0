"""Run the interface with a fake agent and write an exhaustive episode log."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interface.config import ExperimentConfig
from interface.episode_log import flush_episode_log
from interface.loader import default_maze_path, load_task
from interface.runner import build_runner
from interface.smoke_tests.plans import v01_empty_room_trajectory, v04_single_key_trajectory
from interface.smoke_tests.smoke_prompting_observation_querying import ProbeAgent


def _replay_actions_for_maze(maze_path: Path) -> list[str]:
    if maze_path.stem == "V01_empty_room":
        return v01_empty_room_trajectory()
    if maze_path.stem == "V04_single_key":
        return v04_single_key_trajectory()
    raise SystemExit(
        f"No hardcoded replay plan for {maze_path.name}. Add one to interface/smoke_tests/plans.py"
    )


class ReplayProbeAgent(ProbeAgent):
    """ProbeAgent that replays a fixed action list (step_by_step) before falling back."""

    def __init__(
        self,
        full_trajectory_actions: list[str],
        *,
        replay_actions: list[str] | None = None,
    ) -> None:
        super().__init__(full_trajectory_actions)
        self._replay_queue = list(replay_actions or [])

    def __call__(self, messages: list[dict]) -> str:
        if self._replay_queue:
            action = self._replay_queue.pop(0)
            return f"FINAL_OUTPUT: {action}"
        return super().__call__(messages)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay interface episode and write exhaustive logs (prompts, state, frames)."
    )
    parser.add_argument("--maze", default="V04_single_key.json")
    parser.add_argument("--observation", default="image_text", choices=["text_only", "image_text", "image_only"])
    parser.add_argument("--prompting", default="standard", choices=["minimal", "standard", "verbose"])
    parser.add_argument("--context-window", default="last3", choices=["current", "last3"])
    parser.add_argument(
        "--querying",
        default="step_by_step",
        choices=["step_by_step", "subgoal", "full_trajectory"],
    )
    parser.add_argument("--tag", default="", help="Optional suffix on output directory name.")
    args = parser.parse_args()

    maze_path = default_maze_path(args.maze)
    backend, spec = load_task(maze_path)
    config = ExperimentConfig(
        prompting=args.prompting,
        observation=args.observation,
        context_window=args.context_window,
        querying=args.querying,
        chat_history="stateless",
    )
    runner = build_runner(config, backend, spec)

    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = Path(__file__).resolve().parent / "results" / f"replay_{maze_path.stem}{suffix}"

    replay_actions = _replay_actions_for_maze(maze_path)
    agent = ReplayProbeAgent(replay_actions, replay_actions=replay_actions)

    result = runner.run(agent, verbose=True, maze_path=maze_path)
    episode_path = flush_episode_log(result, out_dir)

    queries = sum(1 for r in result["transcript"] if r.get("kind") == "query")
    steps = sum(1 for r in result["transcript"] if r.get("kind") == "step")
    print(f"success={result['success']} end_reason={result['end_reason']} steps={result['steps_used']}")
    print(f"queries={queries} logged_steps={steps}")
    print(f"episode={episode_path}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
