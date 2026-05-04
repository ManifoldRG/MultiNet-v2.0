from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
V2_ROOT = Path(__file__).resolve().parents[3]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from nlu_benchmark.agents import ClaudeAnthropicAgent, ClaudeAnthropicConfig
from nlu_benchmark.loader import load_maze
from nlu_benchmark.runner import ExperimentRunner
from nlu_benchmark.smoke_trace import trace_prepare, trace_reset, trace_step, trace_write_text_artifacts


def _ensure_anthropic_api_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    for directory in Path(__file__).resolve().parents:
        key_file = directory / "api_key.txt"
        if key_file.is_file():
            os.environ["ANTHROPIC_API_KEY"] = key_file.read_text().strip()
            return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test: Claude agent episode in NLU env (PNG trace under results/smoke_*_claude/).",
    )
    parser.add_argument("--maze", default="V04_single_key.json", help="Maze JSON filename under sample mazes/")
    parser.add_argument("--tag", default="", help="Optional output tag suffix.")
    args = parser.parse_args()

    maze_path = ROOT / "nlu_benchmark" / "sample mazes" / args.maze
    maze_stem = Path(args.maze).stem
    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = Path(__file__).resolve().parent / "results" / f"smoke_{maze_stem}_claude{suffix}"

    if not maze_path.is_file():
        print(f"Missing maze file: {maze_path}")
        return

    trace_prepare(out_dir)

    _ensure_anthropic_api_key()

    runner = ExperimentRunner.from_json(str(maze_path.resolve()))
    agent = ClaudeAnthropicAgent(config=ClaudeAnthropicConfig())

    try:
        result = runner.run(agent, verbose=False)
    except Exception as e:
        print(f"runner.run raised: {e}")
        return

    transcript = result["transcript"]
    planned_actions = [rec["action"] for rec in transcript]

    env = load_maze(maze_path)
    state = env.reset()

    lines = trace_reset(out_dir, state)

    for step, action in enumerate(planned_actions, start=1):
        before = state.agent_pos
        state, event = trace_step(out_dir, lines, step, action, env, position_before=before)
        if event.type == "DONE":
            break

    trace_write_text_artifacts(out_dir, lines, planned_actions)

    print(f"\nsuccess={state.agent_pos == state.goal}")
    print(f"steps_used={state.step_count}")
    print(f"out={out_dir}")


if __name__ == "__main__":
    main()
