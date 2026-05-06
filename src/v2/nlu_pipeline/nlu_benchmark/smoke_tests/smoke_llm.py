from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
V2_ROOT = Path(__file__).resolve().parents[3]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from nlu_benchmark.agents import (
    ClaudeAnthropicAgent,
    ClaudeAnthropicConfig,
    DEFAULT_LOCAL_MODEL,
    LocalLLMConfig,
    LocalTransformersAgent,
)
from nlu_benchmark.config import ExperimentConfig
from nlu_benchmark.loader import load_maze_from_dict, task_dict_shrink_dimensions_minus_two
from nlu_benchmark.runner import ExperimentRunner, build_runner
from nlu_benchmark.smoke_trace import trace_prepare, trace_reset, trace_step, trace_write_text_artifacts


def _configure_benchmark_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.WARNING)
    if level == logging.WARNING:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _ensure_anthropic_api_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    for directory in Path(__file__).resolve().parents:
        key_file = directory / "api_key.txt"
        if key_file.is_file():
            os.environ["ANTHROPIC_API_KEY"] = key_file.read_text().strip()
            return


def _persist_llm_queries_jsonl(out_dir: Path, records: List[dict]) -> None:
    if not records:
        return
    path = out_dir / "llm_queries.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_episode_json(
    out_dir: Path,
    result: Dict[str, object],
    *,
    backend: str,
    model: str,
) -> None:
    payload = {
        "success": result["success"],
        "steps_used": result["steps_used"],
        "config": result["config"],
        "transcript": result["transcript"],
        "smoke": {"backend": backend, "model": model},
    }
    (out_dir / "episode.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


class _AgentRecorder:
    """Delegates to a real agent and records each assistant reply for ``llm_queries.jsonl``."""

    __slots__ = ("_inner", "_records", "_query_seq")

    def __init__(self, inner: Callable[[List[dict]], str], records: List[dict]) -> None:
        self._inner = inner
        self._records = records
        self._query_seq = 0

    def __call__(self, messages: List[dict]) -> str:
        self._query_seq += 1
        text = self._inner(messages)
        self._records.append(
            {
                "query": self._query_seq,
                "messages_in_context": len(messages),
                "reply": text,
            }
        )
        return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test: LLM-guided maze episode (PNG trace under results/). "
            "Writes llm_queries.jsonl (model replies) and episode.json (transcript + config). "
            "Anthropic runs in the cloud; --backend local uses Hugging Face. "
            "--log-level INFO for query timing; -v for per-step prints."
        ),
    )
    parser.add_argument("--maze", default="V04_single_key.json", help="Maze JSON filename under sample mazes/")
    parser.add_argument("--tag", default="", help="Optional output tag suffix.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-step progress from ExperimentRunner.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Structured logs from nlu_benchmark (timestamps, LLM query stats; default: no extra logs).",
    )
    parser.add_argument(
        "--backend",
        choices=("anthropic", "local"),
        default="anthropic",
        help="anthropic: Claude API (remote). local: Hugging Face Transformers on this machine.",
    )
    parser.add_argument(
        "--hf-model",
        default=DEFAULT_LOCAL_MODEL,
        help="With --backend local: Hugging Face model id (default: %(default)s).",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='With --backend local: passed to from_pretrained device_map (e.g. "auto", "cuda:0").',
    )
    parser.add_argument(
        "--chat-history",
        choices=("stateless", "rolling", "full"),
        default=None,
        help="ExperimentConfig.chat_history (default: rolling = last N turns in API).",
    )
    parser.add_argument(
        "--chat-turns-max",
        type=int,
        default=None,
        metavar="N",
        help="ExperimentConfig.chat_turns_max for rolling mode (default: 3).",
    )
    args = parser.parse_args()
    _configure_benchmark_logging(args.log_level)

    maze_path = ROOT / "nlu_benchmark" / "sample mazes" / args.maze
    maze_stem = Path(args.maze).stem
    suffix = f"_{args.tag}" if args.tag else ""
    out_slug = "claude" if args.backend == "anthropic" else "local"
    out_dir = Path(__file__).resolve().parent / "results" / f"smoke_{maze_stem}_{out_slug}{suffix}"

    if not maze_path.is_file():
        print(f"Missing maze file: {maze_path}")
        return

    trace_prepare(out_dir)

    task_data = task_dict_shrink_dimensions_minus_two(
        json.loads(maze_path.read_text(encoding="utf-8"))
    )
    maze_env = load_maze_from_dict(task_data)

    exp_cfg = ExperimentConfig()
    if args.chat_history is not None:
        exp_cfg.chat_history = args.chat_history
    if args.chat_turns_max is not None:
        exp_cfg.chat_turns_max = args.chat_turns_max

    runner = build_runner(exp_cfg, maze_env, maze_json_path=str(maze_path.resolve()))

    query_log: List[dict] = []
    if args.backend == "anthropic":
        _ensure_anthropic_api_key()
        claude_cfg = ClaudeAnthropicConfig()
        agent_inner = ClaudeAnthropicAgent(config=claude_cfg)
        model_id = claude_cfg.model
    else:
        llm_cfg = LocalLLMConfig(model=args.hf_model, device_map=args.device_map)
        agent_inner = LocalTransformersAgent(config=llm_cfg)
        model_id = llm_cfg.model

    agent = _AgentRecorder(agent_inner, query_log)

    try:
        result = runner.run(agent, verbose=args.verbose)
    except Exception as e:
        _persist_llm_queries_jsonl(out_dir, query_log)
        print(f"runner.run raised: {e}")
        return

    _persist_llm_queries_jsonl(out_dir, query_log)
    _write_episode_json(out_dir, result, backend=args.backend, model=model_id)

    transcript = result["transcript"]
    planned_actions = [rec["action"] for rec in transcript]

    env = load_maze_from_dict(task_data)
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
