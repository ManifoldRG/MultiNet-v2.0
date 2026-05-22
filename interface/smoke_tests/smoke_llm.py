"""End-to-end smoke test with a real LLM agent (Anthropic Claude or local Transformers).

Replaces PR1 ``smoke_claude.py``; uses ``flush_episode_log`` for on-disk artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interface.agents import (
    ClaudeAnthropicAgent,
    ClaudeAnthropicConfig,
    DEFAULT_LOCAL_MODEL,
    LocalLLMConfig,
    LocalTransformersAgent,
)
from interface.config import ExperimentConfig
from interface.episode_log import flush_episode_log
from interface.loader import default_maze_path, load_task
from interface.runner import build_runner


def _configure_logging(level_name: str) -> None:
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


def _persist_llm_queries_jsonl(out_dir: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path = out_dir / "llm_queries.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class _AgentRecorder:
    """Delegates to a real agent and records each assistant reply for ``llm_queries.jsonl``."""

    __slots__ = ("_inner", "_records", "_query_seq")

    def __init__(self, inner: Callable[[list[dict]], str], records: list[dict[str, Any]]) -> None:
        self._inner = inner
        self._records = records
        self._query_seq = 0

    def __call__(self, messages: list[dict]) -> str:
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
            "End-to-end LLM maze episode. Writes llm_queries.jsonl plus exhaustive "
            "episode log (episode.json, frames/, queries/). "
            "Anthropic runs in the cloud; --backend local uses Hugging Face."
        ),
    )
    parser.add_argument("--maze", default="V10_distractor_chain.json")
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
        help="Structured logs from interface agents (default: WARNING).",
    )
    parser.add_argument(
        "--backend",
        choices=("anthropic", "local"),
        default="anthropic",
        help="anthropic: Claude API. local: Hugging Face Transformers on this machine.",
    )
    parser.add_argument(
        "--hf-model",
        default=DEFAULT_LOCAL_MODEL,
        help="With --backend local: Hugging Face model id (default: %(default)s).",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='With --backend local: device_map for from_pretrained (e.g. "auto", "cuda:0").',
    )
    parser.add_argument("--prompting", default="standard", choices=["minimal", "standard", "verbose"])
    parser.add_argument("--observation", default="image_text", choices=["text_only", "image_text", "image_only"])
    parser.add_argument("--context-window", default="last3", choices=["current", "last3"])
    parser.add_argument(
        "--querying",
        default="step_by_step",
        choices=["step_by_step", "subgoal", "full_trajectory"],
    )
    parser.add_argument(
        "--chat-history",
        choices=("stateless", "rolling", "full"),
        default="stateless",
        help="ExperimentConfig.chat_history (default: stateless).",
    )
    parser.add_argument(
        "--chat-turns-max",
        type=int,
        default=3,
        metavar="N",
        help="ExperimentConfig.chat_turns_max for rolling mode (default: 3).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        metavar="SECS",
        help="Anthropic API read timeout in seconds (default: 180). Ignored for --backend local.",
    )
    args = parser.parse_args()
    _configure_logging(args.log_level)

    maze_path = default_maze_path(args.maze)
    if not maze_path.is_file():
        raise SystemExit(f"Missing maze file: {maze_path}")

    maze_stem = maze_path.stem
    suffix = f"_{args.tag}" if args.tag else ""
    out_slug = "claude" if args.backend == "anthropic" else "local"
    out_dir = Path(__file__).resolve().parent / "results" / f"smoke_{maze_stem}_{out_slug}{suffix}"

    backend, spec = load_task(maze_path)
    config = ExperimentConfig(
        prompting=args.prompting,
        observation=args.observation,
        context_window=args.context_window,
        querying=args.querying,
        chat_history=args.chat_history,
        chat_turns_max=args.chat_turns_max,
    )
    runner = build_runner(config, backend, spec)

    query_log: list[dict[str, Any]] = []
    if args.backend == "anthropic":
        _ensure_anthropic_api_key()
        claude_cfg = ClaudeAnthropicConfig(timeout=args.timeout)
        agent_inner = ClaudeAnthropicAgent(config=claude_cfg)
        model_id = claude_cfg.model
    else:
        llm_cfg = LocalLLMConfig(model=args.hf_model, device_map=args.device_map)
        agent_inner = LocalTransformersAgent(config=llm_cfg)
        model_id = llm_cfg.model

    agent = _AgentRecorder(agent_inner, query_log)

    try:
        result = runner.run(agent, verbose=args.verbose, maze_path=maze_path)
    except Exception as exc:
        out_dir.mkdir(parents=True, exist_ok=True)
        _persist_llm_queries_jsonl(out_dir, query_log)
        raise SystemExit(f"runner.run raised: {exc}") from exc

    episode_path = flush_episode_log(result, out_dir)
    _persist_llm_queries_jsonl(out_dir, query_log)

    report = out_dir / "report.txt"
    report.write_text(
        report.read_text(encoding="utf-8")
        + "\n"
        + "\n".join(
            [
                f"backend={args.backend}",
                f"model={model_id}",
                f"llm_queries={len(query_log)}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"success={result['success']} end_reason={result['end_reason']} steps={result['steps_used']}")
    print(f"queries={result['query_count']} llm_queries={len(query_log)}")
    print(f"episode={episode_path}")
    print(f"out={out_dir}")


if __name__ == "__main__":
    main()
