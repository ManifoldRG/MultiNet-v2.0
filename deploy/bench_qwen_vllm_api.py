"""Benchmark a vLLM OpenAI-compatible Qwen server with maze-style prompts."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import time
from pathlib import Path

from interface.config import ExperimentConfig
from interface.loader import load_task
from interface.runner import build_runner
from prompting_experiments.prompt_templates import feedback as feedback_templates


DEFAULT_TASKS = [
    "mazes/validation_10/V01_empty_room.json",
    "mazes/validation_10/V04_single_key.json",
    "mazes/validation_10/V05_single_switch.json",
]


def _append_instruction(message: dict, instruction: str) -> dict:
    message = dict(message)
    content = message.get("content", "")
    if isinstance(content, str):
        message["content"] = f"{content}\n\n{instruction}"
        return message
    if isinstance(content, list):
        blocks = [dict(block) if isinstance(block, dict) else block for block in content]
        for block in reversed(blocks):
            if isinstance(block, dict) and block.get("type") == "text":
                block["text"] = f"{block.get('text', '')}\n\n{instruction}"
                break
        else:
            blocks.append({"type": "text", "text": instruction})
        message["content"] = blocks
        return message
    message["content"] = instruction
    return message


def build_maze_messages(task_path: str | Path, *, output_tokens: int) -> list[dict]:
    backend, spec = load_task(task_path)
    backend.configure(spec)
    _, state, _ = backend.reset(seed=spec.seed)
    config = ExperimentConfig(observation="text_only", querying="full_trajectory")
    runner = build_runner(config, backend, spec)
    system_prompt, user_message = runner.build_prompt_message(
        state,
        feedback_templates.INITIAL_FEEDBACK,
        [],
    )
    instruction = (
        "Throughput benchmark mode: reply with FINAL_OUTPUT followed by "
        f"about {output_tokens} valid action tokens. Use only MOVE_NORTH, "
        "MOVE_SOUTH, MOVE_EAST, MOVE_WEST, INTERACT, DONE. Do not explain."
    )
    return [
        {"role": "system", "content": system_prompt},
        _append_instruction(user_message, instruction),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--requests", type=int, default=96)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--task", action="append", dest="tasks")
    args = parser.parse_args(argv)

    from interface.agents.qwen_vllm_api import QwenVLLMAPIAgent, QwenVLLMAPIConfig

    tasks = args.tasks or DEFAULT_TASKS
    prompts = [build_maze_messages(task, output_tokens=args.max_tokens) for task in tasks]
    prompt_iter = itertools.cycle(prompts)
    jobs = [next(prompt_iter) for _ in range(args.requests)]

    def run_one(messages: list[dict]) -> dict[str, int]:
        agent = QwenVLLMAPIAgent(
            QwenVLLMAPIConfig(
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                enable_thinking=False,
            )
        )
        agent(messages)
        return agent.last_usage or {}

    t0 = time.perf_counter()
    usages: list[dict[str, int]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for usage in pool.map(run_one, jobs):
            usages.append(usage)
    elapsed = time.perf_counter() - t0

    output_tokens = sum(int(u.get("output_tokens", 0)) for u in usages)
    total_tokens = sum(int(u.get("total_tokens", 0)) for u in usages)
    input_tokens = sum(int(u.get("input_tokens", 0)) for u in usages)
    output_tok_s = output_tokens / elapsed if elapsed > 0 else 0.0
    total_tok_s = total_tokens / elapsed if elapsed > 0 else 0.0

    print(f"requests {len(usages)}")
    print(f"concurrency {args.concurrency}")
    print(f"elapsed_seconds {elapsed:.3f}")
    print(f"input_tokens {input_tokens}")
    print(f"output_tokens {output_tokens}")
    print(f"total_tokens {total_tokens}")
    print(f"output_tokens_per_second {output_tok_s:.3f}")
    print(f"total_tokens_per_second {total_tok_s:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
