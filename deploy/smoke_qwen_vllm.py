"""Operator smoke: load Qwen via offline vLLM, generate, and report tok/s."""

from __future__ import annotations

import argparse
import time

DEFAULT_MODEL = "Qwen/Qwen3.6-27B-FP8"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen offline-vLLM load + throughput smoke.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--enforce-eager", dest="enforce_eager", action="store_true", default=True)
    parser.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--prompt",
        default="Continue with plain text. Count upward from 1, separated by commas, and do not stop early.",
    )
    args = parser.parse_args(argv)

    from interface.agents.qwen_vllm import QwenVLLMAgent, QwenVLLMConfig

    load_t0 = time.perf_counter()
    agent = QwenVLLMAgent(
        config=QwenVLLMConfig(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            local_files_only=args.local_files_only,
            enable_thinking=False,
        )
    )
    load_s = time.perf_counter() - load_t0

    # Warmup a short generation so the timed pass is less dominated by one-time setup.
    agent.config.max_tokens = min(16, args.max_tokens)
    agent([{"role": "user", "content": args.prompt}])

    agent.config.max_tokens = args.max_tokens
    t0 = time.perf_counter()
    out = agent([{"role": "user", "content": args.prompt}])
    elapsed = time.perf_counter() - t0
    out_tokens = int((agent.last_usage or {}).get("output_tokens", 0))
    tok_s = out_tokens / elapsed if elapsed > 0 and out_tokens > 0 else 0.0

    print(f"model {args.model}")
    print(f"load_seconds {load_s:.3f}")
    print(f"output_tokens {out_tokens}")
    print(f"generate_seconds {elapsed:.3f}")
    print(f"tokens_per_second {tok_s:.3f}")
    print(f"usage {agent.last_usage}")
    print(f"sample {out[:500].replace(chr(10), ' ')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
