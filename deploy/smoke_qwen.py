"""Operator smoke: load Qwen3.6 via HF generate, do a tiny turn, report tok/s.

GPU + weights required for a real run. `--self-test` runs no model and only
exercises the kernel probe + verdict formatting (used by CI and the installer's
dry-run)."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Optional

DEFAULT_MODEL = "Qwen/Qwen3.6-27B"
TARGET_TOK_S = 100.0


def kernels_active() -> Dict[str, bool]:
    status: Dict[str, bool] = {}
    for mod in ("flash_attn", "fla", "causal_conv1d"):
        try:
            __import__(mod)
            status[mod] = True
        except Exception:  # noqa: BLE001 - any import error => unavailable
            status[mod] = False
    return status


def format_verdict(tok_per_s: float, target: float = TARGET_TOK_S) -> str:
    status = "MEETS" if tok_per_s >= target else "BELOW"
    return (
        f"[{status} target] {tok_per_s:.1f} tok/s vs {target:.0f} "
        f"(single-stream HF generate; see docs/future_directions.md for the vLLM path)"
    )


def _run_model(model: str, max_new_tokens: int) -> float:
    """Load the agent (no quantization) and time a single decode. Returns tok/s."""
    from interface.agents.qwen35_vl import Qwen35VLAgent, Qwen35VLConfig

    agent = Qwen35VLAgent(
        config=Qwen35VLConfig(
            model=model,
            load_in_4bit=False,
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_2",
            max_new_tokens=max_new_tokens,
            local_files_only=True,
        )
    )
    messages = [{"role": "user", "content": "Hello, how are you? Reply in one short sentence."}]
    t0 = time.perf_counter()
    agent(messages)
    elapsed = time.perf_counter() - t0
    out_tokens = int((agent.last_usage or {}).get("output_tokens", 0))
    if out_tokens == 0:
        print("WARNING: agent reported 0 output tokens; tok/s is not meaningful.")
    return (out_tokens / elapsed) if elapsed > 0 else 0.0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen3.6 load + throughput smoke.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--quick", action="store_true", help="Text-only (no image turn).")
    parser.add_argument("--self-test", action="store_true", help="No GPU: print kernels + a sample verdict.")
    args = parser.parse_args(argv)

    print(f"kernels: {kernels_active()}")
    if args.self_test:
        print(format_verdict(0.0))
        return 0

    tok_s = _run_model(args.model, args.max_new_tokens)
    print(format_verdict(tok_s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
