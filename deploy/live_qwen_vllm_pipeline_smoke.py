"""Operator smoke: run one pipeline task with the offline Qwen vLLM agent."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a tiny live Qwen vLLM pipeline smoke.")
    parser.add_argument("--manifest", default="gridworld/fixtures/manifest.smoke_eval.json")
    parser.add_argument("--task", default="validation_10_v01_empty_room")
    parser.add_argument("--artifacts-root", default="artifacts/qwen36_27b_fp8_vllm_live_smoke")
    parser.add_argument("--run-set-id", default="qwen36_27b_fp8_vllm_live_smoke")
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B-FP8")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    from interface.agents.qwen_vllm import QwenVLLMAgent, QwenVLLMConfig
    from scripts.run_pipeline import run_pipeline

    agent = QwenVLLMAgent(
        config=QwenVLLMConfig(
            model=args.model,
            temperature=0.0,
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=True,
            local_files_only=True,
            enable_thinking=False,
        )
    )
    payloads = run_pipeline(
        manifest_path=args.manifest,
        experiment=args.task,
        agent=agent,
        agent_name=f"{args.model}-vLLM-live-smoke",
        seeds=[0],
        artifacts_root=args.artifacts_root,
        run_set_id=args.run_set_id,
        difficulty_max_static_score=1000.0,
        force=args.force,
    )
    print("payload_keys", sorted(payloads.keys()))
    print("calibration", payloads.get("scoring_calibration_summary"))
    print("last_usage", agent.last_usage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
