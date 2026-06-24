# Future Directions

Deferred work intentionally kept out of the current scope. Each item records the
"why now" trigger so we can pick it up without re-deriving context.

## Inference engine: vLLM / SGLang serving (the real path to 100 tok/s)

The Qwen runner currently uses in-process HuggingFace `model.generate()`
(`interface/agents/qwen35_vl.py`). Single-stream HF generate on a 27B dense
hybrid (Gated DeltaNet + gated attention) will likely land **below** the
~100 tok/s A100 target even with `flash-attn` + `flash-linear-attention` +
`causal-conv1d` and bf16. `deploy/smoke_qwen.py` measures and reports the actual
rate so this is visible rather than assumed.

The realistic route to the target without quantization is a serving engine:

- **vLLM ≥ 0.19** or **SGLang ≥ 0.5.10**, both of which support Qwen3.6 and expose
  an **OpenAI-compatible** HTTP API. A local server gives continuous batching,
  prefix caching, paged KV, and (SGLang) multi-token prediction (~3–5× decode).
- Integration would add a `qwen_vllm` agent that POSTs to `localhost` — nearly
  identical in shape to the existing `interface/agents/kimi_k26.py` (also an
  OpenAI-compatible client). No change to the coordinator/worker lifecycle:
  the worker VM runs its own server, the agent just points at it.

This is deferred because we chose not to rebuild the backend now.

## Other deferred items

- **SSH auto-launcher** for the cluster (read `deploy/cluster.example.json` and
  start each node's role over SSH). Today we print the commands instead.
- **FP8 / quantized serving** as a throughput/VRAM lever (explicitly avoided now).
- **Multi-GPU tensor parallelism** for larger checkpoints.
- **systemd units** generated from the cluster inventory for unattended runners.
- **Pre-release cleanup action** to keep `docs/superpowers/**` and generated
  artifacts out of the release repo on merge (tracked separately).
