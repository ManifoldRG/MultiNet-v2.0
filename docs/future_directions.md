# Future Directions

Deferred work intentionally kept out of the current scope. Each item records the
"why now" trigger so we can pick it up without re-deriving context.

## Inference engine: offline vLLM first, server later

The Qwen runner currently uses in-process HuggingFace `model.generate()`
(`interface/agents/qwen35_vl.py`). Single-stream HF generate on a 27B dense
hybrid (Gated DeltaNet + gated attention) will likely land **below** the
~100 tok/s A100 target even with `flash-attn` + `flash-linear-attention` +
`causal-conv1d` and bf16. `deploy/smoke_qwen.py` measures and reports the actual
rate so this is visible rather than assumed.

The first migration step is the in-process offline vLLM agent
(`interface/agents/qwen_vllm.py`) with `Qwen/Qwen3.6-27B-FP8`. This keeps the
pipeline contract simple for the current target of **one Qwen worker per VM**:
the worker process owns one vLLM engine and no localhost server is required.

If we later need multiple local worker processes per GPU, or want external
clients to share one hot model, switch the same provider to **vLLM/SGLang
serving** via an OpenAI-compatible localhost API. A local server gives
continuous batching, prefix caching, paged KV, and (SGLang) multi-token
prediction (~3-5x decode), but it adds process supervision and health checks.

KTransformers remains a fallback to evaluate if vLLM/SGLang cannot hit the
required throughput or memory envelope on the A100 40GB worker shape.

## Other deferred items

- **SSH auto-launcher** for the cluster (read `deploy/cluster.example.json` and
  start each node's role over SSH). Today we print the commands instead.
- **FP8 / quantized serving** as a throughput/VRAM lever (explicitly avoided now).
- **Multi-GPU tensor parallelism** for larger checkpoints.
- **systemd units** generated from the cluster inventory for unattended runners.
- **Pre-release cleanup action** to keep `docs/superpowers/**` and generated
  artifacts out of the release repo on merge (tracked separately).
