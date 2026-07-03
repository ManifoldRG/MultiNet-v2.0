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

The in-process offline vLLM agent (`interface/agents/qwen_vllm.py`) now runs
**`Qwen/Qwen3.6-27B` (FP16) on A100-80GB** — we moved off the FP8 checkpoint (see
the decision record below). This keeps the pipeline contract simple for the
current target of **one Qwen worker per VM**: the worker process owns one vLLM
engine and no localhost server is required.

If we later need multiple local worker processes per GPU, or want external
clients to share one hot model, switch the same provider to **vLLM/SGLang
serving** via an OpenAI-compatible localhost API. A local server gives
continuous batching, prefix caching, paged KV, and (SGLang) multi-token
prediction (~3-5x decode), but it adds process supervision and health checks.

KTransformers remains a fallback to evaluate if vLLM/SGLang cannot hit the
required throughput or memory envelope on the A100 40GB worker shape.

## Decision (2026-07): moved off Qwen3.6-27B-FP8 → FP16 on A100-80GB

**We moved the Qwen runner off the `Qwen/Qwen3.6-27B-FP8` checkpoint to FP16/BF16
on the A100-80GB (`a2-ultragpu-1g`, the `qwen-fp16-80` image).** The 2026-06-30
distributed smokes ran FP8 through the offline vLLM agent on `a2-highgpu-1g`
(A100 40GB) and showed that **vLLM alone does not make FP8-on-A100-40GB fast
enough**. Both failure modes trace back to the A100 lacking native FP8 compute:

- **`enforce_eager: true` (no CUDA graphs):** loads (~14 min: 66 shards + engine
  init) and runs, but decode is slow. In smoke `qwen-smoke-eager-20260630-172236`
  the navigation maze (`v01_empty_room`) verified in ~18 min (~32 agent steps),
  but the harder mazes (`v04` key-door, `v05` switch-gate) ground at **~4-5
  min/step** — only ~25-27 steps in ~2 h, i.e. ~6-7 h for a single unit. FP8
  weights run through non-native Marlin weight-only kernels on A100 ("Your GPU
  does not have native support for FP8 computation ... may degrade performance"),
  which is the throughput ceiling.
- **`enforce_eager: false` (CUDA graphs — the speed lever):** OOMs at startup in
  vLLM's cudagraph memory profiling (`profile_cudagraph_memory ->
  _init_minimal_kv_cache_for_profiling -> torch.zeros`, +1.53 GiB) because at
  `gpu_memory_utilization=0.88` only ~498 MiB is free on the 40GB card. CUDA-graph
  capture needs headroom the 40GB shape does not have for a 27B model.

So FP8 bought memory but not speed on A100-40GB, and the CUDA-graph speed path
did not fit. The throughput path we adopted is **FP16/BF16 on A100-80GB
(`a2-ultragpu-1g`)**: native A100 compute plus headroom for CUDA graphs and a
large KV cache — shipped as the `qwen-fp16-80` image running `Qwen/Qwen3.6-27B`
with `enforce_eager: false`. (INT8 W8A8 on A100-40GB — A100 has native INT8 —
remains a cheaper alternative worth a test, per the section below.)

This is purely a model-throughput finding. The distributed pipeline itself —
coordinator work-stealing / queue hand-off to the next maze, and the
progress-aware stall detector — was validated end-to-end in the same smoke and is
not blocked by this (the smoke's actual goal: the freed worker correctly stole
the 3rd unit, and the monitor never false-stalled while `progress_total` climbed
for ~2 h with the verified count frozen at 1).

## Qwen INT8 and A100 80GB rental checkpoint

As of 2026-06-29, the `a100-qwen-vllm` boot disk is 150 GB (`/dev/root`: 145G
size, 70G used, 75G available). The existing Hugging Face cache is 29G, pip
cache is 6.4G, and the checked-out repo plus `.venv-qwen-vllm` is about 8.5G.
The candidate `Avesed/Qwen3.6-27B-INT8-W8A8` checkpoint is about 31.2 GB of
safetensors, close to the current `Qwen/Qwen3.6-27B-FP8` footprint of about
30.9 GB. It should fit on the image alongside the FP8 checkpoint without
deleting the existing model cache, leaving roughly 40 GB free after download.
If a future download needs extra temporary headroom, the pip cache is a safe
first cleanup target; deleting the FP8 checkpoint should not be necessary.

Current recommendation: test INT8 W8A8 on the existing A100 40GB worker before
renting an 80GB A100. A100 has native INT8 Tensor Cores, while the current FP8
checkpoint runs through non-native FP8 weight-only kernels on A100, so INT8 may
be competitive without changing GPU shape.

If we later need to test full BF16/FP16, or want more KV-cache margin, use
`a2-ultragpu-1g`: 1 NVIDIA A100 80GB, 12 vCPU, 170 GB RAM, and 1 bundled local
SSD. GCP zones found for this shape: `us-central1-a`, `us-central1-c`,
`us-east4-c`, `us-east5-a`, `us-east5-b`, `europe-west4-a`, and
`asia-southeast1-c`.

Pricing basis: Cloud Billing Catalog API for Compute Engine SKUs queried on
2026-06-29. Source docs: [Cloud Billing `services.skus.list`][billing-skus] and
[Compute Engine accelerator-optimized machines][a2-machines]. Hourly total below
is GPU + 12 A2 core-hours + 170 GiB A2 RAM-hours. This excludes boot persistent
disk, snapshots/images, network egress, taxes, and any committed-use/reservation
effects.

| Region | Example zone(s) | On-demand USD/h | Spot/preemptible USD/h |
| --- | --- | ---: | ---: |
| `us-central1` | `us-central1-a`, `us-central1-c` | 5.028 | 2.738 |
| `us-east5` | `us-east5-a`, `us-east5-b` | 5.524 | 1.656 |
| `europe-west4` | `europe-west4-a` | 5.536 | 2.601 |
| `us-east4` | `us-east4-c` | 5.663 | 2.272 |
| `asia-southeast1` | `asia-southeast1-c` | 6.202 | 3.130 |

For comparison, the current Tokyo `a2-highgpu-1g` A100 40GB shape is about
4.050 USD/h on-demand and 2.228 USD/h spot/preemptible using the same SKU
calculation.

[billing-skus]: https://docs.cloud.google.com/billing/docs/reference/rest/v1/services.skus/list
[a2-machines]: https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines

## Other deferred items

- **SSH auto-launcher** for the cluster (read `deploy/cluster.example.json` and
  start each node's role over SSH). Today we print the commands instead.
- **FP8 / quantized serving** as a throughput/VRAM lever (explicitly avoided now).
- **Multi-GPU tensor parallelism** for larger checkpoints.
- **systemd units** generated from the cluster inventory for unattended runners.
- **Pre-release cleanup action** to keep `docs/superpowers/**` and generated
  artifacts out of the release repo on merge (tracked separately).
