# VM Setup & Orchestration Tooling — Design

**Date:** 2026-06-24
**Branch:** `Distributed-run-pipeline`
**Status:** Approved (brainstorming) → ready for implementation plan

## Purpose

Make a fresh A100 VM go from a bare image to a **verified Qwen3.6 runner** that can
be snapshotted and replicated, and add the API-key and cluster tooling so the
4-node smoke test (2 Qwen workers + 1 Kimi API-client + coordinator) is
push-button to stand up.

Three deliverables, plus a forward-looking doc:

1. A single invasive setup script for a Qwen3.6 A100 VM + a load/throughput smoke.
2. A `.env.example` and a key round-trip ("Hello, how are you?") smoke test.
3. A single cluster-inventory file, a per-node preflight checker, and a
   launch-command printer.
4. `docs/future_directions.md` recording the vLLM/SGLang path to the throughput
   target and other deferred work.

## Context & constraints (decided during brainstorming)

- **Inference stays HF `generate()`.** We are *not* rebuilding the backend now.
  The existing `interface/agents/qwen35_vl.py` (`Qwen35VLAgent` /
  `Qwen35VLConfig`) is reused as-is.
- **Model:** default **`Qwen/Qwen3.6-27B`** (dense), apples-to-apples with the
  prior Qwen3.5-27B local runs. Keep the door open to A/B the
  **`Qwen/Qwen3.6-35B-A3B`** (MoE, ~3B active). Both are confirmed multimodal
  (native vision encoder), so the harness's image observations work on both.
- **Architecture note (why the kernels matter):** Qwen3.6 is a hybrid —
  Gated DeltaNet *linear-attention* sublayers (3 of every 4) plus gated softmax
  attention. `flash-linear-attention` and `causal-conv1d` accelerate the
  DeltaNet sublayers; `flash-attn` (FA2) accelerates the gated-attention
  sublayers.
- **Throughput target is ~100 tok/s on A100, no quantization.** Honest caveat:
  **single-stream HF `generate()` on the 27B dense hybrid will likely land below
  100 tok/s** even with all kernels. The smoke **measures and reports** actual
  tok/s rather than asserting it; vLLM/SGLang is documented as the real path.
- **Setup is invasive:** verify/install NVIDIA driver + CUDA toolkit (apt, root,
  idempotent — skipped when `nvidia-smi` is already healthy). Ubuntu/apt base
  image assumed.
- **Keys:** cover both providers the paid runs use — `ANTHROPIC_API_KEY`
  (Claude) and `MOONSHOT_API_KEY` (Kimi).
- **Orchestrator helper shape:** inventory file + per-node preflight +
  command-printer. **No SSH automation** (deferred).
- **Small footprint:** minimal code/fixture touches; no backend rewrite. (See
  the small-PR-footprint working preference.)

## File layout

A new top-level `deploy/` directory keeps shell/ops tooling out of the Python
`scripts/` package.

```
deploy/
  setup_qwen_vm.sh         # Item 1 — single invasive installer
  smoke_qwen.py            # Item 1 — load + generate; reports tok/s; PASS/FAIL
  check_api_keys.py        # Item 2 — "Hello, how are you?" round-trip per provider
  preflight.py             # Item 3 — per-node readiness checker (role-aware)
  print_launch_commands.py # Item 3 — cluster.json (+plan) -> exact role commands
  cluster.example.json     # Item 3 — the single IP-inventory file
.env.example               # Item 2 — ANTHROPIC_API_KEY + MOONSHOT_API_KEY template
docs/future_directions.md  # vLLM/SGLang path + deferred items
```

## Item 1 — Qwen3.6 VM setup + smoke

### `deploy/setup_qwen_vm.sh`

Single idempotent installer. Runs under `set -euo pipefail`. Tees output to
`deploy/setup_qwen_vm.log` and prints a final PASS/FAIL summary + a
"ready to snapshot" line.

Flags:
- `--model {27b|moe|both}` (default `27b`)
- `--venv PATH` (default `./.venv-qwen`)
- `--cuda 12.8`
- `--skip-driver`
- `--dry-run` (print actions, change nothing)

Steps, each guarded by an "already done?" check so the script is re-runnable on
a half-set-up box:

1. **Driver/CUDA.** If `nvidia-smi` healthy → skip. Else apt-install the
   recommended NVIDIA driver. Verify/install CUDA toolkit 12.8 (matches the
   cu128 wheels; provides `nvcc` for building kernels). Abort with a clear
   message if the driver's max CUDA < 12.8.
2. **Build deps:** `build-essential ninja-build git python3-venv python3-dev`.
3. **venv:** create at `--venv`; upgrade `pip wheel setuptools`.
4. **Torch (pinned):** `torch==2.9.1+cu128 torchvision torchaudio` from the
   cu128 index (the stack proven in `docs/qwen35_hf_local_pipeline_notes.md`).
5. **Transformers + runtime:** latest `transformers`, `accelerate`,
   `bitsandbytes`, `pillow`, `einops`, `huggingface_hub[cli]`. Pin the exact
   `transformers` version *after* the smoke confirms Qwen3.6 loads (noted in
   script comments).
6. **Performance kernels:** `flash-attn` (FA2, `--no-build-isolation`),
   `flash-linear-attention`, `causal-conv1d`. Each wrapped so a **build failure
   is a warning, not fatal** — the model still runs without them, just slower;
   the smoke reports which are active.
7. **Weights:** `hf download Qwen/Qwen3.6-27B` (+ `Qwen/Qwen3.6-35B-A3B` when
   `--model both|moe`) including `*.safetensors *.json *.jinja *.txt`. Respects
   the repo's `HF_HUB_ENABLE_HF_TRANSFER=0`.
8. **Verify:** import torch/transformers, print CUDA availability, then run
   `smoke_qwen.py --quick`.

Lint with `shellcheck` if available. Its only automated "test" is `--dry-run`
plus the final verify step.

### `deploy/smoke_qwen.py`

Builds `Qwen35VLAgent` with `load_in_4bit=False`, `torch_dtype="bfloat16"`,
`attn_implementation="flash_attention_2"` for the requested model id.

- Runs (a) a tiny text turn and (b) a tiny image turn (a maze PNG).
- Reports **decode tok/s** (`last_usage.output_tokens / elapsed`), which kernels
  are active, and peak VRAM.
- Prints a one-line verdict: measured tok/s **vs the 100 tok/s target**,
  explicitly labeled "single-stream HF generate; see docs/future_directions.md".
- `--quick` → text-only.
- `--functional` → runs the existing pipeline on the **validation_10 maze that
  Qwen3.5 failed (maze 2)** so the dense-vs-MoE difference can be eyeballed.
- GPU-only → operator-run, not CI.

### Minimal code/fixture touches

- Add `Qwen3_6ForConditionalGeneration` to `qwen35_vl.py:_model_class()` (it
  currently falls back to `AutoModel*`; an explicit entry is safer).
- Add `gridworld/fixtures/run_config.smoke_qwen36.json` (group `qwen36-27b`,
  model `Qwen/Qwen3.6-27B`, `load_in_4bit: false`).

## Item 2 — API keys

### `.env.example` (repo root)

```
ANTHROPIC_API_KEY=
MOONSHOT_API_KEY=
```

Header comment documents both load paths (env vars, or the existing gitignored
`api_key.txt` = line 1 Anthropic / line 2 Moonshot) and the
`set -a && source .env && set +a` idiom. **No new dotenv dependency** —
`check_api_keys.py` parses `.env` itself. The pipeline's existing key mechanism
(`interface/agents/api_keys.py`, env vars) is left untouched.

### `deploy/check_api_keys.py`

- Resolves each key in order: env → `.env` → `api_key.txt`.
- For each provider whose key is present, sends a real **"Hello, how are you?"**
  1-turn request through the *existing* `ClaudeAgent` / `KimiK26Agent`
  (`max_tokens ≈ 32`); prints provider → OK + reply snippet + token usage.
- Absent key → skip with a warning.
- Exit non-zero **only** if a *present* key fails auth.
- Reusing the real agents also smoke-tests the agent code path cheaply.

## Item 3 — Cluster inventory + preflight + command-printer

### `deploy/cluster.example.json`

The single inventory file:

```json
{
  "coordinator": {
    "host": "10.0.0.1",
    "port": 8765,
    "artifacts_root": "artifacts/cond/prompt",
    "run_set_id": "cond_prompt"
  },
  "storage_config": "gridworld/fixtures/storage_config.example.json",
  "workers": [
    {"name": "qwen-1", "host": "10.0.0.2", "model_group": "qwen36-27b", "hardware_profile": "local-gpu", "venv": "/opt/qwen/.venv-qwen"},
    {"name": "qwen-2", "host": "10.0.0.3", "model_group": "qwen36-27b", "hardware_profile": "local-gpu"}
  ],
  "api_clients": [
    {"name": "kimi", "model_group": "kimi-api", "runs_on": "coordinator"}
  ]
}
```

### `deploy/preflight.py`

Invoked per node: `--cluster cluster.json --node NAME --role {coordinator|worker|api-client}`.
Prints a PASS/FAIL table; non-zero exit on any FAIL. Role checks:

- **worker:** `nvidia-smi` + free VRAM; weights for its `model_group` present in
  the HF cache; kernels importable (`flash_attn`, `flash_linear_attention`,
  `causal_conv1d` — warn-only); `GET <coordinator>/status` reachable.
- **coordinator:** port bindable; `job_plan.json` present at `artifacts_root`
  (warn if not yet prepared); `storage_config` valid + `gsutil` present when
  storage is active.
- **api-client:** required API key present (delegates to `check_api_keys`).

### `deploy/print_launch_commands.py`

Reads `cluster.json` (+ the prepared `job_plan.json` if present) and prints the
exact copy-paste `multinet-run-pipeline --distributed-role …` command for each
node:

- coordinator: `coordinator-prepare`, `coordinator-serve`, `coordinator-finalize`
  (+ `--storage-config` when set).
- each worker: `worker --coordinator-url http://<coord-host>:<port>
  --model-group <g> --hardware-profile <p>`.
- api-client: `coordinator-run-api-client --model-group <g>`.

Cross-checks every worker/api-client `model_group` against the plan's groups and
warns on mismatch. **No SSH.**

## Item 4 — `docs/future_directions.md`

Records:
- **vLLM ≥0.19 / SGLang ≥0.5.10 serving path** as the real route to 100 tok/s:
  OpenAI-compatible endpoint → a `qwen_vllm` agent mirroring the existing Kimi
  agent; continuous batching, prefix caching; SGLang multi-token-prediction
  (3–5×).
- The HF-generate single-stream throughput ceiling (why the target may be
  missed today, and that this is expected — not a regression).
- Deferred items: SSH auto-launcher, FP8/quantization option, multi-GPU tensor
  parallelism, systemd units generated from `cluster.json`.

## Testing strategy

**CI-testable (pytest, no GPU/network) — added under `tests/`:**

- `test_check_api_keys.py` — monkeypatch agent calls + key sources: present-key →
  round-trip invoked & PASS; absent-key → skipped/warned (no call);
  present-but-failing → non-zero exit. No real API hits.
- `test_preflight.py` — mock `subprocess`/`socket`/`urllib` and a fake HF cache:
  assert each role's PASS/FAIL logic and exit codes.
- `test_print_launch_commands.py` — fixture `cluster.json` + synthetic
  `job_plan.json`: assert emitted commands carry the right role /
  `--coordinator-url` / `--model-group`, and that a worker `model_group` absent
  from the plan triggers the mismatch warning.
- `test_cluster_inventory.py` — schema validation for `cluster.example.json`
  (required fields, group strings) so a malformed inventory fails loudly.
- Extend the `_model_class()` test to assert `Qwen3_6ForConditionalGeneration`
  is preferred when present.

**Operator-run (GPU/keys/cluster required), NOT CI — documented as explicit
steps, mirroring the existing M6 pattern:**

- `setup_qwen_vm.sh` end-to-end + `smoke_qwen.py` (A100 + weights).
- `check_api_keys.py` live round-trip (real keys; tiny token spend).
- `preflight.py` across actual nodes.

## Out of scope (deferred to `future_directions.md`)

- vLLM/SGLang agent and serving integration.
- SSH auto-launcher.
- FP8 / quantization.
- Multi-GPU tensor parallelism.

## Research grounding (sources)

- [Qwen/Qwen3.6-27B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Qwen3.5 & Qwen3.6 Usage Guide — vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- [Qwen3-Next-80B-A3B-Instruct · Hugging Face](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
