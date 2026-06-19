# Qwen3.5 HF Local Pipeline Notes

These notes capture the local setup changes discovered while running the
`feature/run-pipeline` validation-10 test pass with the Hugging Face
Transformers-backed Qwen agent.

## Target

- Agent path: `interface/agents/qwen35_vl.py`
- Model: `Qwen/Qwen3.5-27B`
- Run config: `gridworld/fixtures/run_config.qwen35_27b_hf_validation10.json`
- Validation generation budget: `max_tokens=4096`
- Manifest: `gridworld/fixtures/manifest.json`
- Prompt variants: `Observation format` (`text_only`, `image_text`, `image_only`)
- Artifacts root: `artifacts/qwen35_27b_hf_validation10`

## Local Environment Requirements

The stock environment could not load Qwen3.5 because `transformers==4.44.0`
did not know the `qwen3_5` architecture. `transformers==5.10.2` recognizes the
model, but it requires a newer Torch than the original `torch==2.3.1`.

Working local versions after setup:

- Python: `/home/sean/mosaic/bin/python` (`Python 3.10.14`)
- `transformers==5.10.2`
- `torch==2.9.1+cu128`
- `torchvision==0.24.1+cu128`
- `torchaudio==2.9.1+cu128`
- `bitsandbytes==0.49.2`
- NVIDIA driver supports CUDA 12.9, so use CUDA 12.8 PyTorch wheels rather than
  CUDA 13 wheels.

An intermediate `torch==2.12.0+cu130` install pulled CUDA 13 wheels. Those were
removed because this driver reports CUDA 12.9 and PyTorch CUDA 13 could not
initialize. The final Python CUDA package set should contain only `*-cu12`
NVIDIA packages, not no-suffix CUDA 13 packages such as `cuda-toolkit`,
`nvidia-cublas`, or `nvidia-cudnn-cu13`.

Install command used for the driver-compatible Torch stack:

```bash
python -m pip install -U --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.9.1+cu128" "torchvision==0.24.1+cu128" "torchaudio==2.9.1+cu128"
```

Install command used for Qwen3.5 Transformers support:

```bash
python -m pip install -U "transformers>=5.10.2" "accelerate>=1.8.0" bitsandbytes
```

Note: this may conflict with `sentence-transformers==3.0.1`, which requires
`transformers<5.0.0`. The run-pipeline path used here does not depend on
`sentence-transformers`.

## Model Cache

The existing cache had only Qwen metadata and tokenizer files. The weight shards
had to be downloaded once:

```bash
hf download Qwen/Qwen3.5-27B \
  --include "*.safetensors" \
  --include "*.index.json" \
  --include "*.json" \
  --include "*.jinja" \
  --include "*.txt" \
  --max-workers 16
```

After download, the local cache was about 53 GB and contained 11 safetensors
shards.

## Runtime Config Notes

The 4-bit load path needs the model forced onto GPU:

```json
"device_map": {
  "": 0
},
"load_in_4bit": true
```

Using `"device_map": "auto"` caused bitsandbytes to reject CPU/disk-dispatched
modules. On the RTX 4090, the forced 4-bit load used about 22-23 GB VRAM and fit
with little headroom.

Qwen3.5 spent the output budget on chain-of-thought-style map analysis and did
not emit `FINAL_OUTPUT` until the chat template thinking mode was disabled. The
agent now passes:

```python
enable_thinking=False
```

The prompt template was also tightened to request no explanation before
`FINAL_OUTPUT`. After the first out-of-sandbox validation pass reached
`validation_10_v02_winding_corridor`, Qwen again spent the full 512 output
tokens on path analysis during `text_only` and failed all parse retries. The
per-step user prompt and parse-failure feedback now repeat the one-line output
contract at the decision point:

```text
Reply exactly as one line: FINAL_OUTPUT: <one valid action>
```

Stage-3 episode cache keys now include the condition set, prompt variant,
resolved `ExperimentConfig`, and runtime model config such as `provider`,
`model`, `temperature`, `max_tokens`, `enable_thinking`, quantization, and
attention settings. The sidecar at each `run_inputs.json` records both the full
model config and the runtime subset used in the hash. This prevents an old
episode from being silently reused after changing the prompt contract or the
generation budget. For the validation run, keep `max_tokens=4096`.

## Fast Attention Kernels

Transformers emitted this warning during Qwen3.5 loading:

```text
The fast path is not available because one of the required library is not installed.
Falling back to torch implementation.
```

It specifically points to:

- `flash-linear-attention`
- `causal-conv1d`

Without these kernels, Qwen3.5 generation can be impractically slow even after
the model loads successfully. Dedicated-hardware runs should install the
versions of these packages compatible with the active Torch/CUDA stack before
starting the full suite.

Dedicated-hardware setup should include:

```bash
python -m pip install -U flash-linear-attention causal-conv1d
```

If `causal-conv1d` imports with an undefined Torch/CUDA symbol after a normal
install, rebuild it against the active Torch wheel:

```bash
python -m pip install --force-reinstall --no-build-isolation --no-cache-dir --no-deps causal-conv1d==1.6.2.post1
```

Verify the imports before starting the full pipeline:

```bash
python -c "import fla; import causal_conv1d; print('fast kernels ok')"
```

If bitsandbytes emits a separate `kernels` package warning while all model
modules are resident on GPU, treat that as secondary. The blocking performance
warning for this run is the missing Qwen fast attention path above.

## Current Local Run Status

Completed locally:

- Fast-kernel install and verification:

```text
flash-linear-attention==0.5.0
causal-conv1d==1.6.2.post1
import fla; import causal_conv1d -> fast kernels ok
```

- Static scoring for all 16 manifest rows (`all`: tests 1-3) under
  `artifacts/qwen35_27b_hf_validation10/tasks`.
- Focused pipeline tests:

```bash
python -m pytest tests/test_run_pipeline.py tests/test_interface_token_usage.py -q
```

Result: `28 passed`.

- Qwen smoke run on `validation_10_v01_empty_room` for all three observation
  variants:

```bash
python -m scripts.run_pipeline \
  --run-config gridworld/fixtures/run_config.qwen35_27b_hf_smoke.json \
  --manifest gridworld/fixtures/manifest.json \
  --artifacts-root artifacts/qwen35_27b_hf_smoke \
  --run-set-id qwen35_27b_hf_smoke \
  --seeds 0 \
  --conditions "Observation format" \
  --force
```

Result: 3/3 successful runs over V01:

- `text_only`: success, 11 steps, optimality ratio 1.0
- `image_text`: success, 12 steps, optimality ratio 0.9167
- `image_only`: success, 12 steps, optimality ratio 0.9167

An out-of-sandbox validation pass was later started from the same artifact root
and killed manually after a long run. It did not finish the full suite or write
the top-level aggregate reports, but it did write per-episode `episode.json`,
`run_inputs.json`, and `run_score.json` files for 19 fresh completed rows. The
run directory still also contains 21 older pre-fix rows, so the partial summary
below explicitly separates fresh rows from stale rows.

Partial summary artifacts:

- `artifacts/qwen35_27b_hf_validation10/partial_results_summary/summary.md`
- `artifacts/qwen35_27b_hf_validation10/partial_results_summary/summary.json`
- `artifacts/qwen35_27b_hf_validation10/partial_results_summary/fresh_rows.csv`
- `artifacts/qwen35_27b_hf_validation10/partial_results_summary/coverage_audit.csv`
- `artifacts/qwen35_27b_hf_validation10/partial_results_summary/run_config.remaining_tasks.json`

Fresh-only aggregate report artifacts regenerated from existing `episode.json`
and `run_score.json` files without model calls:

- `artifacts/qwen35_27b_hf_validation10/partial_results_summary/partial_fresh_reports/episode_runs.fresh.jsonl`
- `artifacts/qwen35_27b_hf_validation10/partial_results_summary/partial_fresh_reports/partial_report_metadata.json`
- `artifacts/qwen35_27b_hf_validation10/partial_results_summary/partial_fresh_reports/reports/qwen35_27b_hf_validation10_partial_fresh/scoring_calibration_summary.json`
- `artifacts/qwen35_27b_hf_validation10/partial_results_summary/partial_fresh_reports/reports/qwen35_27b_hf_validation10_partial_fresh/complexity_distance_summary.json`
- `artifacts/qwen35_27b_hf_validation10/partial_results_summary/partial_fresh_reports/reports/qwen35_27b_hf_validation10_partial_fresh/mechanism_ordering_pairs.json`
- `artifacts/qwen35_27b_hf_validation10/partial_results_summary/partial_fresh_reports/reports/qwen35_27b_hf_validation10_partial_fresh/models/Qwen_Qwen3.5-27B.json`

Fresh partial results:

```text
fresh_completed_rows: 19
fresh_coverage_needed_rows: 29
stale_rows: 21
physically_missing_rows: 8
successes: 3
parse_failed: 7
truncated: 9
input_tokens: 1,167,145
output_tokens: 583,219
total_tokens: 1,750,364
recorded_llm_latency: 665.96 minutes
```

All fresh successes were on `validation_10_v01_empty_room`; the run reached
through `validation_10_v07_chain_sk/text_only` before it was killed. Episode
files do not record full wall-clock duration, only per-query `llm_latency_s`, so
the timing above is summed model-call latency rather than total process runtime.
The remaining fresh coverage is split across `test1` (11 rows), `test2` (6
rows), and `test3` (12 rows). All 16 manifest tasks already have
`canonical_paths.json` and `scored_static.json`; every existing episode file has
a `run_score.json` sidecar.
The generated remaining-tasks config keeps `max_tokens=4096` and targets the 10
tasks that still have stale or missing rows. The current `0.1.1` Stage-3 cache
key includes the prompt/observation config and runtime model config, so patched
non-`--force` runs should invalidate older `0.1.0` sidecars. Keep `--force`
when writing into this mixed artifact root if you want the overwrite to be
explicit.

Earlier, inside the sandboxed Codex environment, fresh PyTorch processes
reported:

```text
CUDA initialization: CUDA driver initialization failed
torch 2.9.1+cu128, torch.cuda.is_available() == False
```

On June 18, 2026, this sandboxed shell still reported:

```text
Can't initialize NVML
torch 2.9.1+cu128, torch.cuda.is_available() == False, device_count == 0
```

The fast attention kernel imports still succeeded in the same shell:

```text
import fla; import causal_conv1d -> fast kernels ok
```

At the same time `nvidia-smi` still saw the RTX 4090. This appears to be a
local CUDA runtime/driver state issue rather than a pipeline artifact issue.
Additional diagnostics on June 11, 2026:

```text
ctypes CDLL("libcuda.so.1") loaded /usr/lib/x86_64-linux-gnu/libcuda.so.575.57.08
cuInit(0) -> 100
cuDeviceGetCount(...) -> 3, count=0
```

Recent kernel logs also showed repeated NVIDIA driver allocation failures:

```text
NVRM: Check failed: Out of memory [NV_ERR_NO_MEMORY]
NVRM: sysmemConstruct_IMPL: *** Cannot allocate sysmem through fb heap
```

After removing the stray CUDA 13 Python wheels and force-reinstalling the CUDA
12.8 wheel libraries, Torch imported correctly again, but CUDA initialization
still failed because the direct driver API still failed.

An attempted targeted reset also failed because the RTX 4090 is the primary
display GPU:

```text
nvidia-smi --gpu-reset -i 0
GPU Reset couldn't run because GPU 00000000:01:00.0 is the primary GPU.
```

The next full run should be started on dedicated hardware, or after a driver/GPU
reset or reboot on this machine. Do not start the full Qwen suite until both
checks pass:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import fla; import causal_conv1d; print('fast kernels ok')"
```

Expected result: CUDA is available, the device name prints `NVIDIA GeForce RTX
4090` or the dedicated hardware GPU, and both fast-kernel imports succeed.

## Full Run Command

Once CUDA initialization is healthy and the fast kernels are installed, run:

```bash
python -m scripts.run_pipeline \
  --run-config gridworld/fixtures/run_config.qwen35_27b_hf_validation10.json \
  --manifest gridworld/fixtures/manifest.json \
  --artifacts-root artifacts/qwen35_27b_hf_validation10 \
  --run-set-id qwen35_27b_hf_validation10 \
  --seeds 0 \
  --conditions "Observation format" \
  --force
```

Expected report outputs:

- `artifacts/qwen35_27b_hf_validation10/episode_runs.jsonl`
- `artifacts/qwen35_27b_hf_validation10/reports/qwen35_27b_hf_validation10/scoring_calibration_summary.json`
- `artifacts/qwen35_27b_hf_validation10/reports/qwen35_27b_hf_validation10/complexity_distance_summary.json`
- `artifacts/qwen35_27b_hf_validation10/reports/qwen35_27b_hf_validation10/mechanism_ordering_pairs.json`
- `artifacts/qwen35_27b_hf_validation10/reports/qwen35_27b_hf_validation10/models/Qwen_Qwen3.5-27B.json`

The current `artifacts/qwen35_27b_hf_validation10/runs` directory contains
episodes from aborted/pre-fix attempts. Use `--force` for the next full run if
you want to overwrite those attempts explicitly.
