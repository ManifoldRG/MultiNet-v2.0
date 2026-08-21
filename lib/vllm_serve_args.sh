#!/usr/bin/env bash
# vllm_serve_args — single source of truth for the served-vLLM (Qwen) argument
# string. SOURCE this file; do not execute it.
#
# TWO-TIER PHASE BOUNDARY (docs/qwen-two-tier-rerun-design.md). The served-vLLM
# path IGNORES the run-config's max_model_len/gpu_memory_utilization, so the KV
# tradeoff is controlled here, at the serve line. Env knobs (all optional):
#
#   QWEN_MAX_MODEL_LEN           (default 16384)  phase 2 -> 96000
#   QWEN_MAX_NUM_SEQS            (default 64)     phase 2 -> ~3 (KV-cache cost)
#   QWEN_GPU_MEMORY_UTILIZATION  (default 0.9)
#
# UNSET env reproduces today's exact phase-1 hard-coded string. The phase-2
# reload (lib/distributed_start.sh::reload_gpu_worker) exports the phase-2 knobs
# before relaunching the worker; served serve-args cannot change at runtime, so
# switching phases is a real ~14-min server reload.
#
# The served model name is site-specific (--served-model-name "$MODEL") and stays
# at each call site; --port/--dtype/--trust-remote-code are NOT phase-dependent
# and live here so the phase-1 string is reproduced verbatim.
vllm_serve_args() {
  local mml="${QWEN_MAX_MODEL_LEN:-16384}"
  local seqs="${QWEN_MAX_NUM_SEQS:-64}"
  local gmu="${QWEN_GPU_MEMORY_UTILIZATION:-0.9}"
  echo "--port 8000 --gpu-memory-utilization ${gmu} --max-model-len ${mml} --max-num-seqs ${seqs} --dtype bfloat16 --trust-remote-code"
}
