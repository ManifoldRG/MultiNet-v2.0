# shellcheck shell=bash
# GPU worker teardown that actually frees the GPU before a batch restart.
#
# Why this exists: the GPU worker runs vLLM 0.19 (v1 engine) IN-PROCESS via
# `LLM()`. vLLM v1 runs its GPU-holding `EngineCore` as a SEPARATE child process
# with a different cmdline, so `pkill -f 'distributed-role worker'` (which matches
# only the parent) leaves the EngineCore orphaned holding ~70 GiB -> the next
# batch's `LLM()` OOMs. This teardown SIGTERMs the worker AND every GPU
# compute-app (engine-agnostic: catches the orphan whatever its cmdline), then
# POLLS `nvidia-smi` until the GPU reports no compute-apps.
#
# It deliberately does NOT escalate to SIGKILL: SIGTERM lets vLLM's EngineCore
# release CUDA cleanly, whereas SIGKILL on a live CUDA process wedges the GPU and
# hangs the next CUDA init. If the GPU will not free after SIGTERM within the
# timeout, the function FAILS CLOSED (nonzero) so the caller refuses to start a
# new worker that would only OOM — surface it to a human instead.
#
# `gpu_teardown_local` runs ON the GPU VM and uses only local `nvidia-smi`/`pkill`
# so it is unit-testable off-GPU with PATH shims (tests/test_gpu_teardown.py).

# List PIDs of processes currently holding GPU compute memory (one per line).
_gpu_compute_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | tr -d ' ' | grep -E '^[0-9]+$' || true
}

# SIGTERM the worker + GPU apps, then wait for the GPU to actually free.
# Env overrides (used by tests): GPU_TEARDOWN_TIMEOUT (s), GPU_TEARDOWN_INTERVAL (s).
gpu_teardown_local() {
  local timeout="${GPU_TEARDOWN_TIMEOUT:-90}" interval="${GPU_TEARDOWN_INTERVAL:-3}"

  # 1. Ask the worker parent to stop (best-effort; it has no vLLM cleanup path).
  pkill -TERM -f 'distributed-role worker' 2>/dev/null || true

  # 2. SIGTERM whatever still holds the GPU (the orphaned vLLM EngineCore). This
  #    is the load-bearing step the old pkill-by-cmdline missed.
  local pids
  pids="$(_gpu_compute_pids)"
  if [ -n "$pids" ]; then
    # shellcheck disable=SC2086
    kill -TERM $pids 2>/dev/null || true
  fi

  # 3. Poll until the GPU reports no compute-apps (condition-based, not a sleep).
  local waited=0
  while [ "$waited" -lt "$timeout" ]; do
    if [ -z "$(_gpu_compute_pids)" ]; then
      echo "gpu_teardown: GPU free after ${waited}s" >&2
      return 0
    fi
    sleep "$interval"
    waited=$((waited + interval))
  done

  # 4. Still held after SIGTERM -> FAIL CLOSED. Do NOT SIGKILL (wedges CUDA).
  echo "gpu_teardown: GPU still has compute-apps after ${timeout}s SIGTERM;" \
       "refusing to start a new worker (would OOM). Manual attention needed:" \
       "$(_gpu_compute_pids | tr '\n' ' ')" >&2
  return 1
}
