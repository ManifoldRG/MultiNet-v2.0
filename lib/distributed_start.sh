#!/usr/bin/env bash
# Distributed-run START recipes. SOURCE this file; do not execute it.
# Provides start_coordinator + start_worker, generalized from launch_qwen_smoke.sh
# (coordinator prepare/serve; GPU worker) and launch_smoke_4vm.sh (API worker + key
# delivery). Consumed by launch_distributed.sh (replaces its stub start hooks).

worker_field() {  # $1 vm-name  $2 field  -> value from TOPO_JSON on stdout
  printf '%s' "$TOPO_JSON" | python3 -c '
import json, sys
name, field = sys.argv[1], sys.argv[2]
for w in json.load(sys.stdin).get("workers", []):
    if w.get("name") == name:
        print(w.get(field, "")); break
' "$1" "$2"
}

start_coordinator() {  # uses globals COORD ZONE RUN_ID RUN_CONFIG MANIFEST [SEEDS] [DIFFICULTY_MAX] [CONDITIONS] [PROMPT_VARIANT]
  local seeds="${SEEDS:-0}" diff="${DIFFICULTY_MAX:-1000.0}"
  # Conditional sweeps: --conditions is required (run_pipeline's H1/H2 guard
  # rejects a mispaired prepare) and --prompt-variant selects one dedup variant.
  # Empty when unset -> the remote guarded appends skip both flags.
  local conditions="${CONDITIONS:-}" prompt_variant="${PROMPT_VARIANT:-}"
  gcloud compute ssh "$COORD" --zone "$ZONE" \
    --command "RUN_ID='$RUN_ID' RUN_CONFIG='$RUN_CONFIG' MANIFEST='$MANIFEST' SEEDS='$seeds' DIFFICULTY_MAX='$diff' CONDITIONS='$conditions' PROMPT_VARIANT='$prompt_variant' bash -s" <<'REMOTE'
set -euo pipefail
cd ~/MultiNet-v2.0
source .venv-multinet/bin/activate
if python - <<'PY'
import socket, time
# Wait for a just-killed prior coordinator-serve to release :8765 rather than
# failing on the transient race. The killed serve's worker connections linger in
# TIME_WAIT (~60s) and block a fresh bind without SO_REUSEADDR, so allow 90s.
deadline = time.time() + 90
ok = False
while time.time() < deadline:
    s = socket.socket()
    try:
        s.bind(("0.0.0.0", 8765))
        ok = True
    except OSError:
        ok = False
    finally:
        s.close()
    if ok:
        break
    time.sleep(1)
raise SystemExit(0 if ok else 1)
PY
then :; else echo "Port 8765 still in use on the coordinator after 90s." >&2; exit 1; fi
mkdir -p "artifacts/$RUN_ID"
prepare_args=()
[[ -n "${CONDITIONS:-}" ]] && prepare_args+=(--conditions "$CONDITIONS")
[[ -n "${PROMPT_VARIANT:-}" ]] && prepare_args+=(--prompt-variant "$PROMPT_VARIANT")
python -m scripts.run_pipeline \
  --distributed-role coordinator-prepare \
  --run-config "$RUN_CONFIG" \
  --manifest "$MANIFEST" \
  --seeds "$SEEDS" \
  --artifacts-root "artifacts/$RUN_ID" \
  --run-set-id "$RUN_ID" \
  --difficulty-max-static-score "$DIFFICULTY_MAX" \
  ${prepare_args[@]+"${prepare_args[@]}"}
nohup python -m scripts.run_pipeline \
  --distributed-role coordinator-serve \
  --artifacts-root "artifacts/$RUN_ID" \
  --host 0.0.0.0 --port 8765 \
  > "artifacts/$RUN_ID/coordinator-serve.log" 2>&1 &
echo "$!" > "artifacts/$RUN_ID/coordinator-serve.pid"
coord_up=0
for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8765/status >/dev/null; then coord_up=1; break; fi
  sleep 2
done
[[ "$coord_up" -eq 1 ]] || { echo "Coordinator not healthy on :8765 within ~60s." >&2; exit 1; }
REMOTE
}

# start_coordinator_combined: like start_coordinator but prepares ONE massive job from
# several batches (MASSIVE_BATCHES = space-separated batch indices) via
# scripts.prepare_combined_job, so the fleet stays saturated instead of idling through
# each batch's hard-maze tail. Globals: COORD ZONE RUN_ID MANIFEST [SEEDS]
# [DIFFICULTY_MAX] MASSIVE_BATCHES [SWEEP_TOPO].
start_coordinator_combined() {
  local seeds="${SEEDS:-0}" diff="${DIFFICULTY_MAX:-1000.0}" batch_args="" n
  for n in $MASSIVE_BATCHES; do batch_args="$batch_args --batch $n"; done
  gcloud compute ssh "$COORD" --zone "$ZONE" \
    --command "RUN_ID='$RUN_ID' MANIFEST='$MANIFEST' SEEDS='$seeds' DIFFICULTY_MAX='$diff' BATCH_ARGS='$batch_args' SWEEP_TOPO='${SWEEP_TOPO:-}' bash -s" <<'REMOTE'
set -euo pipefail
cd ~/MultiNet-v2.0
source .venv-multinet/bin/activate
if python - <<'PY'
import socket, time
deadline = time.time() + 90
ok = False
while time.time() < deadline:
    s = socket.socket()
    try:
        s.bind(("0.0.0.0", 8765)); ok = True
    except OSError:
        ok = False
    finally:
        s.close()
    if ok:
        break
    time.sleep(1)
raise SystemExit(0 if ok else 1)
PY
then :; else echo "Port 8765 still in use on the coordinator after 90s." >&2; exit 1; fi
mkdir -p "artifacts/$RUN_ID"
# shellcheck disable=SC2086
SWEEP_TOPO="$SWEEP_TOPO" python -m scripts.prepare_combined_job \
  --artifacts-root "artifacts/$RUN_ID" --run-set-id "$RUN_ID" \
  --manifest "$MANIFEST" --difficulty-max-static-score "$DIFFICULTY_MAX" \
  --seeds $SEEDS $BATCH_ARGS
nohup python -m scripts.run_pipeline \
  --distributed-role coordinator-serve \
  --artifacts-root "artifacts/$RUN_ID" \
  --host 0.0.0.0 --port 8765 \
  > "artifacts/$RUN_ID/coordinator-serve.log" 2>&1 &
echo "$!" > "artifacts/$RUN_ID/coordinator-serve.pid"
coord_up=0
for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8765/status >/dev/null; then coord_up=1; break; fi
  sleep 2
done
[[ "$coord_up" -eq 1 ]] || { echo "Coordinator not healthy on :8765 within ~60s." >&2; exit 1; }
REMOTE
}

start_worker() {  # $1 vm  $2 coord_ip  — metadata from TOPO_JSON
  local vm="$1" coord_ip="$2" kind group provider model
  kind="$(worker_field "$vm" kind)"
  group="$(worker_field "$vm" model_group)"
  provider="$(worker_field "$vm" provider)"
  model="$(worker_field "$vm" model)"

  if [[ "$kind" == "gpu" ]]; then
    # QWEN_MAX_MODEL_LEN/NUM_SEQS/GPU_MEMORY_UTILIZATION are the two-tier phase
    # knobs — forwarded (empty in phase 1 -> vllm_serve_args defaults reproduce
    # today's string; set by reload_gpu_worker in phase 2). See lib/vllm_serve_args.sh.
    gcloud compute ssh "$vm" --zone "$ZONE" \
      --command "RUN_ID='$RUN_ID' COORD_IP='$coord_ip' GROUP='$group' MODEL='$model' CONCURRENCY='${WORKER_CONCURRENCY:-16}' QWEN_MAX_MODEL_LEN='${QWEN_MAX_MODEL_LEN:-}' QWEN_MAX_NUM_SEQS='${QWEN_MAX_NUM_SEQS:-}' QWEN_GPU_MEMORY_UTILIZATION='${QWEN_GPU_MEMORY_UTILIZATION:-}' bash -s" <<'REMOTE'
set -euo pipefail
cd ~/MultiNet-v2.0
source .venv-qwen-vllm/bin/activate
source lib/vllm_serve_args.sh   # vllm_serve_args(): two-tier phase-dependent serve args
mkdir -p "$HOME/multinet-worker-artifacts/$RUN_ID"

# 1. Persistent vLLM OpenAI server (continuous batching across concurrent episode
#    requests). It is REUSED across batches — only (re)launched if not already
#    serving on :8000 — so switching batches costs no model reload. It intentionally
#    holds the GPU, so there is no orphan-EngineCore problem for next-batch to clean.
#
# TWO-TIER PHASE-TRANSITION POINT. Serve args come from vllm_serve_args() (phase 1
# defaults = 16384/64/0.9; phase 2 = 96000/3 via env). The reuse guard below skips
# relaunch ONLY when a server is already up; the phase-2 reload_gpu_worker runs
# stop_gpu_worker FIRST (GPU freed, server down), so both the curl and pgrep checks
# fail there and the phase-2 server WILL (re)launch. See lib/vllm_serve_args.sh.
if ! curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && ! pgrep -f 'vllm serve' >/dev/null 2>&1; then
  nohup env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    vllm serve "$MODEL" --served-model-name "$MODEL" $(vllm_serve_args) \
    > "$HOME/multinet-worker-artifacts/vllm_server.log" 2>&1 &
  echo "$!" > "$HOME/multinet-worker-artifacts/vllm_server.pid"
fi
# Wait for the server to finish loading the model (first launch ~ minutes).
for _ in $(seq 1 150); do
  curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
  sleep 10
done
curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1 \
  || { echo "vllm server never became ready on :8000" >&2; exit 1; }

# 2. Worker: uses the served qwen_vllm_api agent (per the run-config's base_url),
#    running CONCURRENCY episodes in parallel so the server batches their prefills.
nohup env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.run_pipeline \
  --distributed-role worker \
  --coordinator-url "http://$COORD_IP:8765" \
  --artifacts-root "$HOME/multinet-worker-artifacts/$RUN_ID" \
  --worker-state "$HOME/multinet-worker-artifacts/$RUN_ID/worker_state.json" \
  --model-group "$GROUP" \
  --hardware-profile local-gpu \
  --local-model-cache "$MODEL" \
  --worker-concurrency "$CONCURRENCY" \
  > "$HOME/multinet-worker-artifacts/$RUN_ID/worker.log" 2>&1 &
echo "$!" > "$HOME/multinet-worker-artifacts/$RUN_ID/worker.pid"
REMOTE
    return $?
  fi

  # API worker: resolve the provider's credential env var, quote its value.
  local key_var key_q
  case "$provider" in
    claude) key_var=ANTHROPIC_API_KEY ;;
    kimi)   key_var=MOONSHOT_API_KEY ;;
    *) echo "no credential mapping for provider '$provider' (worker $vm)" >&2; return 1 ;;
  esac
  printf -v key_q '%q' "${!key_var:-}"
  # API_WORKER_ROLE selects the driver on API VMs: "worker" (serial, default)
  # or "lockstep-worker" (batch-API lockstep runner; R1 uses this). For lockstep,
  # API_WORKER_CONCURRENCY is MAX_BATCHES (working-set size; R1: 50) and the
  # coordinator must be serving with --stale-after-seconds >= worst-case batch
  # round (see docs/batch-api-lockstep-runner-design.md).
  # Exactly ONE lockstep worker per API model group — a second one double-pays.
  local api_role="${API_WORKER_ROLE:-worker}"
  case "$api_role" in
    worker|lockstep-worker) ;;
    *) echo "invalid API_WORKER_ROLE '$api_role' (worker|lockstep-worker)" >&2; return 1 ;;
  esac
  gcloud compute ssh "$vm" --zone "$ZONE" \
    --command "RUN_ID='$RUN_ID' COORD_IP='$coord_ip' GROUP='$group' API_ROLE='$api_role' API_CONC='${API_WORKER_CONCURRENCY:-1}' bash -s" <<REMOTE
set -euo pipefail
export ${key_var}=${key_q}
cd ~/MultiNet-v2.0
source .venv-multinet/bin/activate
mkdir -p "\$HOME/multinet-worker-artifacts/\$RUN_ID"
nohup python -m scripts.run_pipeline \\
  --distributed-role "\$API_ROLE" \\
  --coordinator-url "http://\$COORD_IP:8765" \\
  --artifacts-root "\$HOME/multinet-worker-artifacts/\$RUN_ID" \\
  --worker-state "\$HOME/multinet-worker-artifacts/\$RUN_ID/worker_state.json" \\
  --model-group "\$GROUP" \\
  --hardware-profile api-client \\
  --worker-concurrency "\$API_CONC" \\
  > "\$HOME/multinet-worker-artifacts/\$RUN_ID/worker.log" 2>&1 &
echo "\$!" > "\$HOME/multinet-worker-artifacts/\$RUN_ID/worker.pid"
REMOTE
}

# stop_gpu_worker VM: tear down a GPU worker BEFORE a batch restart, freeing the
# GPU. Ships lib/gpu_teardown.sh to the VM and runs gpu_teardown_local there,
# which SIGTERMs the worker AND the orphaned vLLM EngineCore holding the GPU and
# polls until it frees. Returns nonzero (FAIL-CLOSED) if the GPU will not free,
# so the caller must NOT start a new worker (it would OOM). See lib/gpu_teardown.sh.
stop_gpu_worker() {  # $1 vm
  local vm="$1" here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  gcloud compute ssh "$vm" --zone "$ZONE" --command "bash -s" < <(
    cat "$here/gpu_teardown.sh"
    printf '\ngpu_teardown_local\n'
  )
}

# reload_gpu_worker VM COORD_IP: two-tier PHASE-2 reload of one GPU VM. Serve args
# (max_model_len etc.) cannot change at runtime, so phase 2 requires a real server
# reload (~14 min). ORDERING INVARIANT: stop_gpu_worker runs FIRST and fail-closes
# if the GPU will not free — so the server is provably DOWN before start_worker. That
# is what lets start_worker's serve-reuse guard (curl :8000 + pgrep 'vllm serve') fall
# through to (re)launch instead of skipping: after the teardown both checks fail. We
# then export the phase-2 serve env so start_worker forwards it into vllm_serve_args
# on the VM (max-model-len 96000, few seqs — the KV-cache cost of the big context).
# Reuses start_worker verbatim (no new SSH mechanics). Caller sets ZONE/RUN_ID/TOPO_JSON
# (+ optional WORKER_CONCURRENCY for the reduced phase-2 fan-out).
reload_gpu_worker() {  # $1 vm  $2 coord_ip
  local vm="$1" coord_ip="$2"
  stop_gpu_worker "$vm" \
    || { echo "[reload_gpu_worker] $vm: GPU did not free — NOT relaunching (would OOM)" >&2; return 1; }
  # RACE FIX (fail-closed): gpu_teardown only guarantees the GPU is FREE; the `vllm
  # serve` PARENT can linger a beat after its EngineCore dies. start_worker's reuse
  # guard matches `pgrep -f 'vllm serve'`, so a lingering parent would make it SKIP the
  # relaunch → the ~25-min readiness loop times out → aborted reload. Make the guard's
  # precondition true BY CONSTRUCTION: kill any leftover parent and bounded-wait for it
  # to clear before start_worker. Fed via stdin (bash -s), so the remote shell's own
  # command line is just "bash -s" and pgrep -f 'vllm serve' cannot self-match. Same
  # ssh plumbing as stop_gpu_worker. If it never clears, fail closed (no relaunch).
  gcloud compute ssh "$vm" --zone "$ZONE" --command "bash -s" <<'REMOTE' \
    || { echo "[reload_gpu_worker] $vm: lingering 'vllm serve' did not clear — NOT relaunching" >&2; return 1; }
set -uo pipefail
pkill -f 'vllm serve' 2>/dev/null || true   # may already be gone — ignore failure
for _ in $(seq 1 30); do                    # up to ~60s, poll every 2s
  pgrep -f 'vllm serve' >/dev/null 2>&1 || exit 0
  sleep 2
done
echo "'vllm serve' parent still present after ~60s" >&2
exit 1
REMOTE
  QWEN_MAX_MODEL_LEN=96000 QWEN_MAX_NUM_SEQS="${QWEN_PHASE2_MAX_NUM_SEQS:-3}" \
    start_worker "$vm" "$coord_ip"
}
