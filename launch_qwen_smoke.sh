#!/usr/bin/env bash
set -euo pipefail

# Shared cost-safety net (single source of truth).
source "$(dirname "${BASH_SOURCE[0]}")/lib/cost_safety.sh"

# Qwen-only distributed smoke: 1 coordinator + 2 Qwen A100 vLLM workers (no Kimi).
# Validates coordinator work-stealing across 2 machines (3 mazes -> 3 units) and
# the progress-aware stall path, with enforce_eager off to restore CUDA graphs.
#
# Required:
#   export MAX_RUN_DURATION=...   # GCP-native cost floor; no default
# Optional:
#   ZONE=us-central1-a  RUN_ID=...  FRESH=1
#   COORD=mn-qwen-coord  QWEN1=mn-qwen-1  QWEN2=mn-qwen-2
#   QWEN_IMAGE=qwen-fp16-80  QWEN_MODEL=Qwen/Qwen3.6-27B
#   QWEN_DTYPE=bfloat16  ENFORCE_EAGER=true  QWEN_WORKERS_PER_VM=32
#
# Subcommands (no MAX_RUN_DURATION needed):
#   ./launch_qwen_smoke.sh stop      # STOP all 3 VMs, keep disks + data
#   ./launch_qwen_smoke.sh delete    # delete all 3 VMs incl. disks (post-export)

ZONE="${ZONE:-us-central1-a}"
RUN_ID="${RUN_ID:-qwen-smoke-$(date +%Y%m%d-%H%M%S)}"

COORD="${COORD:-mn-qwen-coord}"
QWEN1="${QWEN1:-mn-qwen-1}"
QWEN2="${QWEN2:-mn-qwen-2}"

COORD_IMAGE="${COORD_IMAGE:-multinet-coordinator-n2-20260629}"
QWEN_IMAGE="${QWEN_IMAGE:-qwen-fp16-80}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3.6-27B}"
QWEN_DTYPE="${QWEN_DTYPE:-bfloat16}"
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-8192}"
QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.95}"
QWEN_PORT="${QWEN_PORT:-8000}"
QWEN_WORKERS_PER_VM="${QWEN_WORKERS_PER_VM:-32}"
QWEN_MAX_NUM_SEQS="${QWEN_MAX_NUM_SEQS:-64}"
QWEN_MAX_NUM_BATCHED_TOKENS="${QWEN_MAX_NUM_BATCHED_TOKENS:-8192}"
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
STALL_MINUTES="${STALL_MINUTES:-45}"

declare -A VM_CREATED

# --------------------------------------------------------------------------- #
# Cleanup subcommands (no creds needed)
# --------------------------------------------------------------------------- #

cmd_stop() {
  require_gcloud
  log "spinning down (STOP — disks & data preserved) in $ZONE: $COORD $QWEN1 $QWEN2"
  cs_stop_vms "$ZONE" "$COORD" "$QWEN1" "$QWEN2"
}

cmd_delete() {
  require_gcloud
  log "DELETING (incl. disks/data) in $ZONE: $COORD $QWEN1 $QWEN2"
  cs_delete_vms "$ZONE" "$COORD" "$QWEN1" "$QWEN2"
}

# --------------------------------------------------------------------------- #
# Instance lifecycle
# --------------------------------------------------------------------------- #

ensure_instance() {
  local name="$1" image="$2"
  if instance_exists "$name" "$ZONE"; then
    log "instance exists (reused): $name — GCP floor NOT applied; relying on on-VM watchdog"
    VM_CREATED["$name"]=0
    return
  fi
  log "creating $name from $image (max-run-duration=$MAX_RUN_DURATION, termination=STOP)"
  cs_create_instance "$name" "$image" "$ZONE"
  VM_CREATED["$name"]=1
}

start_coordinator() {
  local qwen_max_in_flight=$(( QWEN_WORKERS_PER_VM * 2 ))
  log "preparing and serving coordinator (Qwen-only)"
  gcloud compute ssh "$COORD" --zone "$ZONE" \
    --command "RUN_ID='$RUN_ID' FRESH='${FRESH:-0}' ENFORCE_EAGER='$ENFORCE_EAGER' QWEN_MODEL='$QWEN_MODEL' QWEN_DTYPE='$QWEN_DTYPE' QWEN_MAX_MODEL_LEN='$QWEN_MAX_MODEL_LEN' QWEN_GPU_MEMORY_UTILIZATION='$QWEN_GPU_MEMORY_UTILIZATION' QWEN_PORT='$QWEN_PORT' QWEN_WORKERS_PER_VM='$QWEN_WORKERS_PER_VM' QWEN_MAX_IN_FLIGHT='$qwen_max_in_flight' bash -s" <<'REMOTE'
set -euo pipefail
cd ~/MultiNet-v2.0
source .venv-multinet/bin/activate

python - <<PY
import json, os
from pathlib import Path
cfg = json.loads(Path("gridworld/fixtures/run_config.smoke_eval_qwen_kimi.json").read_text())
qwen_model = os.environ["QWEN_MODEL"]
cfg["description"] = f"Qwen-only smoke (3 mazes): 2 {qwen_model} vLLM workers, no Kimi."
cfg["models"] = {}
cfg["models"]["qwen36_27b_vllm"] = {
    "provider": "qwen_vllm_api",
    "model": qwen_model,
    "base_url": f"http://127.0.0.1:{os.environ['QWEN_PORT']}/v1",
    "api_key": "EMPTY",
    "temperature": 0.0,
    "max_tokens": 4096,
    "timeout": 240,
    "max_model_len": int(os.environ["QWEN_MAX_MODEL_LEN"]),
    "gpu_memory_utilization": float(os.environ["QWEN_GPU_MEMORY_UTILIZATION"]),
    "dtype": os.environ["QWEN_DTYPE"],
    "enforce_eager": os.environ.get("ENFORCE_EAGER", "false").lower() == "true",
    "local_files_only": True,
    "enable_thinking": False,
    "group": "qwen36-27b",
    "hardware_profile": "local-gpu",
    "worker_count": int(os.environ["QWEN_WORKERS_PER_VM"]) * 2,
    "max_in_flight": int(os.environ["QWEN_MAX_IN_FLIGHT"]),
    "tasks": ["all"],
}
Path("/tmp/run_config.qwen_only.json").write_text(json.dumps(cfg, indent=2) + "\n")
print("model:", qwen_model)
print("dtype:", cfg["models"]["qwen36_27b_vllm"]["dtype"])
print("enforce_eager:", cfg["models"]["qwen36_27b_vllm"]["enforce_eager"])
PY

if python - <<'PY'
import socket
s = socket.socket()
try:
    s.bind(("0.0.0.0", 8765))
except OSError:
    raise SystemExit(1)
finally:
    s.close()
PY
then :; else echo "Port 8765 already in use on the coordinator." >&2; exit 1; fi

if [[ "${FRESH:-0}" == "1" ]]; then rm -rf "artifacts/$RUN_ID"; fi
mkdir -p "artifacts/$RUN_ID"

python -m scripts.run_pipeline \
  --distributed-role coordinator-prepare \
  --run-config /tmp/run_config.qwen_only.json \
  --manifest gridworld/fixtures/manifest.smoke_eval.json \
  --seeds 0 \
  --artifacts-root "artifacts/$RUN_ID" \
  --run-set-id "$RUN_ID" \
  --difficulty-max-static-score 1000.0

nohup python -m scripts.run_pipeline \
  --distributed-role coordinator-serve \
  --artifacts-root "artifacts/$RUN_ID" \
  --host 0.0.0.0 --port 8765 \
  > "artifacts/$RUN_ID/coordinator-serve.log" 2>&1 &
echo "$!" > "artifacts/$RUN_ID/coordinator-serve.pid"

coord_up=0
for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8765/status; then echo; coord_up=1; break; fi
  sleep 2
done
[[ "$coord_up" -eq 1 ]] || { echo "Coordinator not healthy on :8765 within ~60s." >&2; exit 1; }
REMOTE
}

start_qwen_worker() {
  local vm="$1" coord_ip="$2"
  log "starting Qwen vLLM server + worker clients: $vm"
  gcloud compute ssh "$vm" --zone "$ZONE" --command "RUN_ID='$RUN_ID' COORD_IP='$coord_ip' QWEN_MODEL='$QWEN_MODEL' QWEN_DTYPE='$QWEN_DTYPE' QWEN_MAX_MODEL_LEN='$QWEN_MAX_MODEL_LEN' QWEN_GPU_MEMORY_UTILIZATION='$QWEN_GPU_MEMORY_UTILIZATION' QWEN_PORT='$QWEN_PORT' QWEN_WORKERS_PER_VM='$QWEN_WORKERS_PER_VM' QWEN_MAX_NUM_SEQS='$QWEN_MAX_NUM_SEQS' QWEN_MAX_NUM_BATCHED_TOKENS='$QWEN_MAX_NUM_BATCHED_TOKENS' ENFORCE_EAGER='$ENFORCE_EAGER' bash -s" <<'REMOTE'
set -euo pipefail
cd ~/MultiNet-v2.0
source .venv-qwen-vllm/bin/activate
mkdir -p "$HOME/multinet-worker-artifacts/$RUN_ID"

if ! curl -fsS "http://127.0.0.1:$QWEN_PORT/v1/models" >/dev/null 2>&1; then
  eager_arg=()
  if [[ "${ENFORCE_EAGER,,}" == "true" ]]; then eager_arg=(--enforce-eager); fi
  # TWO-TIER PHASE-TRANSITION POINT. This smoke launcher already parameterizes the
  # serve line from its OWN env knobs (QWEN_MAX_MODEL_LEN/QWEN_MAX_NUM_SEQS/
  # QWEN_GPU_MEMORY_UTILIZATION — same names as lib/vllm_serve_args.sh), so phase-2
  # values flow straight through here; it keeps its site-specific flags (--host,
  # --max-num-batched-tokens, --enable-prefix-caching, --generation-config, eager)
  # and its smoke defaults (8192/0.95), so it deliberately does NOT call
  # vllm_serve_args() (which would add --trust-remote-code and drop those flags).
  setsid env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 vllm serve "$QWEN_MODEL" \
    --host 127.0.0.1 \
    --port "$QWEN_PORT" \
    --served-model-name "$QWEN_MODEL" \
    --dtype "$QWEN_DTYPE" \
    --max-model-len "$QWEN_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$QWEN_GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$QWEN_MAX_NUM_SEQS" \
    --max-num-batched-tokens "$QWEN_MAX_NUM_BATCHED_TOKENS" \
    --enable-prefix-caching \
    --generation-config vllm \
    "${eager_arg[@]}" \
    > "$HOME/multinet-worker-artifacts/$RUN_ID/vllm-server.log" 2>&1 < /dev/null &
  echo "$!" > "$HOME/multinet-worker-artifacts/$RUN_ID/vllm-server.pid"
fi

server_up=0
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$QWEN_PORT/v1/models" >/dev/null 2>&1; then server_up=1; break; fi
  sleep 5
done
[[ "$server_up" -eq 1 ]] || { echo "vLLM server not healthy on :$QWEN_PORT." >&2; exit 1; }

for idx in $(seq 1 "$QWEN_WORKERS_PER_VM"); do
  setsid env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.run_pipeline \
    --distributed-role worker \
    --coordinator-url "http://$COORD_IP:8765" \
    --artifacts-root "$HOME/multinet-worker-artifacts/$RUN_ID/worker_$idx" \
    --worker-state "$HOME/multinet-worker-artifacts/$RUN_ID/worker_state_$idx.json" \
    --model-group qwen36-27b \
    --hardware-profile local-gpu \
    --local-model-cache "$QWEN_MODEL" \
    > "$HOME/multinet-worker-artifacts/$RUN_ID/worker_$idx.log" 2>&1 < /dev/null &
  echo "$!" > "$HOME/multinet-worker-artifacts/$RUN_ID/worker_$idx.pid"
done
echo "Qwen worker clients started: $QWEN_WORKERS_PER_VM"
REMOTE
}

print_summary() {
  local mins
  mins="$(watchdog_minutes "$MAX_RUN_DURATION")"
  cat <<EOF

Launched $RUN_ID (Qwen-only: $COORD + $QWEN1 + $QWEN2).
  GCP floor (Layer 0): STOP after $MAX_RUN_DURATION (data preserved).
  On-VM watchdog (Layer 1): guest shutdown at +${mins}m.
  qwen_image=$QWEN_IMAGE.
  qwen_model=$QWEN_MODEL.
  qwen_dtype=$QWEN_DTYPE.
  qwen_workers_per_vm=$QWEN_WORKERS_PER_VM.
  qwen_server_port=$QWEN_PORT.
  enforce_eager=$ENFORCE_EAGER.

Hands-off monitor (Layer 2) — run under Claude /loop (45-min progress-aware stall):
  ./monitor_run.sh --once --coord $COORD --qwen1 $QWEN1 --qwen2 $QWEN2 \\
      --zone $ZONE --run-id $RUN_ID --dest ./artifacts-pulled/$RUN_ID \\
      --stall-minutes $STALL_MINUTES --state-file ./.monitor_state.$RUN_ID.json --complete-actions

Cleanup:
  ZONE=$ZONE $0 stop      # spin down, KEEP data
  ZONE=$ZONE $0 delete    # remove VMs + disks, AFTER export
EOF
}

main() {
  case "${1:-}" in
    stop) cmd_stop; exit 0 ;;
    delete) cmd_delete; exit 0 ;;
  esac

  require_max_run_duration
  validate_run_id
  require_gcloud

  log "run id: $RUN_ID"
  log "cost floor: GCP stop after $MAX_RUN_DURATION; on-VM watchdog +$(watchdog_minutes "$MAX_RUN_DURATION")m"

  ensure_instance "$COORD" "$COORD_IMAGE"
  ensure_instance "$QWEN1" "$QWEN_IMAGE"
  ensure_instance "$QWEN2" "$QWEN_IMAGE"

  wait_for_ssh "$COORD" "$ZONE" || exit 1
  wait_for_ssh "$QWEN1" "$ZONE" || exit 1
  wait_for_ssh "$QWEN2" "$ZONE" || exit 1

  arm_watchdog "$COORD" "$ZONE" "${VM_CREATED[$COORD]:-0}" || exit 1
  arm_watchdog "$QWEN1" "$ZONE" "${VM_CREATED[$QWEN1]:-0}" || exit 1
  arm_watchdog "$QWEN2" "$ZONE" "${VM_CREATED[$QWEN2]:-0}" || exit 1

  assert_no_resource_policy "$COORD" "$ZONE" || exit 1
  assert_no_resource_policy "$QWEN1" "$ZONE" || exit 1
  assert_no_resource_policy "$QWEN2" "$ZONE" || exit 1

  local COORD_IP
  COORD_IP="$(internal_ip "$COORD" "$ZONE")"
  log "coordinator internal IP: $COORD_IP"

  start_coordinator
  start_qwen_worker "$QWEN1" "$COORD_IP"
  start_qwen_worker "$QWEN2" "$COORD_IP"

  print_summary
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
