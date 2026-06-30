#!/usr/bin/env bash
set -euo pipefail

# Qwen-only distributed smoke: 1 coordinator + 2 Qwen A100 vLLM workers (no Kimi).
# Validates coordinator work-stealing across 2 machines (3 mazes -> 3 units) and
# the progress-aware stall path, with enforce_eager off to restore CUDA graphs.
#
# Required:
#   export MAX_RUN_DURATION=...   # GCP-native cost floor; no default
# Optional:
#   ZONE=asia-northeast1-c  RUN_ID=...  FRESH=1
#   COORD=mn-qwen-coord  QWEN1=mn-qwen-1  QWEN2=mn-qwen-2  ENFORCE_EAGER=false
#
# Subcommands (no MAX_RUN_DURATION needed):
#   ./launch_qwen_smoke.sh stop      # STOP all 3 VMs, keep disks + data
#   ./launch_qwen_smoke.sh delete    # delete all 3 VMs incl. disks (post-export)

ZONE="${ZONE:-asia-northeast1-c}"
RUN_ID="${RUN_ID:-qwen-smoke-$(date +%Y%m%d-%H%M%S)}"

COORD="${COORD:-mn-qwen-coord}"
QWEN1="${QWEN1:-mn-qwen-1}"
QWEN2="${QWEN2:-mn-qwen-2}"

COORD_IMAGE="${COORD_IMAGE:-multinet-coordinator-n2-20260629}"
QWEN_IMAGE="${QWEN_IMAGE:-multinet-qwen36-fp8-vllm-a100-20260629}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
STALL_MINUTES="${STALL_MINUTES:-45}"

declare -A VM_CREATED

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

is_valid_duration() {
  local s="${1:-}"
  [[ -n "$s" ]] || return 1
  [[ "$s" =~ ^([0-9]+d)?([0-9]+h)?([0-9]+m)?([0-9]+s)?$ ]] || return 1
  return 0
}

parse_duration_seconds() {
  local s="${1:-}"
  is_valid_duration "$s" || { echo "invalid duration: '${s}'" >&2; return 1; }
  local total=0 num unit rest="$s"
  while [[ "$rest" =~ ^([0-9]+)([dhms]) ]]; do
    num="${BASH_REMATCH[1]}"; unit="${BASH_REMATCH[2]}"
    case "$unit" in
      d) total=$(( total + num * 86400 )) ;;
      h) total=$(( total + num * 3600 )) ;;
      m) total=$(( total + num * 60 )) ;;
      s) total=$(( total + num )) ;;
    esac
    rest="${rest#"${BASH_REMATCH[0]}"}"
  done
  echo "$total"
}

watchdog_minutes() {
  local secs
  secs="$(parse_duration_seconds "${1:-}")" || return 1
  echo "$(( secs / 60 + 60 ))"
}

require_max_run_duration() {
  if [[ -z "${MAX_RUN_DURATION:-}" ]]; then
    echo "MAX_RUN_DURATION is required (no default). e.g. export MAX_RUN_DURATION=6h" >&2
    return 1
  fi
  if ! is_valid_duration "${MAX_RUN_DURATION}"; then
    echo "MAX_RUN_DURATION malformed: '${MAX_RUN_DURATION}'." >&2
    return 1
  fi
}

validate_run_id() {
  [[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Bad RUN_ID: $RUN_ID" >&2; return 1; }
}

require_gcloud() { command -v gcloud >/dev/null 2>&1 || { echo "gcloud not on PATH." >&2; return 1; }; }

cmd_stop() {
  require_gcloud
  log "spinning down (STOP — disks & data preserved) in $ZONE: $COORD $QWEN1 $QWEN2"
  local vm
  for vm in "$COORD" "$QWEN1" "$QWEN2"; do
    gcloud compute instances stop "$vm" --zone "$ZONE" --quiet || true
  done
}

cmd_delete() {
  require_gcloud
  log "DELETING (incl. disks/data) in $ZONE: $COORD $QWEN1 $QWEN2"
  local vm
  for vm in "$COORD" "$QWEN1" "$QWEN2"; do
    gcloud compute instances delete "$vm" --zone "$ZONE" --quiet || true
  done
}

instance_exists() { gcloud compute instances describe "$1" --zone "$ZONE" >/dev/null 2>&1; }

ensure_instance() {
  local name="$1" image="$2"
  if instance_exists "$name"; then
    log "instance exists (reused): $name — GCP floor NOT applied; relying on on-VM watchdog"
    VM_CREATED["$name"]=0
    return
  fi
  log "creating $name from $image (max-run-duration=$MAX_RUN_DURATION, termination=STOP)"
  gcloud compute instances create "$name" \
    --zone "$ZONE" \
    --source-machine-image="$image" \
    --max-run-duration="$MAX_RUN_DURATION" \
    --instance-termination-action=STOP
  VM_CREATED["$name"]=1
}

wait_for_ssh() {
  local name="$1"
  log "waiting for SSH: $name"
  for _ in $(seq 1 60); do
    if gcloud compute ssh "$name" --zone "$ZONE" --command "true" >/dev/null 2>&1; then
      log "SSH ready: $name"; return
    fi
    sleep 5
  done
  echo "Timed out waiting for SSH on $name" >&2; exit 1
}

arm_watchdog() {
  local name="$1" mins
  mins="$(watchdog_minutes "$MAX_RUN_DURATION")"
  log "arming on-VM shutdown watchdog on $name (+${mins}m)"
  if gcloud compute ssh "$name" --zone "$ZONE" \
       --command "sudo shutdown -c 2>/dev/null || true; sudo shutdown -h +${mins}" >/dev/null 2>&1; then
    log "watchdog armed: $name (+${mins}m)"; return 0
  fi
  if [[ "${VM_CREATED[$name]:-0}" == "1" ]]; then
    log "WARNING: could not arm watchdog on $name; GCP floor still protects this fresh VM"; return 0
  fi
  echo "FATAL: could not arm watchdog on reused VM $name (no GCP floor)." >&2; exit 1
}

assert_no_resource_policy() {
  local name="$1" disk_uris disk_uri disk_name policies
  disk_uris="$(gcloud compute instances describe "$name" --zone "$ZONE" --format='value(disks[].source)')"
  disk_uris="${disk_uris//;/ }"
  [[ -n "$disk_uris" ]] || { echo "Could not enumerate disks for $name" >&2; exit 1; }
  for disk_uri in $disk_uris; do
    disk_name="${disk_uri##*/}"
    policies="$(gcloud compute disks describe "$disk_name" --zone "$ZONE" --format='value(resourcePolicies)' || true)"
    [[ -z "$policies" ]] || { echo "Disk $disk_name for $name has resourcePolicies: $policies" >&2; exit 1; }
    log "no disk resource policy: $name / $disk_name"
  done
}

internal_ip() {
  gcloud compute instances describe "$1" --zone "$ZONE" --format='value(networkInterfaces[0].networkIP)'
}

start_coordinator() {
  log "preparing and serving coordinator (Qwen-only)"
  gcloud compute ssh "$COORD" --zone "$ZONE" \
    --command "RUN_ID='$RUN_ID' FRESH='${FRESH:-0}' ENFORCE_EAGER='$ENFORCE_EAGER' bash -s" <<'REMOTE'
set -euo pipefail
cd ~/MultiNet-v2.0
source .venv-multinet/bin/activate

python - <<PY
import json, os
from pathlib import Path
cfg = json.loads(Path("gridworld/fixtures/run_config.smoke_eval_qwen_kimi.json").read_text())
cfg["description"] = "Qwen-only smoke (3 mazes): 2 Qwen3.6 FP8 vLLM workers, no Kimi."
cfg["models"] = {}
cfg["models"]["qwen36_27b_fp8_vllm"] = {
    "provider": "qwen_vllm",
    "model": "Qwen/Qwen3.6-27B-FP8",
    "temperature": 0.0,
    "max_tokens": 4096,
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.88,
    "enforce_eager": os.environ.get("ENFORCE_EAGER", "false").lower() == "true",
    "local_files_only": True,
    "enable_thinking": False,
    "group": "qwen36-27b",
    "hardware_profile": "local-gpu",
    "worker_count": 2,
    "max_in_flight": 2,
    "tasks": ["all"],
}
Path("/tmp/run_config.qwen_only.json").write_text(json.dumps(cfg, indent=2) + "\n")
print("enforce_eager:", cfg["models"]["qwen36_27b_fp8_vllm"]["enforce_eager"])
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
  log "starting Qwen worker: $vm"
  gcloud compute ssh "$vm" --zone "$ZONE" --command "RUN_ID='$RUN_ID' COORD_IP='$coord_ip' bash -s" <<'REMOTE'
set -euo pipefail
cd ~/MultiNet-v2.0
source .venv-qwen-vllm/bin/activate
mkdir -p "$HOME/multinet-worker-artifacts/$RUN_ID"

nohup env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.run_pipeline \
  --distributed-role worker \
  --coordinator-url "http://$COORD_IP:8765" \
  --artifacts-root "$HOME/multinet-worker-artifacts/$RUN_ID" \
  --worker-state "$HOME/multinet-worker-artifacts/$RUN_ID/worker_state.json" \
  --model-group qwen36-27b \
  --hardware-profile local-gpu \
  --local-model-cache Qwen/Qwen3.6-27B-FP8 \
  > "$HOME/multinet-worker-artifacts/$RUN_ID/worker.log" 2>&1 &
echo "$!" > "$HOME/multinet-worker-artifacts/$RUN_ID/worker.pid"
echo "Qwen worker started: $(cat "$HOME/multinet-worker-artifacts/$RUN_ID/worker.pid")"
REMOTE
}

print_summary() {
  local mins
  mins="$(watchdog_minutes "$MAX_RUN_DURATION")"
  cat <<EOF

Launched $RUN_ID (Qwen-only: $COORD + $QWEN1 + $QWEN2).
  GCP floor (Layer 0): STOP after $MAX_RUN_DURATION (data preserved).
  On-VM watchdog (Layer 1): guest shutdown at +${mins}m.
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

  wait_for_ssh "$COORD"
  wait_for_ssh "$QWEN1"
  wait_for_ssh "$QWEN2"

  arm_watchdog "$COORD"
  arm_watchdog "$QWEN1"
  arm_watchdog "$QWEN2"

  assert_no_resource_policy "$COORD"
  assert_no_resource_policy "$QWEN1"
  assert_no_resource_policy "$QWEN2"

  local COORD_IP
  COORD_IP="$(internal_ip "$COORD")"
  log "coordinator internal IP: $COORD_IP"

  start_coordinator
  start_qwen_worker "$QWEN1" "$COORD_IP"
  start_qwen_worker "$QWEN2" "$COORD_IP"

  print_summary
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
