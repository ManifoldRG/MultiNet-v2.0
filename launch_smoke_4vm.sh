#!/usr/bin/env bash
set -euo pipefail

# Shared cost-safety net (single source of truth).
source "$(dirname "${BASH_SOURCE[0]}")/lib/cost_safety.sh"

# Launch the initial 4-VM distributed smoke:
#   1 coordinator, 2 Qwen A100 vLLM workers, 1 Kimi API worker.
#
# Required:
#   export MOONSHOT_API_KEY=...
#   export MAX_RUN_DURATION=...   # GCP-native cost floor; no default (see below)
#
# Optional overrides:
#   ZONE=asia-northeast1-c
#   RUN_ID=smoke4-YYYYMMDD-HHMMSS
#   FRESH=1                       # wipe artifacts/<RUN_ID> instead of resuming
#   COORD=mn-smoke-coord
#   QWEN1=mn-smoke-qwen-1
#   QWEN2=mn-smoke-qwen-2
#   KIMI=mn-smoke-kimi-1
#
# Subcommands (no API key / MAX_RUN_DURATION needed):
#   ./launch_smoke_4vm.sh stop      # spin all 4 VMs down, KEEP disks + run data
#   ./launch_smoke_4vm.sh delete    # delete all 4 VMs incl. disks (post-export only)
#
# Cost safety net (three defense-in-depth layers):
#   Layer 0  GCP-native: every created VM gets --max-run-duration=$MAX_RUN_DURATION
#            --instance-termination-action=STOP (server-side stop, data preserved).
#   Layer 1  On-VM watchdog: each VM schedules `sudo shutdown -h` at the floor + 1h.
#   Layer 2  monitor_run.sh (operator-side, via Claude /loop) finalizes, pulls data,
#            and stops early on completion/stall.

ZONE="${ZONE:-asia-northeast1-c}"
RUN_ID="${RUN_ID:-smoke4-$(date +%Y%m%d-%H%M%S)}"

COORD="${COORD:-mn-smoke-coord}"
QWEN1="${QWEN1:-mn-smoke-qwen-1}"
QWEN2="${QWEN2:-mn-smoke-qwen-2}"
KIMI="${KIMI:-mn-smoke-kimi-1}"

COORD_IMAGE="${COORD_IMAGE:-multinet-coordinator-n2-20260629}"
QWEN_IMAGE="${QWEN_IMAGE:-multinet-qwen36-fp8-vllm-a100-20260629}"
KIMI_IMAGE="${KIMI_IMAGE:-multinet-api-runner-e2-20260629}"

# Tracks whether each VM was created this run (1) or reused (0). Drives the
# watchdog failure policy: a created VM is covered by the GCP floor, a reused one
# is not.
declare -A VM_CREATED

require_moonshot() {
  if [[ -z "${MOONSHOT_API_KEY:-}" ]]; then
    echo "MOONSHOT_API_KEY is required for the Kimi worker." >&2
    echo "Run: export MOONSHOT_API_KEY=..." >&2
    return 1
  fi
}

# --------------------------------------------------------------------------- #
# Cleanup subcommands (no creds needed)
# --------------------------------------------------------------------------- #

cmd_stop() {
  require_gcloud
  log "spinning down (STOP — disks & run data preserved) in $ZONE: $COORD $QWEN1 $QWEN2 $KIMI"
  cs_stop_vms "$ZONE" "$COORD" "$QWEN1" "$QWEN2" "$KIMI"
}

cmd_delete() {
  require_gcloud
  log "DELETING (incl. boot disks / run data) in $ZONE: $COORD $QWEN1 $QWEN2 $KIMI"
  cs_delete_vms "$ZONE" "$COORD" "$QWEN1" "$QWEN2" "$KIMI"
}

# --------------------------------------------------------------------------- #
# Instance lifecycle
# --------------------------------------------------------------------------- #

ensure_instance() {
  local name="$1" image="$2"
  if instance_exists "$name" "$ZONE"; then
    log "instance exists (reused): $name — GCP max-run-duration floor NOT applied; relying on on-VM watchdog"
    VM_CREATED["$name"]=0
    return
  fi
  log "creating $name from $image (max-run-duration=$MAX_RUN_DURATION, termination=STOP)"
  cs_create_instance "$name" "$image" "$ZONE"
  VM_CREATED["$name"]=1
}

start_coordinator() {
  log "preparing and serving coordinator"
  gcloud compute ssh "$COORD" --zone "$ZONE" --command "RUN_ID='$RUN_ID' FRESH='${FRESH:-0}' bash -s" <<'REMOTE'
set -euo pipefail
cd ~/MultiNet-v2.0
source .venv-multinet/bin/activate

python - <<'PY'
import json
from pathlib import Path

cfg = json.loads(Path("gridworld/fixtures/run_config.smoke_eval_qwen_kimi.json").read_text())
cfg["description"] = "4 VM smoke: 2 Qwen3.6 FP16 vLLM workers + 1 Kimi API worker."
cfg["models"].pop("qwen35_27b_hf", None)
cfg["models"]["qwen36_27b_fp8_vllm"] = {
    "provider": "qwen_vllm",
    "model": "Qwen/Qwen3.6-27B",
    "temperature": 0.0,
    "max_tokens": 4096,
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.88,
    "enforce_eager": True,
    "local_files_only": True,
    "enable_thinking": False,
    "group": "qwen36-27b",
    "hardware_profile": "local-gpu",
    "worker_count": 2,
    "max_in_flight": 2,
    "tasks": ["all"],
}
Path("/tmp/run_config.smoke_qwen36_vllm_kimi.json").write_text(json.dumps(cfg, indent=2) + "\n")
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
then
  :
else
  echo "Port 8765 is already in use on the coordinator." >&2
  exit 1
fi

# Only wipe when FRESH=1. Re-running with the same RUN_ID otherwise resumes:
# coordinator-prepare preserves already-verified units, so an unconditional
# wipe here would re-pay for completed work.
if [[ "${FRESH:-0}" == "1" ]]; then
  rm -rf "artifacts/$RUN_ID"
fi
mkdir -p "artifacts/$RUN_ID"

python -m scripts.run_pipeline \
  --distributed-role coordinator-prepare \
  --run-config /tmp/run_config.smoke_qwen36_vllm_kimi.json \
  --manifest gridworld/fixtures/manifest.smoke_eval.json \
  --seeds 0 \
  --artifacts-root "artifacts/$RUN_ID" \
  --run-set-id "$RUN_ID" \
  --difficulty-max-static-score 1000.0

nohup python -m scripts.run_pipeline \
  --distributed-role coordinator-serve \
  --artifacts-root "artifacts/$RUN_ID" \
  --host 0.0.0.0 \
  --port 8765 \
  > "artifacts/$RUN_ID/coordinator-serve.log" 2>&1 &
echo "$!" > "artifacts/$RUN_ID/coordinator-serve.pid"

coord_up=0
for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8765/status; then
    echo
    coord_up=1
    break
  fi
  sleep 2
done
if [[ "$coord_up" -ne 1 ]]; then
  echo "Coordinator did not become healthy on 127.0.0.1:8765 within ~60s." >&2
  exit 1
fi
REMOTE
}

start_qwen_worker() {
  local vm="$1"
  local coord_ip="$2"
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
  --local-model-cache Qwen/Qwen3.6-27B \
  > "$HOME/multinet-worker-artifacts/$RUN_ID/worker.log" 2>&1 &
echo "$!" > "$HOME/multinet-worker-artifacts/$RUN_ID/worker.pid"
echo "Qwen worker started: $(cat "$HOME/multinet-worker-artifacts/$RUN_ID/worker.pid")"
REMOTE
}

start_kimi_worker() {
  local coord_ip="$1"
  local moonshot_q
  printf -v moonshot_q '%q' "$MOONSHOT_API_KEY"
  log "starting Kimi API worker: $KIMI"
  gcloud compute ssh "$KIMI" --zone "$ZONE" --command "RUN_ID='$RUN_ID' COORD_IP='$coord_ip' bash -s" <<REMOTE
set -euo pipefail
export MOONSHOT_API_KEY=${moonshot_q}
cd ~/MultiNet-v2.0
source .venv-multinet/bin/activate
mkdir -p "\$HOME/multinet-worker-artifacts/\$RUN_ID"

nohup python -m scripts.run_pipeline \\
  --distributed-role worker \\
  --coordinator-url "http://\$COORD_IP:8765" \\
  --artifacts-root "\$HOME/multinet-worker-artifacts/\$RUN_ID" \\
  --worker-state "\$HOME/multinet-worker-artifacts/\$RUN_ID/worker_state.json" \\
  --model-group kimi-api \\
  --hardware-profile api-client \\
  > "\$HOME/multinet-worker-artifacts/\$RUN_ID/worker.log" 2>&1 &
echo "\$!" > "\$HOME/multinet-worker-artifacts/\$RUN_ID/worker.pid"
echo "Kimi worker started: \$(cat "\$HOME/multinet-worker-artifacts/\$RUN_ID/worker.pid")"
REMOTE
}

print_summary() {
  local mins
  mins="$(watchdog_minutes "$MAX_RUN_DURATION")"
  cat <<EOF

Launched $RUN_ID.
  GCP floor (Layer 0): stop after $MAX_RUN_DURATION of runtime (data preserved).
  On-VM watchdog (Layer 1): guest shutdown at +${mins}m as backstop.

Monitor:
  gcloud compute ssh $COORD --zone $ZONE --command 'curl -s http://127.0.0.1:8765/status; echo'

Coordinator log:
  gcloud compute ssh $COORD --zone $ZONE --command 'tail -f ~/MultiNet-v2.0/artifacts/$RUN_ID/coordinator-serve.log'

Qwen logs:
  gcloud compute ssh $QWEN1 --zone $ZONE --command 'tail -f ~/multinet-worker-artifacts/$RUN_ID/worker.log'
  gcloud compute ssh $QWEN2 --zone $ZONE --command 'tail -f ~/multinet-worker-artifacts/$RUN_ID/worker.log'

Kimi log:
  gcloud compute ssh $KIMI --zone $ZONE --command 'tail -f ~/multinet-worker-artifacts/$RUN_ID/worker.log'

Hands-off monitor (Layer 2) — finalize + pull data + STOP on completion (run under Claude /loop):
  /loop 12m ./monitor_run.sh --once --coord $COORD --qwen1 $QWEN1 --qwen2 $QWEN2 --kimi $KIMI \\
      --zone $ZONE --run-id $RUN_ID --dest ./artifacts-pulled/$RUN_ID \\
      --stall-minutes 30 --state-file ./.monitor_state.$RUN_ID.json --complete-actions

Manual finalize (if not using the monitor) after /status shows every unit verified
(the unit count is printed by coordinator-prepare above; this smoke should report 6 = 3 mazes x 2 models):
  gcloud compute ssh $COORD --zone $ZONE --command 'cd ~/MultiNet-v2.0 && source .venv-multinet/bin/activate && python -m scripts.run_pipeline --distributed-role coordinator-finalize --artifacts-root artifacts/$RUN_ID --run-set-id $RUN_ID'

Cleanup (data is on the persistent disks until you delete):
  ZONE=$ZONE $0 stop      # spin down, KEEP data
  ZONE=$ZONE $0 delete    # remove VMs + disks, AFTER you have exported results
EOF
}

main() {
  case "${1:-}" in
    stop) cmd_stop; exit 0 ;;
    delete) cmd_delete; exit 0 ;;
    teardown)
      echo "The 'teardown' verb was split: use 'stop' (keep data) or 'delete' (remove disks)." >&2
      echo "  $0 stop    or    $0 delete" >&2
      exit 2 ;;
  esac

  require_max_run_duration
  require_moonshot
  validate_run_id
  require_gcloud

  log "run id: $RUN_ID"
  log "cost floor: GCP stop after $MAX_RUN_DURATION; on-VM watchdog +$(watchdog_minutes "$MAX_RUN_DURATION")m"

  ensure_instance "$COORD" "$COORD_IMAGE"
  ensure_instance "$QWEN1" "$QWEN_IMAGE"
  ensure_instance "$QWEN2" "$QWEN_IMAGE"
  ensure_instance "$KIMI" "$KIMI_IMAGE"

  wait_for_ssh "$COORD" "$ZONE" || exit 1
  wait_for_ssh "$QWEN1" "$ZONE" || exit 1
  wait_for_ssh "$QWEN2" "$ZONE" || exit 1
  wait_for_ssh "$KIMI" "$ZONE" || exit 1

  arm_watchdog "$COORD" "$ZONE" "${VM_CREATED[$COORD]:-0}" || exit 1
  arm_watchdog "$QWEN1" "$ZONE" "${VM_CREATED[$QWEN1]:-0}" || exit 1
  arm_watchdog "$QWEN2" "$ZONE" "${VM_CREATED[$QWEN2]:-0}" || exit 1
  arm_watchdog "$KIMI" "$ZONE" "${VM_CREATED[$KIMI]:-0}" || exit 1

  assert_no_resource_policy "$COORD" "$ZONE" || exit 1
  assert_no_resource_policy "$QWEN1" "$ZONE" || exit 1
  assert_no_resource_policy "$QWEN2" "$ZONE" || exit 1
  assert_no_resource_policy "$KIMI" "$ZONE" || exit 1

  local COORD_IP
  COORD_IP="$(internal_ip "$COORD" "$ZONE")"
  log "coordinator internal IP: $COORD_IP"

  start_coordinator
  start_qwen_worker "$QWEN1" "$COORD_IP"
  start_qwen_worker "$QWEN2" "$COORD_IP"
  start_kimi_worker "$COORD_IP"

  print_summary
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
