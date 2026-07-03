#!/usr/bin/env bash
# Distributed-run START recipes. SOURCE this file; do not execute it.
# Provides start_coordinator + start_worker, generalized from launch_qwen_smoke.sh
# (coordinator prepare/serve; GPU worker) and launch_smoke_4vm.sh (API worker + key
# delivery). Consumed by launch_distributed.sh (replaces its stub start hooks).
# See docs/superpowers/specs/2026-07-01-distributed-start-hooks-design.md

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

start_worker() {  # $1 vm  $2 coord_ip  — metadata from TOPO_JSON
  local vm="$1" coord_ip="$2" kind group provider model
  kind="$(worker_field "$vm" kind)"
  group="$(worker_field "$vm" model_group)"
  provider="$(worker_field "$vm" provider)"
  model="$(worker_field "$vm" model)"

  if [[ "$kind" == "gpu" ]]; then
    gcloud compute ssh "$vm" --zone "$ZONE" \
      --command "RUN_ID='$RUN_ID' COORD_IP='$coord_ip' GROUP='$group' MODEL='$model' bash -s" <<'REMOTE'
set -euo pipefail
cd ~/MultiNet-v2.0
source .venv-qwen-vllm/bin/activate
mkdir -p "$HOME/multinet-worker-artifacts/$RUN_ID"
nohup env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.run_pipeline \
  --distributed-role worker \
  --coordinator-url "http://$COORD_IP:8765" \
  --artifacts-root "$HOME/multinet-worker-artifacts/$RUN_ID" \
  --worker-state "$HOME/multinet-worker-artifacts/$RUN_ID/worker_state.json" \
  --model-group "$GROUP" \
  --hardware-profile local-gpu \
  --local-model-cache "$MODEL" \
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
  gcloud compute ssh "$vm" --zone "$ZONE" \
    --command "RUN_ID='$RUN_ID' COORD_IP='$coord_ip' GROUP='$group' bash -s" <<REMOTE
set -euo pipefail
export ${key_var}=${key_q}
cd ~/MultiNet-v2.0
source .venv-multinet/bin/activate
mkdir -p "\$HOME/multinet-worker-artifacts/\$RUN_ID"
nohup python -m scripts.run_pipeline \\
  --distributed-role worker \\
  --coordinator-url "http://\$COORD_IP:8765" \\
  --artifacts-root "\$HOME/multinet-worker-artifacts/\$RUN_ID" \\
  --worker-state "\$HOME/multinet-worker-artifacts/\$RUN_ID/worker_state.json" \\
  --model-group "\$GROUP" \\
  --hardware-profile api-client \\
  > "\$HOME/multinet-worker-artifacts/\$RUN_ID/worker.log" 2>&1 &
echo "\$!" > "\$HOME/multinet-worker-artifacts/\$RUN_ID/worker.pid"
REMOTE
}
