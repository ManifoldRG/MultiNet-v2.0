#!/usr/bin/env bash
# Distributed-run START recipes. SOURCE this file; do not execute it.
# Provides start_coordinator + start_worker, generalized from launch_qwen_smoke.sh
# (coordinator prepare/serve; GPU worker) and launch_smoke_4vm.sh (API worker + key
# delivery). Consumed by launch_distributed.sh (replaces its stub start hooks).
# See docs/superpowers/specs/2026-07-01-distributed-start-hooks-design.md

start_coordinator() {  # uses globals COORD ZONE RUN_ID RUN_CONFIG MANIFEST [SEEDS] [DIFFICULTY_MAX]
  local seeds="${SEEDS:-0}" diff="${DIFFICULTY_MAX:-1000.0}"
  gcloud compute ssh "$COORD" --zone "$ZONE" \
    --command "RUN_ID='$RUN_ID' RUN_CONFIG='$RUN_CONFIG' MANIFEST='$MANIFEST' SEEDS='$seeds' DIFFICULTY_MAX='$diff' bash -s" <<'REMOTE'
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
python -m scripts.run_pipeline \
  --distributed-role coordinator-prepare \
  --run-config "$RUN_CONFIG" \
  --manifest "$MANIFEST" \
  --seeds "$SEEDS" \
  --artifacts-root "artifacts/$RUN_ID" \
  --run-set-id "$RUN_ID" \
  --difficulty-max-static-score "$DIFFICULTY_MAX"
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
