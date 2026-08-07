#!/usr/bin/env bash
# sweep_run.sh — thin sequencer over the 9-batch (+ smoke) conditional sweep on a
# SINGLE reused fleet (3 Qwen + 1 Kimi + 1 Claude + coordinator). It composes the
# already-tested primitives (launch_distributed.sh + lib/cost_safety.sh +
# lib/distributed_start.sh) and holds NO cost-safety logic of its own. It STOPs,
# never deletes.
#
# NOTE: -e is intentionally OFF — a tick must survive a nonzero sub-step so the
# fail-closed egress guard can run and the fleet is never left in a half-known state.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/cost_safety.sh"
source "$HERE/lib/distributed_start.sh"

# ---- configuration (env-overridable) --------------------------------------- #
RUNS_DIR="${RUNS_DIR:-.runs}"
SWEEP_ID="${SWEEP_ID:-sweep-$(date +%Y%m%d-%H%M%S)}"
RESULTS_REPO="${RESULTS_REPO:-Multinet-v2-results}"
QWEN_WORKER_COUNT="${QWEN_WORKER_COUNT:-3}"     # GPU fan-out (Task 1 topology override)
DIFFICULTY_MAX="${DIFFICULTY_MAX:-1000}"
# Per-batch on-VM watchdog (fail-closed). No default: the old 6h default
# silently killed a legitimate ~7h run-massive tail before its data was pulled
# (the 134/135 recovery). Batch-starting subcommands require it via
# require_batch_cap; size it to expected batch runtime + margin (e.g. 9h).
BATCH_CAP="${BATCH_CAP:-}"
WORKER_CONCURRENCY="${WORKER_CONCURRENCY:-16}"    # GPU worker: episodes run in parallel (served vLLM batches them)
# The launcher to invoke for provision — an indirection so tests can stub it.
SWEEP_LAUNCHER="${SWEEP_LAUNCHER:-$HERE/launch_distributed.sh}"
# Fleet-topology + first (smoke) run. The fleet shape derives from this config;
# worker-name parity across smoke and all conditional configs is verified in tests.
# SWEEP_TOPO=api provisions an API-only fleet (Kimi+Claude, no A100), so its smoke
# must also drop the qwen model or provision would prepare orphaned qwen units.
if [[ "${SWEEP_TOPO:-}" == "api" ]]; then
  PROVISION_RUN_CONFIG="${PROVISION_RUN_CONFIG:-gridworld/fixtures/run_config.smoke_kimi_claude.json}"
elif [[ "${SWEEP_TOPO:-}" == "kimictx" ]]; then
  PROVISION_RUN_CONFIG="${PROVISION_RUN_CONFIG:-gridworld/fixtures/run_config.smoke_kimi3.json}"
else
  PROVISION_RUN_CONFIG="${PROVISION_RUN_CONFIG:-gridworld/fixtures/run_config.smoke_qwen36_kimi_claude.json}"
fi
PROVISION_MANIFEST="${PROVISION_MANIFEST:-gridworld/fixtures/manifest.smoke_eval.json}"

manifest_path() { echo "$RUNS_DIR/$SWEEP_ID/manifest.json"; }
state_path() { echo "$RUNS_DIR/$SWEEP_ID/sweep_state.json"; }

# ---- manifest readers (the fleet is provisioned once; read its live VMs) ---- #
_require_manifest() {
  local mf; mf="$(manifest_path)"
  [[ -f "$mf" ]] || { echo "[sweep_run] no manifest at $mf — provision first" >&2; return 1; }
}
manifest_zone()  { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["zone"])' "$(manifest_path)"; }
manifest_coord() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["coordinator"]["name"])' "$(manifest_path)"; }
manifest_vm_names() { python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
print(d["coordinator"]["name"], *[w["name"] for w in d.get("workers",[])])' "$(manifest_path)"; }
manifest_worker_names() { python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
print(*[w["name"] for w in d.get("workers",[])])' "$(manifest_path)"; }
manifest_api_vms() { python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
print(*[w["name"] for w in d.get("workers",[]) if w.get("kind")=="api"])' "$(manifest_path)"; }
manifest_gpu_vms() { python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
print(*[w["name"] for w in d.get("workers",[]) if w.get("kind")=="gpu"])' "$(manifest_path)"; }

# ---- batch field resolver (single source of truth = scripts.sweep_state) ---- #
batch_field() {  # $1 n  $2 field  — prints "" for a JSON null
  python3 -c '
import sys
from scripts.sweep_state import BATCHES
b = next(b for b in BATCHES if b["n"] == int(sys.argv[1]))
v = b[sys.argv[2]]
print("" if v is None else v)' "$1" "$2"
}

# Coordinator artifacts namespace for batch N. Batch 0 (smoke) is started by
# `provision` under RUN_ID=$SWEEP_ID (the single-RUN_ID launcher contract), so its
# artifacts land at artifacts/$SWEEP_ID; batches 1..9 run under their own run_id.
# Keeping this in one place is what makes finalize-batch 0 read the right path.
_artifacts_run_id() {  # $1 n
  if [[ "$1" == "0" ]]; then echo "$SWEEP_ID"; else batch_field "$1" run_id; fi
}

# ---- state helpers (thin python wrappers over scripts.sweep_state) ---------- #
_state_mark() {  # $1 n  $2 status  [$3 extra_json_kv...]  — updates + recomputes ETAs
  local n="$1" status="$2"; shift 2
  python3 - "$(state_path)" "$n" "$status" "$@" <<'PY'
import sys, datetime
from scripts.sweep_state import load_state, update_batch, recompute_etas, save_state
p, n, status = sys.argv[1], int(sys.argv[2]), sys.argv[3]
extra = {}
for kv in sys.argv[4:]:
    k, _, v = kv.partition("=")
    extra[k] = v
try:
    st = load_state(p)
except FileNotFoundError:
    sys.exit(0)  # no state yet (e.g. a test path) — nothing to record
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
if status == "running":
    extra.setdefault("started_at", now)
elif status == "complete":
    extra.setdefault("ended_at", now)
update_batch(st, n, status=status, **extra)
st["current_batch"] = n
st["phase"] = status
recompute_etas(st)
save_state(p, st)
PY
}

# =================== subcommands ============================================= #

# provision: bring the fleet UP running batch-0 (smoke) via the tested launcher,
# then initialise sweep tracking. Does NOT start any conditional batch.
cmd_provision() {
  : "${MAX_RUN_DURATION:?MAX_RUN_DURATION is required (GCP provision ceiling, e.g. 120h)}"
  # Export the knobs the launcher (and its start hooks) read from the env, so the
  # multi-zone hunt order (ZONES, NE-Asia first) and difficulty are not silently lost.
  export QWEN_WORKER_COUNT DIFFICULTY_MAX
  [[ -n "${ZONES:-}" ]] && export ZONES
  [[ -n "${ZONE:-}" ]] && export ZONE
  local created; created="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  log "provision: fleet=$SWEEP_ID topology+smoke from $PROVISION_RUN_CONFIG (QWEN_WORKER_COUNT=$QWEN_WORKER_COUNT)"
  RUN_ID="$SWEEP_ID" RUNS_DIR="$RUNS_DIR" \
    RUN_CONFIG="$PROVISION_RUN_CONFIG" MANIFEST="$PROVISION_MANIFEST" \
    "$SWEEP_LAUNCHER" \
    || { echo "[sweep_run] provision (launcher) failed — fleet not up" >&2; return 1; }
  python3 - "$SWEEP_ID" "$created" "$(state_path)" <<'PY'
import sys
from scripts.sweep_state import init_state, save_state
save_state(sys.argv[3], init_state(sys.argv[1], sys.argv[2]))
PY
  # batch 0 (smoke) is what the launcher just started.
  _state_mark 0 running
  log "provision COMPLETE: fleet up, batch 0 (smoke) running; conditional batches NOT started. state=$(state_path)"
}

# next-batch N: on the ALREADY-RUNNING fleet, stop the prior batch's processes,
# re-prepare the coordinator + restart workers for batch N, re-arm the watchdog.
# Never creates or deletes VMs.
require_batch_cap() {
  if [[ -z "${BATCH_CAP:-}" ]]; then
    echo "BATCH_CAP is required for batch-starting subcommands (per-batch watchdog," >&2
    echo "e.g. BATCH_CAP=9h — expected batch runtime + margin). No default: a wrong" >&2
    echo "default either kills a legit run mid-tail or fails to protect." >&2
    return 1
  fi
}

cmd_next_batch() {
  local n="$1"
  _require_manifest || return 1
  require_gcloud || return 1
  require_batch_cap || return 1
  local name run_id art_id run_config manifest conditions prompt_variant zone coord
  name="$(batch_field "$n" name)"; run_id="$(batch_field "$n" run_id)"
  art_id="$(_artifacts_run_id "$n")"   # artifacts namespace (batch 0 -> $SWEEP_ID)
  run_config="$(batch_field "$n" run_config)"; manifest="$(batch_field "$n" manifest)"
  conditions="$(batch_field "$n" conditions)"; prompt_variant="$(batch_field "$n" prompt_variant)"
  zone="$(manifest_zone)"; coord="$(manifest_coord)"

  # (re)START the e2 API runners (stop-apis STOPs them during a batch's Qwen tail).
  local api_vms vm; api_vms="$(manifest_api_vms)"
  if [[ -n "$api_vms" ]]; then
    # shellcheck disable=SC2086
    gcloud compute instances start $api_vms --zone "$zone" --quiet >/dev/null 2>&1 || true
    for vm in $api_vms; do
      wait_for_ssh "$vm" "$zone" || { echo "[sweep_run] ssh wait failed on restarted api VM $vm" >&2; return 1; }
    done
  fi

  # Re-arm the on-VM watchdog with the per-batch cap BEFORE the (fallible) restart,
  # so a partial next-batch still leaves the fleet capped at BATCH_CAP rather than a
  # stale prior arm. created=0 => FAIL-CLOSED on a reused VM (no fresh-VM GCP floor).
  for vm in $(manifest_vm_names); do
    MAX_RUN_DURATION="$BATCH_CAP" arm_watchdog "$vm" "$zone" 0 \
      || { echo "[sweep_run] watchdog re-arm failed on $vm (batch $n)" >&2; return 1; }
  done

  # Derive TOPO_JSON keyed to SWEEP_ID so worker names match the provisioned VMs;
  # the batch's run_config supplies each worker's provider/model to the start hooks
  # (start_coordinator / start_worker) via worker_field, which reads TOPO_JSON.
  export QWEN_WORKER_COUNT
  local topo coord_ip
  topo="$(python3 -m scripts.distributed_topology "$run_config" "$SWEEP_ID")" \
    || { echo "[sweep_run] topology derive failed for $run_config" >&2; return 1; }
  coord_ip="$(internal_ip "$coord" "$zone")" \
    || { echo "[sweep_run] internal_ip failed for $coord" >&2; return 1; }

  # Globals consumed by the start hooks (lib/distributed_start.sh) and worker_field.
  # RUN_ID is the artifacts namespace (art_id): batch 0 -> $SWEEP_ID.
  export COORD="$coord" ZONE="$zone" RUN_ID="$art_id" RUN_CONFIG="$run_config" \
         MANIFEST="$manifest" CONDITIONS="$conditions" PROMPT_VARIANT="$prompt_variant" \
         DIFFICULTY_MAX="$DIFFICULTY_MAX" TOPO_JSON="$topo"

  # Free coordinator port 8765 + stop the prior batch's WORKER processes so the fresh
  # start hooks bind cleanly. GPU workers keep their PERSISTENT vLLM server running
  # (it holds the GPU intentionally and is reused across batches — no model reload, and
  # no orphan-EngineCore to clean), so only the worker process is killed here, same as
  # the API workers. (lib/gpu_teardown.sh::stop_gpu_worker remains available to fully
  # free a GPU if a server ever needs a clean restart.)
  gcloud compute ssh "$coord" --zone "$zone" --command \
    "pkill -f 'distributed-role coordinator-serve' 2>/dev/null; sleep 2; pkill -9 -f 'distributed-role coordinator-serve' 2>/dev/null; true" >/dev/null 2>&1 || true
  for vm in $(manifest_worker_names); do
    gcloud compute ssh "$vm" --zone "$zone" --command \
      "pkill -f 'distributed-role worker' 2>/dev/null; true" >/dev/null 2>&1 || true
  done
  start_coordinator || { echo "[sweep_run] coordinator prepare/serve failed for batch $n ($name)" >&2; return 1; }
  # Start workers in PARALLEL: a GPU worker's start hook blocks until its vLLM server
  # has loaded the model, so a serial loop would load the 3 servers back-to-back
  # (~14min each). Backgrounding lets all 3 load at once. (Reused servers return fast.)
  local wpids=() wrc=0
  for vm in $(manifest_worker_names); do
    start_worker "$vm" "$coord_ip" & wpids+=("$!")
  done
  for wpid in "${wpids[@]}"; do wait "$wpid" || wrc=1; done
  [[ $wrc -eq 0 ]] || { echo "[sweep_run] one or more workers failed to start for batch $n ($name)" >&2; return 1; }

  _state_mark "$n" running
  log "batch $n ($name -> run_id=$art_id) started on the reused fleet; watchdog re-armed @ $BATCH_CAP"
}

# run-massive N [N ...]: prepare+serve ONE combined job from the given batches (all their
# units in a single LPT-ordered plan) so the fleet stays saturated instead of idling
# through each batch's long hard-maze tail. Reuses the persistent GPU servers.
# RUN_ID=cond_massive; finalize with finalize-batch on that run_id, then split per config.
cmd_run_massive() {
  _require_manifest || return 1
  require_gcloud || return 1
  require_batch_cap || return 1
  [[ "$#" -ge 1 ]] || { echo "usage: sweep_run.sh run-massive N [N ...]" >&2; return 2; }
  local run_id="cond_massive" first="$1" run_config manifest zone coord vm
  run_config="$(batch_field "$first" run_config)"   # topo identical across qwen batches
  manifest="$(batch_field "$first" manifest)"
  zone="$(manifest_zone)"; coord="$(manifest_coord)"

  for vm in $(manifest_vm_names); do
    MAX_RUN_DURATION="$BATCH_CAP" arm_watchdog "$vm" "$zone" 0 \
      || { echo "[sweep_run] watchdog re-arm failed on $vm (run-massive)" >&2; return 1; }
  done

  export QWEN_WORKER_COUNT
  local topo coord_ip
  topo="$(python3 -m scripts.distributed_topology "$run_config" "$SWEEP_ID")" \
    || { echo "[sweep_run] topology derive failed for $run_config" >&2; return 1; }
  coord_ip="$(internal_ip "$coord" "$zone")" \
    || { echo "[sweep_run] internal_ip failed for $coord" >&2; return 1; }
  export COORD="$coord" ZONE="$zone" RUN_ID="$run_id" MANIFEST="$manifest" \
         DIFFICULTY_MAX="$DIFFICULTY_MAX" TOPO_JSON="$topo" \
         MASSIVE_BATCHES="$*" SEEDS="${SEEDS:-0}"

  gcloud compute ssh "$coord" --zone "$zone" --command \
    "pkill -f 'distributed-role coordinator-serve' 2>/dev/null; sleep 2; pkill -9 -f 'distributed-role coordinator-serve' 2>/dev/null; true" >/dev/null 2>&1 || true
  for vm in $(manifest_worker_names); do
    gcloud compute ssh "$vm" --zone "$zone" --command \
      "pkill -f 'distributed-role worker' 2>/dev/null; true" >/dev/null 2>&1 || true
  done

  start_coordinator_combined || { echo "[sweep_run] combined coordinator prepare/serve failed" >&2; return 1; }
  local wpids=() wrc=0
  for vm in $(manifest_worker_names); do
    start_worker "$vm" "$coord_ip" & wpids+=("$!")
  done
  for wpid in "${wpids[@]}"; do wait "$wpid" || wrc=1; done
  [[ $wrc -eq 0 ]] || { echo "[sweep_run] one or more workers failed to start (run-massive)" >&2; return 1; }
  log "MASSIVE job (batches $*) started as run_id=$run_id on the reused fleet; watchdog re-armed @ $BATCH_CAP"
}

# reload-qwen-phase2: two-tier PHASE-2 server reload on the ALREADY-RUNNING fleet.
# For each GPU VM: stop_gpu_worker (fail-closed GPU free) then relaunch the vLLM
# server with phase-2 serve args (max-model-len 96000, few seqs) + reduced worker
# concurrency, and health-check /v1/models. Reuses reload_gpu_worker + start_worker
# (no new SSH mechanics). Requires BATCH_CAP (cost-safety parity with the other
# batch-starting subcommands). RUN_CONFIG = the phase-2 qwen run config (derives the
# worker topology); RUN_ID = phase-2 artifacts namespace (default $SWEEP_ID).
# See docs/qwen-two-tier-rerun-design.md.
cmd_reload_qwen_phase2() {
  _require_manifest || return 1
  require_gcloud || return 1
  require_batch_cap || return 1
  : "${RUN_CONFIG:?RUN_CONFIG is required (phase-2 qwen run config; derives worker topology)}"
  local zone coord coord_ip gpu_vms vm topo art_id
  zone="$(manifest_zone)"; coord="$(manifest_coord)"
  gpu_vms="$(manifest_gpu_vms)"
  [[ -n "$gpu_vms" ]] || { log "reload-qwen-phase2: no gpu-kind VMs in manifest — nothing to reload"; return 0; }
  art_id="${RUN_ID:-$SWEEP_ID}"

  export QWEN_WORKER_COUNT
  topo="$(python3 -m scripts.distributed_topology "$RUN_CONFIG" "$SWEEP_ID")" \
    || { echo "[sweep_run] topology derive failed for $RUN_CONFIG" >&2; return 1; }
  coord_ip="$(internal_ip "$coord" "$zone")" \
    || { echo "[sweep_run] internal_ip failed for $coord" >&2; return 1; }

  # Re-arm the on-VM watchdog with the per-batch cap (fail-closed on reused VM),
  # mirroring next-batch/run-massive.
  for vm in $(manifest_vm_names); do
    MAX_RUN_DURATION="$BATCH_CAP" arm_watchdog "$vm" "$zone" 0 \
      || { echo "[sweep_run] watchdog re-arm failed on $vm (reload-qwen-phase2)" >&2; return 1; }
  done

  # Globals consumed by reload_gpu_worker -> start_worker (worker_field reads TOPO_JSON).
  # WORKER_CONCURRENCY is dropped to the phase-2 (narrow) fan-out; the phase-2 serve
  # knobs (max-model-len 96000, few seqs) are set inside reload_gpu_worker.
  export COORD="$coord" ZONE="$zone" RUN_ID="$art_id" TOPO_JSON="$topo" \
         WORKER_CONCURRENCY="${QWEN_PHASE2_WORKER_CONCURRENCY:-3}"

  for vm in $gpu_vms; do
    log "reload-qwen-phase2: reloading $vm to phase-2 serve args (max-model-len 96000)"
    reload_gpu_worker "$vm" "$coord_ip" \
      || { echo "[sweep_run] reload_gpu_worker failed on $vm" >&2; return 1; }
    # Explicit post-reload health-check (start_worker already blocks on readiness;
    # this confirms the phase-2 server answers /v1/models from the driver side).
    gcloud compute ssh "$vm" --zone "$zone" --command \
      "curl -fsS http://127.0.0.1:8000/v1/models >/dev/null" >/dev/null 2>&1 \
      || { echo "[sweep_run] $vm: /v1/models not healthy after reload" >&2; return 1; }
    log "reload-qwen-phase2: $vm healthy on phase-2 serve args"
  done
  log "reload-qwen-phase2 COMPLETE: [$gpu_vms] reloaded to phase-2 (max-model-len 96000); watchdog re-armed @ $BATCH_CAP"
}

# finalize-batch N: aggregate on the coordinator, egress the batch's artifacts to
# DEST/<run_id>, and FAIL-CLOSED if nothing lands (fleet left up, no advance).
cmd_finalize_batch() {
  local n="$1"
  : "${DEST:?DEST is required for egress}"
  _require_manifest || return 1
  require_gcloud || return 1
  local run_id art_id name zone coord dest
  run_id="$(batch_field "$n" run_id)"; name="$(batch_field "$n" name)"
  art_id="$(_artifacts_run_id "$n")"   # coordinator artifacts namespace (batch 0 -> $SWEEP_ID)
  zone="$(manifest_zone)"; coord="$(manifest_coord)"
  dest="$DEST/$run_id"

  gcloud compute ssh "$coord" --zone "$zone" --command \
    "cd ~/MultiNet-v2.0 && source .venv-multinet/bin/activate && python -m scripts.run_pipeline --distributed-role coordinator-finalize --artifacts-root artifacts/$art_id --run-set-id $art_id" \
    >/dev/null 2>&1 \
    || echo "[sweep_run] coordinator-finalize returned nonzero on batch $n ($name); continuing to egress" >&2

  mkdir -p "$dest"
  # Egress via ONE tarball: gcloud scp --recurse is unusably slow per-file for a
  # batch's thousands of decision-frame PNGs (8k+ files/batch). Exclude *.png
  # (analysis needs only episode_runs.jsonl/run jsons; publish strips PNGs anyway)
  # and ship a single compressed archive.
  local tgz="/tmp/egress_${art_id//\//_}.tgz"
  gcloud compute ssh "$coord" --zone "$zone" --command \
    "tar czf '$tgz' -C ~/MultiNet-v2.0/artifacts/$art_id --exclude='*.png' ." >/dev/null 2>&1 || true
  gcloud compute scp --zone "$zone" "$coord:$tgz" "$dest/egress.tgz" >/dev/null 2>&1 || true
  [[ -f "$dest/egress.tgz" ]] && { tar xzf "$dest/egress.tgz" -C "$dest" 2>/dev/null && rm -f "$dest/egress.tgz"; }
  if [[ -z "$(ls -A "$dest" 2>/dev/null)" ]]; then
    echo "[sweep_run] egress landed nothing at $dest — FAIL-CLOSED (fleet left up, not advancing)" >&2
    return 40
  fi
  _state_mark "$n" complete "egress=$dest"
  log "batch $n ($name -> $run_id) egress verified at $dest"
}

# stop-apis: STOP only the e2 API runners (they idle through Qwen's tail). The
# coordinator (n2) and Qwen (A100) VMs are never in this set. next-batch STARTs them.
cmd_stop_apis() {
  _require_manifest || return 1
  require_gcloud || return 1
  local zone api_vms; zone="$(manifest_zone)"; api_vms="$(manifest_api_vms)"
  [[ -n "$api_vms" ]] || { log "stop-apis: no api-kind VMs in manifest"; return 0; }
  log "stop-apis: STOP e2 API runners in $zone: $api_vms"
  # shellcheck disable=SC2086
  cs_stop_vms "$zone" $api_vms
}

# publish <run_id>: mirror the egressed batch into the results repo (no PNGs),
# co-locate its summaries, and commit+push. Runs git ONLY in $RESULTS_REPO.
cmd_publish() {
  local run_id="$1"
  : "${DEST:?DEST is required to locate the egressed batch}"
  local src="$DEST/$run_id" dstdir="$RESULTS_REPO/$SWEEP_ID/$run_id"
  [[ -d "$src" ]] || { echo "[sweep_run] nothing to publish at $src" >&2; return 1; }
  mkdir -p "$dstdir"
  rsync -a --exclude='*.png' "$src/" "$dstdir/"
  if compgen -G "artifacts/summaries/${run_id}__*.md" >/dev/null 2>&1; then
    mkdir -p "$dstdir/summaries"
    cp artifacts/summaries/"${run_id}"__*.md "$dstdir/summaries/" 2>/dev/null || true
  fi
  (
    cd "$RESULTS_REPO" || exit 1
    git add -A || exit 1
    # Nothing new (idempotent re-publish, or a retry after the commit already
    # landed but the push failed) is success, not a failure to remediate.
    if git diff --cached --quiet; then echo "[sweep_run] publish: nothing new for $run_id" >&2; exit 0; fi
    git commit -m "results: $SWEEP_ID $run_id" || exit 1
    if ! { git pull --rebase origin main && git push origin main; }; then
      git pull --rebase origin main && git push origin main   # retry once on non-fast-forward
    fi
  ) || { echo "[sweep_run] publish git flow failed for $run_id" >&2; return 1; }
  log "published $run_id -> $dstdir (committed+pushed $RESULTS_REPO)"
}

# teardown: STOP every VM in the manifest (preserve disks). NEVER deletes.
cmd_teardown() {
  _require_manifest || return 1
  require_gcloud || return 1
  local zone names; zone="$(manifest_zone)"; names="$(manifest_vm_names)"
  log "TEARDOWN (STOP, preserve disks) in $zone: $names"
  # shellcheck disable=SC2086
  cs_stop_vms "$zone" $names
}

cmd_status() {
  python3 -c '
import sys
from scripts.sweep_state import load_state, render_table
print(render_table(load_state(sys.argv[1])))' "$(state_path)"
}

main() {
  case "${1:-}" in
    provision)       shift; cmd_provision "$@" ;;
    next-batch)      shift; cmd_next_batch "$@" ;;
    run-massive)     shift; cmd_run_massive "$@" ;;
    reload-qwen-phase2) shift; cmd_reload_qwen_phase2 "$@" ;;
    finalize-batch)  shift; cmd_finalize_batch "$@" ;;
    stop-apis)       shift; cmd_stop_apis "$@" ;;
    publish)         shift; cmd_publish "$@" ;;
    teardown)        shift; cmd_teardown "$@" ;;
    status)          shift; cmd_status "$@" ;;
    *) echo "usage: sweep_run.sh {provision|next-batch N|run-massive N...|reload-qwen-phase2|finalize-batch N|stop-apis|publish RUN_ID|status|teardown}" >&2; return 2 ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
