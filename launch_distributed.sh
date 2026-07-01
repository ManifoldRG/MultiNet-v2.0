#!/usr/bin/env bash
set -euo pipefail

# Generic run_config-driven distributed provisioner. Derives VM topology from a
# run_config, finds A100 capacity across zones, verifies on-VM code matches the
# local committed sha, applies the cost-safety net, starts the fleet, and writes
# .runs/<run_id>/manifest.json. See
# docs/superpowers/specs/2026-06-30-distributed-provisioner-design.md
#
# Required: RUN_CONFIG, MANIFEST, MAX_RUN_DURATION.
# Subcommands (no creds / no MAX_RUN_DURATION): stop | delete (operate on the manifest).

source "$(dirname "${BASH_SOURCE[0]}")/lib/cost_safety.sh"

RUN_ID="${RUN_ID:-dist-$(date +%Y%m%d-%H%M%S)}"
ZONE="${ZONE:-us-central1-c}"          # default/coordinator zone; the hunt may override
RUNS_DIR="${RUNS_DIR:-.runs}"
COORD_IMAGE="${COORD_IMAGE:-multinet-coordinator-n2-20260629}"
QWEN_IMAGE="${QWEN_IMAGE:-qwen-fp16-80}"          # FP16 on A100-80GB (a2-ultragpu-1g)
API_IMAGE="${API_IMAGE:-multinet-api-runner-e2-20260629}"
# a2-ultragpu-1g (A100-80GB) zones, us-central1 first per the FP16 migration.
ZONES="${ZONES:-us-central1-a us-central1-b us-central1-c us-central1-f us-east1-b us-east4-c europe-west4-a europe-west4-b asia-southeast1-b asia-southeast1-c asia-northeast1-a asia-northeast1-c me-west1-b me-west1-c}"

manifest_path() { echo "${RUNS_DIR}/${RUN_ID}/manifest.json"; }

# Read VM names (coordinator + workers) from an existing manifest, space-separated.
manifest_vm_names() {
  python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
print(d["coordinator"]["name"], *[w["name"] for w in d["workers"]])
' "$(manifest_path)"
}

manifest_zone() {
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["zone"])' "$(manifest_path)"
}

cmd_stop() {
  require_gcloud
  local mf; mf="$(manifest_path)"
  [[ -f "$mf" ]] || { echo "no manifest at $mf" >&2; return 1; }
  local zone; zone="$(manifest_zone)"
  log "STOP (preserve disks/data) in $zone: $(manifest_vm_names)"
  # shellcheck disable=SC2046
  cs_stop_vms "$zone" $(manifest_vm_names)
}

cmd_delete() {
  require_gcloud
  local mf; mf="$(manifest_path)"
  [[ -f "$mf" ]] || { echo "no manifest at $mf" >&2; return 1; }
  local zone; zone="$(manifest_zone)"
  log "DELETE (incl. disks/data) in $zone: $(manifest_vm_names)"
  # shellcheck disable=SC2046
  cs_delete_vms "$zone" $(manifest_vm_names)
}

# Create one GPU/coordinator VM with the cost-safety floor. $1 name $2 image $3 zone.
_create_in_zone() { cs_create_instance "$1" "$2" "$3"; }

# Delete any of the given VMs in a zone — safe ONLY pre-data (empty fresh VMs).
rollback_zone() {  # $1 zone  $2.. vms
  local zone="$1"; shift
  log "rolling back partial creation in $zone"
  cs_delete_vms "$zone" "$@"
}

# Try to create all GPU VMs (scarce, first) then the coordinator in one zone.
# $1 zone  $2 coord  $3.. gpu_vms. Returns 0 on full success, 1 (after rollback) otherwise.
try_zone() {  # $1 zone  $2 coord  $3.. gpu_vms
  local zone="$1" coord="$2"; shift 2
  local gpu_vms=("$@") vm
  for vm in "${gpu_vms[@]}"; do
    if ! _create_in_zone "$vm" "$QWEN_IMAGE" "$zone"; then
      rollback_zone "$zone" "${gpu_vms[@]}" "$coord"; return 1
    fi
  done
  if ! _create_in_zone "$coord" "$COORD_IMAGE" "$zone"; then
    rollback_zone "$zone" "${gpu_vms[@]}" "$coord"; return 1
  fi
  return 0
}

# Iterate $ZONES; first zone fitting all GPU VMs + coordinator wins. Echoes the
# winning zone on stdout (last line) and returns 0; returns 1 if all exhausted.
hunt_zones() {  # $1 coord  $2.. gpu_vms
  local coord="$1"; shift
  local gpu_vms=("$@") z
  for z in $ZONES; do
    log "=== attempting zone $z ==="
    if try_zone "$z" "$coord" "${gpu_vms[@]}"; then
      log "landed GPU fleet in $z"
      echo "$z"
      return 0
    fi
    log "zone $z unavailable (A100 stockout/quota); next"
  done
  echo "ALL ZONES STOCKED OUT — no A100 capacity. Nothing left running." >&2
  return 1
}

require_clean_tree() {
  if [[ "${ALLOW_DIRTY:-0}" == "1" ]]; then
    log "ALLOW_DIRTY=1: skipping clean-tree gate (dev only)"; return 0
  fi
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Working tree is dirty. A paid run must be on a committed sha." >&2
    echo "Commit/stash, or set ALLOW_DIRTY=1 for a dev run." >&2
    return 1
  fi
  return 0
}

# Push the exact tracked tree at $sha onto one VM and write a sha sentinel.
sync_code_to_vm() {  # $1 sha  $2 zone  $3 vm
  local sha="$1" zone="$2" vm="$3"
  git archive --format=tar "$sha" \
    | gcloud compute ssh "$vm" --zone "$zone" --command \
        "tar -x -C ~/MultiNet-v2.0 && echo \"$sha\" > ~/MultiNet-v2.0/.deployed_sha"
}

# Verify the on-VM code matches $sha: sentinel + content spot-check. Returns 1 on mismatch.
verify_code_on_vm() {  # $1 sha  $2 zone  $3 vm
  local sha="$1" zone="$2" vm="$3" got expected_hash got_hash
  if ! got="$(gcloud compute ssh "$vm" --zone "$zone" --command "cat ~/MultiNet-v2.0/.deployed_sha" 2>/dev/null)"; then
    echo "code-sync: could not read .deployed_sha from $vm (ssh/connection or missing file)" >&2; return 1
  fi
  if [[ "$got" != "$sha" ]]; then
    echo "code-sync mismatch on $vm: deployed_sha='$got' expected='$sha'" >&2; return 1
  fi
  expected_hash="$(git show "$sha:scripts/distributed_run_pipeline.py" | sha256sum | awk '{print $1}')"
  got_hash="$(gcloud compute ssh "$vm" --zone "$zone" --command \
    "sha256sum ~/MultiNet-v2.0/scripts/distributed_run_pipeline.py" 2>/dev/null | awk '{print $1}')"
  if [[ "$expected_hash" != "$got_hash" ]]; then
    echo "code-sync content mismatch on $vm: distributed_run_pipeline.py hash differs" >&2; return 1
  fi
  log "code verified on $vm @ $sha"
  return 0
}

sync_and_verify() {  # $1 sha  $2 zone  $3.. vms
  local sha="$1" zone="$2"; shift 2
  local vm
  for vm in "$@"; do
    if ! sync_code_to_vm "$sha" "$zone" "$vm"; then
      echo "code-sync push failed on $vm" >&2; return 1
    fi
    verify_code_on_vm "$sha" "$zone" "$vm" || return 1
  done
  return 0
}

main() {
  case "${1:-}" in
    stop) cmd_stop; exit 0 ;;
    delete) cmd_delete; exit 0 ;;
  esac

  : "${RUN_CONFIG:?RUN_CONFIG is required (path to a run_config JSON)}"
  : "${MANIFEST:?MANIFEST is required (path to a task manifest JSON)}"
  require_max_run_duration
  validate_run_id
  require_gcloud
  python3 -m scripts.distributed_topology "$RUN_CONFIG" "$RUN_ID" --check-credentials >/dev/null

  log "run id: $RUN_ID  floor: $MAX_RUN_DURATION  watchdog +$(watchdog_minutes "$MAX_RUN_DURATION")m"
  # Provisioning sequence (hunt → code-sync → start → manifest) is added in Tasks 6–8.
  echo "[launch_distributed] skeleton: provisioning not yet implemented" >&2
  return 0
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
