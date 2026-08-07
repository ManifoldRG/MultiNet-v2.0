#!/usr/bin/env bash
set -euo pipefail

# Generic run_config-driven distributed provisioner. Derives VM topology from a
# run_config, finds A100 capacity across zones, verifies on-VM code matches the
# local committed sha, applies the cost-safety net, starts the fleet, and writes
# .runs/<run_id>/manifest.json.
#
# Required: RUN_CONFIG, MANIFEST, MAX_RUN_DURATION.
# Subcommands (no creds / no MAX_RUN_DURATION): stop | delete (operate on the manifest).

source "$(dirname "${BASH_SOURCE[0]}")/lib/cost_safety.sh"
source "$(dirname "${BASH_SOURCE[0]}")/lib/distributed_start.sh"

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
  local zone names
  zone="$(manifest_zone)"; names="$(manifest_vm_names)"
  log "STOP (preserve disks/data) in $zone: $names"
  # shellcheck disable=SC2086
  cs_stop_vms "$zone" $names
}

cmd_delete() {
  require_gcloud
  local mf; mf="$(manifest_path)"
  [[ -f "$mf" ]] || { echo "no manifest at $mf" >&2; return 1; }
  local zone names
  zone="$(manifest_zone)"; names="$(manifest_vm_names)"
  log "DELETE (incl. disks/data) in $zone: $names"
  # shellcheck disable=SC2086
  cs_delete_vms "$zone" $names
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
    if try_zone "$z" "$coord" "${gpu_vms[@]}" >&2; then
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
  # `git archive` skips submodule contents, and the VM can't fetch them itself
  # (the image leaves ogbench uninitialized; its remote needs creds the VM
  # lacks). Ship each submodule's tracked tree at the commit the superproject
  # pins at $sha, so ogbench-sourced mazes (conditional_eval) exist at prepare
  # time instead of failing FileNotFound on the coordinator.
  local path sub_sha
  while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    sub_sha="$(git rev-parse "$sha:$path" 2>/dev/null)" \
      || { echo "code-sync: cannot resolve submodule '$path' at $sha" >&2; return 1; }
    git -C "$path" archive --format=tar "$sub_sha" \
      | gcloud compute ssh "$vm" --zone "$zone" --command \
          "mkdir -p ~/MultiNet-v2.0/$path && tar -x -C ~/MultiNet-v2.0/$path" \
      || { echo "code-sync: submodule '$path' push failed on $vm" >&2; return 1; }
  done < <(git ls-tree -r "$sha" | awk '$2 == "commit" { print $4 }')
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
  # Fail-closed on submodules: git archive skips them, so confirm each one's tree
  # actually landed (the guarded failure was an empty ogbench dir → mazes missing
  # at prepare time). sync_code_to_vm overwrites with the pinned-sha content, so
  # presence implies freshness here.
  local path
  while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    if ! gcloud compute ssh "$vm" --zone "$zone" --command \
         "test -n \"\$(find ~/MultiNet-v2.0/$path -type f -print -quit 2>/dev/null)\"" 2>/dev/null; then
      echo "code-sync: submodule '$path' is empty/missing on $vm" >&2; return 1
    fi
  done < <(git ls-tree -r "$sha" | awk '$2 == "commit" { print $4 }')
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

# Populate COORD, GPU_VMS[], API_VMS[], ALL_WORKERS[] from the topology helper.
derive_names() {
  local topo; topo="$(python3 -m scripts.distributed_topology "$RUN_CONFIG" "$RUN_ID")"
  COORD="$(printf '%s' "$topo" | python3 -c 'import json,sys; print(json.load(sys.stdin)["coordinator"]["name"])')"
  mapfile -t GPU_VMS < <(printf '%s' "$topo" | python3 -c 'import json,sys; [print(w["name"]) for w in json.load(sys.stdin)["workers"] if w["kind"]=="gpu"]')
  mapfile -t API_VMS < <(printf '%s' "$topo" | python3 -c 'import json,sys; [print(w["name"]) for w in json.load(sys.stdin)["workers"] if w["kind"]=="api"]')
  TOPO_JSON="$topo"
}

write_manifest() {  # $1 zone  $2 coord_ip
  local zone="$1" ip="$2" mf; mf="$(manifest_path)"
  mkdir -p "$(dirname "$mf")"
  RUN_ID="$RUN_ID" ZONE="$zone" COORD_IP="$ip" CODE_SHA="$CODE_SHA" \
  MAX_RUN_DURATION="$MAX_RUN_DURATION" TOPO_JSON="$TOPO_JSON" \
  python3 - "$mf" <<'PY'
import json, os, sys, datetime
topo = json.loads(os.environ["TOPO_JSON"])
out = {
    "run_id": os.environ["RUN_ID"],
    "zone": os.environ["ZONE"],
    "max_run_duration": os.environ["MAX_RUN_DURATION"],
    "code_sha": os.environ["CODE_SHA"],
    "coordinator": {"name": topo["coordinator"]["name"], "internal_ip": os.environ["COORD_IP"]},
    "workers": [{"name": w["name"], "kind": w["kind"], "model_group": w["model_group"]}
                for w in topo["workers"]],
    "artifacts_root_remote": f"artifacts/{os.environ['RUN_ID']}",
    "created_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
open(sys.argv[1], "w").write(json.dumps(out, indent=2) + "\n")
PY
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

  require_clean_tree
  CODE_SHA="$(git rev-parse HEAD)"
  [[ -n "${COORD:-}" ]] || derive_names

  # --- create phase: DELETE-safe (empty, pre-data VMs) -------------------------
  if [[ "${#GPU_VMS[@]}" -gt 0 ]]; then
    ZONE="$(hunt_zones "$COORD" "${GPU_VMS[@]}")" || exit 2
  else
    cs_create_instance "$COORD" "$COORD_IMAGE" "$ZONE" || exit 2
  fi
  local vm
  for vm in "${API_VMS[@]:-}"; do
    [[ -n "$vm" ]] || continue
    if ! cs_create_instance "$vm" "$API_IMAGE" "$ZONE"; then
      echo "API VM create failed: $vm; rolling back (pre-data)" >&2
      cs_delete_vms "$ZONE" "$COORD" "${GPU_VMS[@]:-}" "${API_VMS[@]:-}"
      exit 2
    fi
  done

  # --- from here on, VMs may hold data: failures STOP, never DELETE ------------
  local ALL_VMS=("$COORD")
  [[ "${#GPU_VMS[@]}" -gt 0 ]] && ALL_VMS+=("${GPU_VMS[@]}")
  [[ "${#API_VMS[@]}" -gt 0 ]] && ALL_VMS+=("${API_VMS[@]}")
  abort_stop() { echo "[launch_distributed] $1; STOPping all (data preserved)" >&2; cs_stop_vms "$ZONE" "${ALL_VMS[@]}"; exit 1; }

  for vm in "${ALL_VMS[@]}"; do wait_for_ssh "$vm" "$ZONE" || abort_stop "ssh wait failed on $vm"; done
  sync_and_verify "$CODE_SHA" "$ZONE" "${ALL_VMS[@]}" || abort_stop "code-sync verification failed"
  for vm in "${ALL_VMS[@]}"; do arm_watchdog "$vm" "$ZONE" 1 || abort_stop "watchdog arm failed on $vm"; done
  for vm in "${ALL_VMS[@]}"; do assert_no_resource_policy "$vm" "$ZONE" || abort_stop "resource policy found on $vm"; done

  local COORD_IP
  COORD_IP="$(internal_ip "$COORD" "$ZONE")" || abort_stop "internal_ip failed"
  start_coordinator || abort_stop "coordinator start failed"
  for vm in "${GPU_VMS[@]:-}" "${API_VMS[@]:-}"; do
    [[ -n "$vm" ]] || continue
    start_worker "$vm" "$COORD_IP" || abort_stop "worker start failed on $vm"
  done

  write_manifest "$ZONE" "$COORD_IP" || abort_stop "write_manifest failed"
  log "LAUNCH COMPLETE: run_id=$RUN_ID zone=$ZONE manifest=$(manifest_path)"
  return 0
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
