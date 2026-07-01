#!/usr/bin/env bash
# Cost-safety foundation for distributed runs. SOURCE this file; do not execute it.
# Single source of truth for: duration math, the GCP max-run-duration/STOP floor,
# the on-VM shutdown watchdog, the resource-policy guard, and stop/delete.
# Every cloud function takes zone explicitly (no global $ZONE dependency) and
# returns non-zero on failure (never exits) so callers can STOP-then-abort.

log() { printf '[%s] %s\n' "$(date -Is)" "$*" >&2; }

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
    echo "MAX_RUN_DURATION is required (no default — a wrong default either kills a" >&2
    echo "legit run or fails to protect). Set it to expected runtime + margin." >&2
    echo "  e.g. export MAX_RUN_DURATION=12h" >&2
    return 1
  fi
  if ! is_valid_duration "${MAX_RUN_DURATION}"; then
    echo "MAX_RUN_DURATION malformed: '${MAX_RUN_DURATION}' (use forms like 6h, 90m, 1d12h)." >&2
    return 1
  fi
  return 0
}

validate_run_id() {
  if [[ ! "${RUN_ID:-}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_ID must contain only letters, numbers, dot, underscore, and dash: ${RUN_ID:-}" >&2
    return 1
  fi
  return 0
}

require_gcloud() {
  command -v gcloud >/dev/null 2>&1 || { echo "gcloud is not on PATH." >&2; return 1; }
}

instance_exists() {  # $1 name  $2 zone
  gcloud compute instances describe "$1" --zone "$2" >/dev/null 2>&1
}

cs_create_instance() {  # $1 name  $2 image  $3 zone  — applies the GCP floor
  gcloud compute instances create "$1" \
    --zone "$3" \
    --source-machine-image="$2" \
    --max-run-duration="$MAX_RUN_DURATION" \
    --instance-termination-action=STOP
}

wait_for_ssh() {  # $1 name  $2 zone
  local name="$1" zone="$2"
  log "waiting for SSH: $name"
  local _i
  for _i in $(seq 1 60); do
    if gcloud compute ssh "$name" --zone "$zone" --command "true" >/dev/null 2>&1; then
      log "SSH ready: $name"; return 0
    fi
    sleep 5
  done
  echo "Timed out waiting for SSH on $name" >&2
  return 1
}

arm_watchdog() {  # $1 name  $2 zone  $3 created_flag(1|0)
  local name="$1" zone="$2" created="${3:-0}" mins
  mins="$(watchdog_minutes "$MAX_RUN_DURATION")" || return 1
  log "arming on-VM shutdown watchdog on $name (+${mins}m)"
  if gcloud compute ssh "$name" --zone "$zone" \
       --command "sudo shutdown -c 2>/dev/null || true; sudo shutdown -h +${mins}" >/dev/null 2>&1; then
    log "watchdog armed: $name (+${mins}m)"; return 0
  fi
  if [[ "$created" == "1" ]]; then
    log "WARNING: could not arm watchdog on $name; GCP floor still protects this fresh VM"; return 0
  fi
  echo "FATAL: could not arm watchdog on reused VM $name (no GCP floor)." >&2
  return 1
}

internal_ip() {  # $1 name  $2 zone
  gcloud compute instances describe "$1" --zone "$2" \
    --format='value(networkInterfaces[0].networkIP)'
}

assert_no_resource_policy() {  # $1 name  $2 zone
  local name="$1" zone="$2" disk_uris disk_uri disk_name policies
  disk_uris="$(gcloud compute instances describe "$name" --zone "$zone" --format='value(disks[].source)')"
  disk_uris="${disk_uris//;/ }"
  [[ -n "$disk_uris" ]] || { echo "Could not enumerate disks for $name" >&2; return 1; }
  for disk_uri in $disk_uris; do
    disk_name="${disk_uri##*/}"
    policies="$(gcloud compute disks describe "$disk_name" --zone "$zone" --format='value(resourcePolicies)' || true)"
    [[ -z "$policies" ]] || { echo "Disk $disk_name for $name has resourcePolicies: $policies" >&2; return 1; }
    log "no disk resource policy: $name / $disk_name"
  done
}

cs_stop_vms() {  # $1 zone  $2.. vms  — STOP, preserve disks
  local zone="$1"; shift
  local vm
  for vm in "$@"; do
    [[ -n "$vm" ]] || continue
    gcloud compute instances stop "$vm" --zone "$zone" --quiet || true
  done
}

cs_delete_vms() {  # $1 zone  $2.. vms  — DELETE (caller asserts the VMs hold no data)
  local zone="$1"; shift
  local vm
  for vm in "$@"; do
    [[ -n "$vm" ]] || continue
    gcloud compute instances delete "$vm" --zone "$zone" --quiet || true
  done
}
