#!/usr/bin/env bash
set -euo pipefail

# Passive monitor (Layer 2) for a distributed run. One-shot per invocation;
# intended to be driven on an interval (e.g. Claude's /loop). It reads the
# coordinator /status, classifies the run, and exits with a code the
# scheduler/Claude interprets:
#
#   0   running  (keep looping)
#   10  complete (all units verified; with --complete-actions: finalize+egress+stop done)
#   20  stalled  (no progress for --stall-minutes; alert)
#   2   usage / fetch error
#
# All automatic actions preserve data: finalize -> egress (scp) -> STOP.
# Egress MUST succeed before any spin-down; nothing here ever deletes.

ZONE=""
COORD=""
QWEN1=""
QWEN2=""
KIMI=""
RUN_ID=""
DEST=""
STALL_MINUTES=""
STATE_FILE=""
COMPLETE_ACTIONS=0

usage() {
  cat >&2 <<'USAGE'
usage: monitor_run.sh --once --coord NAME --zone ZONE --run-id ID
                      [--qwen1 NAME --qwen2 NAME --kimi NAME]
                      [--dest DIR] [--stall-minutes N] [--state-file PATH]
                      [--complete-actions]
USAGE
  exit 2
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --once) shift ;;                                  # one-shot is the only mode
      --complete-actions) COMPLETE_ACTIONS=1; shift ;;
      --coord) COORD="${2:-}"; shift 2 ;;
      --qwen1) QWEN1="${2:-}"; shift 2 ;;
      --qwen2) QWEN2="${2:-}"; shift 2 ;;
      --kimi) KIMI="${2:-}"; shift 2 ;;
      --zone) ZONE="${2:-}"; shift 2 ;;
      --run-id) RUN_ID="${2:-}"; shift 2 ;;
      --dest) DEST="${2:-}"; shift 2 ;;
      --stall-minutes) STALL_MINUTES="${2:-}"; shift 2 ;;
      --state-file) STATE_FILE="${2:-}"; shift 2 ;;
      *) echo "unknown argument: $1" >&2; usage ;;
    esac
  done
}

now_epoch() { echo "${MONITOR_NOW_EPOCH:-$(date +%s)}"; }

# Test seam: MONITOR_TEST_STATUS short-circuits the gcloud call.
fetch_status() {
  if [[ -n "${MONITOR_TEST_STATUS:-}" ]]; then
    printf '%s' "$MONITOR_TEST_STATUS"
    return 0
  fi
  gcloud compute ssh "$COORD" --zone "$ZONE" \
    --command 'curl -fsS http://127.0.0.1:8765/status' 2>/dev/null
}

# Reads /status JSON on stdin, prints: "<total>\t<verified>\t<failed>\t<signature>".
# Signature changes whenever any unit advances, so an unchanged signature over
# time means no progress.
status_fields() {
  # NOTE: must use `python3 -c` (not `python3 - <<HEREDOC`): a heredoc would
  # become python's script source and consume stdin, leaving the piped JSON
  # unreadable via sys.stdin.
  python3 -c '
import json, sys
d = json.load(sys.stdin)
units = d.get("units", {}) or {}
total = int(d.get("unit_count", 0))
verified = int(units.get("verified", 0))
failed = int(units.get("failed", 0))
progress = int(d.get("progress_total", 0))
sig = "|".join(f"{k}:{units[k]}" for k in sorted(units)) + f"|total:{total}|progress:{progress}"
print(f"{total}\t{verified}\t{failed}\t{sig}")
'
}

state_read_sig() {
  [[ -n "$STATE_FILE" && -f "$STATE_FILE" ]] || { echo ""; return 0; }
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("sig",""))' "$STATE_FILE" 2>/dev/null || echo ""
}

state_read_ts() {
  [[ -n "$STATE_FILE" && -f "$STATE_FILE" ]] || { echo "0"; return 0; }
  python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("ts",0)))' "$STATE_FILE" 2>/dev/null || echo "0"
}

state_write() {  # $1=sig $2=ts
  [[ -n "$STATE_FILE" ]] || return 0
  python3 - "$STATE_FILE" "$1" "$2" <<'PY'
import json, sys
with open(sys.argv[1], "w") as fh:
    json.dump({"sig": sys.argv[2], "ts": int(sys.argv[3])}, fh)
PY
}

# --- completion actions (overridable in tests) ------------------------------ #

run_finalize() {
  gcloud compute ssh "$COORD" --zone "$ZONE" --command \
    "cd ~/MultiNet-v2.0 && source .venv-multinet/bin/activate && python -m scripts.run_pipeline --distributed-role coordinator-finalize --artifacts-root artifacts/$RUN_ID --run-set-id $RUN_ID"
}

egress_artifacts() {
  if [[ -z "$DEST" ]]; then
    echo "[monitor] --dest is required to egress before spin-down" >&2
    return 1
  fi
  mkdir -p "$DEST"
  gcloud compute scp --recurse --zone "$ZONE" \
    "$COORD:~/MultiNet-v2.0/artifacts/$RUN_ID" "$DEST"
}

stop_all_vms() {
  local vm
  for vm in "$COORD" "$QWEN1" "$QWEN2" "$KIMI"; do
    [[ -n "$vm" ]] || continue
    gcloud compute instances stop "$vm" --zone "$ZONE" --quiet || true
  done
}

# Safety-critical ordering: finalize, then egress; spin down ONLY if egress
# succeeded. A failed egress leaves the data on the (still-running) coordinator.
complete_actions() {
  if ! run_finalize; then
    echo "[monitor] finalize failed; not egressing or stopping" >&2
    return 1
  fi
  if egress_artifacts; then
    stop_all_vms
  else
    echo "[monitor] egress failed; NOT stopping — data remains on the coordinator disk" >&2
    return 1
  fi
}

main_once() {
  local json fields total verified failed sig
  json="$(fetch_status)" || { echo "[monitor] could not fetch status from ${COORD:-?}" >&2; return 2; }
  if [[ -z "$json" ]]; then echo "[monitor] empty status response" >&2; return 2; fi
  fields="$(printf '%s' "$json" | status_fields)" || { echo "[monitor] could not parse status" >&2; return 2; }
  IFS=$'\t' read -r total verified failed sig <<<"$fields"
  echo "[monitor] verified=${verified}/${total} failed=${failed}"

  # Stall detection compares the signature against the previous tick, which only
  # works if the previous tick was persisted. Without --state-file every tick
  # starts blank, the signature always looks "changed", and a hang is never
  # caught — warn loudly rather than silently disabling a safety check.
  if [[ -n "$STALL_MINUTES" && -z "$STATE_FILE" ]]; then
    echo "[monitor] WARNING: --stall-minutes set without --state-file; stall detection needs state persisted across ticks and is DISABLED this way." >&2
  fi

  if [[ "$total" -gt 0 && "$verified" -eq "$total" ]]; then
    echo "[monitor] run complete"
    if [[ "$COMPLETE_ACTIONS" -eq 1 ]]; then
      complete_actions || return 2
    fi
    return 10
  fi

  local now prev_sig prev_ts change_ts elapsed
  now="$(now_epoch)"
  prev_sig="$(state_read_sig)"
  prev_ts="$(state_read_ts)"
  [[ "$prev_ts" -gt 0 ]] || prev_ts="$now"
  change_ts="$prev_ts"
  if [[ "$sig" != "$prev_sig" ]]; then
    change_ts="$now"   # progress -> reset the stall clock
  fi
  state_write "$sig" "$change_ts"

  if [[ -n "$STALL_MINUTES" && "$sig" == "$prev_sig" ]]; then
    elapsed=$(( now - change_ts ))
    if [[ "$elapsed" -ge $(( STALL_MINUTES * 60 )) ]]; then
      echo "[monitor] STALLED: no progress for ${elapsed}s (failed=${failed})" >&2
      return 20
    fi
  fi
  return 0
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  parse_args "$@"
  rc=0
  main_once || rc=$?
  exit "$rc"
fi
