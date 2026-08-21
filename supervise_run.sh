#!/usr/bin/env bash
# supervise_run.sh — self-contained distributed-run supervisor.
# Consumes .runs/<run_id>/manifest.json, drives a live run to a terminal state,
# owns the guaranteed STOP on every terminal path, NEVER deletes. Does not use or
# modify monitor_run.sh.
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib/cost_safety.sh"

# ---- inputs / defaults (env-overridable where noted) ----------------------- #
MANIFEST=""
DEST=""
INTERVAL="${INTERVAL:-600}"
STALL_MINUTES="${STALL_MINUTES:-60}"
HARDCAP_OVERRIDE=""
ONCE=0
CONSEC_ERR_MAX="${CONSEC_ERR_MAX:-3}"
EGRESS_FAILED_CODE=40   # graceful terminal reached but egress FAILED: VMs left up, no verdict, retry
LOG_TAIL_LINES="${LOG_TAIL_LINES:-120}"

# ---- populated by load_manifest ------------------------------------------- #
RUN_ID=""; ZONE=""; COORD=""; MAX_RUN_DURATION_MF=""; CREATED_AT=""; ARTIFACTS_ROOT_REMOTE=""
WORKERS=()

# ---- populated by supervise_main / resolve_hardcap ------------------------- #
STATE_FILE=""; VERDICT_FILE=""; LOGDIR=""
HARDCAP_DEADLINE=0; HARDCAP_SRC=""

# ---- last-probe counts (for the verdict line) ------------------------------ #
LAST_T="?"; LAST_V="?"; LAST_F="?"

# NOTE: log() is provided by lib/cost_safety.sh (timestamped, -> stderr).

usage() {
  cat >&2 <<'USAGE'
usage: supervise_run.sh --manifest PATH --dest DIR
                        [--interval SEC] [--stall-minutes N] [--hardcap DURATION] [--once]
USAGE
  return 2
}

now_epoch() { echo "${SUPERVISOR_NOW_EPOCH:-$(date +%s)}"; }

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --manifest) MANIFEST="${2:-}"; shift 2 ;;
      --dest) DEST="${2:-}"; shift 2 ;;
      --interval) INTERVAL="${2:-}"; shift 2 ;;
      --stall-minutes) STALL_MINUTES="${2:-}"; shift 2 ;;
      --hardcap) HARDCAP_OVERRIDE="${2:-}"; shift 2 ;;
      --once) ONCE=1; shift ;;
      *) echo "unknown argument: $1" >&2; usage; return 2 ;;
    esac
  done
  [[ -n "$MANIFEST" ]] || { echo "--manifest is required" >&2; usage; return 2; }
  [[ -n "$DEST" ]] || { echo "--dest is required" >&2; usage; return 2; }
  [[ "$INTERVAL" =~ ^[0-9]+$ ]] || { echo "--interval must be a non-negative integer (seconds): $INTERVAL" >&2; return 2; }
  [[ "$STALL_MINUTES" =~ ^[0-9]+$ ]] || { echo "--stall-minutes must be a non-negative integer: $STALL_MINUTES" >&2; return 2; }
  return 0
}

load_manifest() {
  [[ -f "$MANIFEST" ]] || { echo "no manifest at $MANIFEST" >&2; return 1; }
  local kv
  kv="$(python3 - "$MANIFEST" <<'PY'
import json, sys, shlex
def out(k, v): print(f"{k}={shlex.quote(str(v))}")
try:
    d = json.load(open(sys.argv[1]))
    out("RUN_ID", d["run_id"])
    out("ZONE", d["zone"])
    out("MAX_RUN_DURATION_MF", d["max_run_duration"])
    out("CREATED_AT", d.get("created_at", ""))
    out("COORD", d["coordinator"]["name"])
    out("ARTIFACTS_ROOT_REMOTE", d["artifacts_root_remote"])
    print("WORKERS=(" + " ".join(shlex.quote(w["name"]) for w in d["workers"]) + ")")
except KeyError as e:
    sys.stderr.write(f"manifest missing required key: {e}\n"); sys.exit(1)
except (ValueError, TypeError) as e:
    sys.stderr.write(f"manifest malformed: {e}\n"); sys.exit(1)
PY
)" || { echo "failed to parse manifest $MANIFEST" >&2; return 1; }
  eval "$kv"
}

iso_to_epoch() {  # $1 ISO8601 (…Z) -> epoch on stdout, or nonzero
  python3 - "$1" <<'PY'
import sys, datetime
s = sys.argv[1]
if not s:
    sys.exit(1)
try:
    dt = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
except ValueError:
    sys.exit(1)
if dt.tzinfo is None:
    # Reject tz-naive input: a missing offset would silently anchor the hardcap to
    # local time and skew the deadline. The frozen manifest always emits a 'Z'.
    sys.stderr.write(f"created_at is timezone-naive (needs an offset like 'Z'): {s}\n")
    sys.exit(1)
print(int(dt.timestamp()))
PY
}

resolve_hardcap() {
  local dur secs created
  if [[ -n "$HARDCAP_OVERRIDE" ]]; then dur="$HARDCAP_OVERRIDE"; HARDCAP_SRC="override"
  else dur="$MAX_RUN_DURATION_MF"; HARDCAP_SRC="manifest"; fi
  secs="$(parse_duration_seconds "$dur")" || { echo "bad hardcap duration: $dur" >&2; return 1; }
  created="$(iso_to_epoch "$CREATED_AT")" || { echo "manifest created_at missing/unparseable: '$CREATED_AT'" >&2; return 1; }
  HARDCAP_DEADLINE=$(( created + secs ))
}

log_resolved_config() {
  local hc
  if [[ -n "$HARDCAP_OVERRIDE" ]]; then hc="$HARDCAP_OVERRIDE"; else hc="$MAX_RUN_DURATION_MF"; fi
  log "supervise start: run_id=$RUN_ID zone=$ZONE coord=$COORD workers=[${WORKERS[*]}] dest=$DEST interval=${INTERVAL}s stall=${STALL_MINUTES}m hardcap=${hc}(${HARDCAP_SRC}) created_at=$CREATED_AT deadline_epoch=$HARDCAP_DEADLINE"
}

probe_status() {  # echoes "T V F R P PROG" or returns 1
  local json
  if [[ -n "${SUPERVISOR_TEST_STATUS:-}" ]]; then
    json="$SUPERVISOR_TEST_STATUS"
  else
    json="$(gcloud compute ssh "$COORD" --zone "$ZONE" \
              --command 'curl -fsS http://127.0.0.1:8765/status' 2>/dev/null)" || return 1
  fi
  [[ -n "$json" ]] || return 1
  printf '%s' "$json" | python3 -c '
import json, sys
d = json.load(sys.stdin); u = d.get("units", {}) or {}
print(int(d.get("unit_count", 0)), int(u.get("verified", 0)), int(u.get("failed", 0)),
      int(u.get("running", 0)), int(u.get("pending", 0)), int(d.get("progress_total", 0)))
' 2>/dev/null || return 1
}

make_sig() {  # $1..$6 = T V F R P PROG
  printf 'v:%s|f:%s|r:%s|p:%s|prog:%s|total:%s' "$2" "$3" "$4" "$5" "$6" "$1"
}

state_read_sig() {
  [[ -f "$STATE_FILE" ]] || { echo ""; return 0; }
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("sig") or "")' "$STATE_FILE" 2>/dev/null || echo ""
}

state_read_ts() {
  [[ -f "$STATE_FILE" ]] || { echo 0; return 0; }
  python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("ts",0)))' "$STATE_FILE" 2>/dev/null || echo 0
}

state_write() {  # $1 sig  $2 ts
  python3 - "$STATE_FILE" "$1" "$2" <<'PY'
import json, sys
json.dump({"sig": sys.argv[2], "ts": int(sys.argv[3])}, open(sys.argv[1], "w"))
PY
}

snapshot_logs() {
  mkdir -p "$LOGDIR" || true
  local w
  for w in "${WORKERS[@]}"; do
    [[ -n "$w" ]] || continue
    gcloud compute ssh "$w" --zone "$ZONE" \
      --command "tail -n ${LOG_TAIL_LINES} ${WORKER_LOG_PATH:-~/multinet-worker-artifacts/$RUN_ID/worker.log} 2>/dev/null" \
      >"$LOGDIR/$w.worker.log" 2>/dev/null || true
  done
  gcloud compute ssh "$COORD" --zone "$ZONE" \
    --command "tail -n ${LOG_TAIL_LINES} ~/MultiNet-v2.0/${ARTIFACTS_ROOT_REMOTE}/coordinator-serve.log 2>/dev/null" \
    >"$LOGDIR/coord-serve.log" 2>/dev/null || true
  return 0
}

pull() {  # land the run's artifact tree at EXACTLY $DEST; return 1 on failure
  local tmp base
  tmp="$(mktemp -d)"
  base="$(basename "$ARTIFACTS_ROOT_REMOTE")"
  if gcloud compute scp --recurse --zone "$ZONE" \
       "$COORD:~/MultiNet-v2.0/${ARTIFACTS_ROOT_REMOTE}" "$tmp" >/dev/null 2>&1 \
       && [[ -d "$tmp/$base" ]]; then
    rm -rf "$DEST"
    mkdir -p "$(dirname "$DEST")"
    mv "$tmp/$base" "$DEST" || { rm -rf "$tmp"; return 1; }
    rm -rf "$tmp"
    return 0
  fi
  rm -rf "$tmp"
  return 1
}

run_finalize() {
  gcloud compute ssh "$COORD" --zone "$ZONE" --command \
    "cd ~/MultiNet-v2.0 && source .venv-multinet/bin/activate && python -m scripts.run_pipeline --distributed-role coordinator-finalize --artifacts-root ${ARTIFACTS_ROOT_REMOTE} --run-set-id ${RUN_ID}"
}

stop_vms() {
  cs_stop_vms "$ZONE" "$COORD" "${WORKERS[@]}"
}

write_verdict() {  # $1 state  $2 code
  printf '%s code=%s verified=%s failed=%s total=%s\n' \
    "$1" "$2" "$LAST_V" "$LAST_F" "$LAST_T" >"$VERDICT_FILE"
}

read_verdict() { [[ -f "$VERDICT_FILE" ]] && cat "$VERDICT_FILE"; }

verdict_code() {  # echo code, return 0; return 1 if no/malformed verdict
  [[ -f "$VERDICT_FILE" ]] || return 1
  local line; line="$(cat "$VERDICT_FILE")"
  [[ "$line" =~ code=([0-9]+) ]] || return 1
  echo "${BASH_REMATCH[1]}"
}

graceful_terminate() {  # $1 state  $2 code — natural end; guarantee egress before STOP
  local state="$1" code="$2"
  run_finalize || log "finalize returned nonzero on $state (continuing)"
  if pull; then
    stop_vms
    write_verdict "$state" "$code"
    log "$state: finalized, pulled, STOPped -> exit $code"
    return "$code"
  fi
  # Pull failed on a graceful terminal: do NOT stop, write NO verdict (so a restart
  # re-attaches and retries), and exit the dedicated egress-failed code so automation
  # never mistakes this VMs-still-running state for success.
  log "ALERT: pull FAILED on $state; NOT stopping and writing NO verdict — VMs left up, data on coordinator disk; exit ${EGRESS_FAILED_CODE} (a restart will retry)."
  return "$EGRESS_FAILED_CODE"
}

abnormal_terminate() {  # $1 state  $2 code — force-terminate; STOP preserves disk
  local state="$1" code="$2"
  pull || log "pull failed on $state (best-effort); data remains on coordinator disk"
  stop_vms
  write_verdict "$state" "$code"
  log "$state: pulled(best-effort), STOPped -> exit $code"
  return "$code"
}

supervise_once() {  # returns 0|10|21|20|30|3
  snapshot_logs
  local now; now="$(now_epoch)"
  if (( now >= HARDCAP_DEADLINE )); then
    log "HARDCAP reached (now=$now >= deadline=$HARDCAP_DEADLINE)"
    abnormal_terminate hardcap 30; return $?
  fi

  local counts
  counts="$(probe_status)" || { log "probe failed"; return 3; }
  local T V F R P PROG
  read -r T V F R P PROG <<<"$counts"
  LAST_T="$T"; LAST_V="$V"; LAST_F="$F"
  log "counts: total=$T verified=$V failed=$F running=$R pending=$P progress=$PROG"

  if (( T > 0 && V == T )); then graceful_terminate complete 10; return $?; fi
  if (( T > 0 && R == 0 && P == 0 && V < T )); then graceful_terminate partial 21; return $?; fi

  # progress-aware stall bookkeeping
  local sig prev_sig prev_ts change_ts elapsed
  sig="$(make_sig "$T" "$V" "$F" "$R" "$P" "$PROG")"
  prev_sig="$(state_read_sig)"; prev_ts="$(state_read_ts)"
  (( prev_ts > 0 )) || prev_ts="$now"
  change_ts="$prev_ts"
  [[ "$sig" != "$prev_sig" ]] && change_ts="$now"
  state_write "$sig" "$change_ts"
  if [[ "$sig" == "$prev_sig" ]]; then
    elapsed=$(( now - change_ts ))
    if (( elapsed >= STALL_MINUTES * 60 )); then
      log "STALLED: no progress for ${elapsed}s (failed=$F)"
      abnormal_terminate stalled 20; return $?
    fi
  fi
  return 0
}

supervise_loop() {
  local consec_err=0 ticks=0 rc
  while true; do
    rc=0; supervise_once || rc=$?
    case "$rc" in
      0) consec_err=0 ;;
      3) consec_err=$(( consec_err + 1 ))
         log "probe error (consec=${consec_err}/${CONSEC_ERR_MAX})"
         if (( consec_err >= CONSEC_ERR_MAX )); then
           abnormal_terminate error 2; return $?
         fi ;;
      *) return "$rc" ;;   # terminal 10|21|20|30 (actions already done in supervise_once)
    esac
    ticks=$(( ticks + 1 ))
    if [[ -n "${SUPERVISOR_MAX_TICKS:-}" ]] && (( ticks >= SUPERVISOR_MAX_TICKS )); then
      log "SUPERVISOR_MAX_TICKS=${SUPERVISOR_MAX_TICKS} reached; exiting loop (non-terminal)"
      return 0
    fi
    sleep "$INTERVAL"
  done
}

supervise_main() {
  parse_args "$@" || return $?
  load_manifest || return 2
  local statedir; statedir="$(dirname "$MANIFEST")"
  STATE_FILE="$statedir/supervisor.state"
  VERDICT_FILE="$statedir/verdict"
  LOGDIR="$statedir/logs"
  resolve_hardcap || return 2

  # Resume short-circuit: a verdict means already-terminal. Do this BEFORE any
  # cloud call so a resolved run is free to re-check.
  local code
  if code="$(verdict_code)"; then
    log "verdict present ($(read_verdict | tr -d '\n')); already terminal; exiting $code"
    return "$code"
  fi

  log_resolved_config
  if [[ "$ONCE" == "1" ]]; then
    local rc=0; supervise_once || rc=$?
    return "$rc"
  fi
  supervise_loop
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  rc=0
  supervise_main "$@" || rc=$?
  exit "$rc"
fi
