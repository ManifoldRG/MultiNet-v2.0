"""Bash tests for supervise_run.sh (self-contained distributed-run supervisor).

All cloud calls are stubbed: fake `gcloud` on PATH, or `SUPERVISOR_TEST_STATUS` to
inject /status, or function overrides. No real cloud is ever invoked.
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def bash(snippet: str, env: dict | None = None) -> subprocess.CompletedProcess:
    e = os.environ.copy()
    e.pop("MAX_RUN_DURATION", None)
    if env:
        e.update(env)
    return subprocess.run(["bash", "-c", snippet], capture_output=True, text=True, cwd=REPO, env=e)


def iso_epoch(s: str) -> int:
    return int(datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def _status(unit_count, verified=0, failed=0, running=0, pending=0, progress_total=0) -> str:
    return json.dumps({
        "job_id": "j", "unit_count": unit_count,
        "units": {"verified": verified, "failed": failed, "running": running, "pending": pending},
        "worker_count": 2, "progress_total": progress_total,
    })


def _write_manifest(tmp_path: Path, *, run_id="r1", zone="z1",
                    created_at="2026-07-01T00:00:00Z", max_run_duration="12h",
                    workers=None) -> Path:
    workers = workers if workers is not None else [
        {"name": f"{run_id}-w-0", "kind": "gpu", "model_group": "g"},
        {"name": f"{run_id}-api-0", "kind": "api", "model_group": "k"},
    ]
    d = {
        "run_id": run_id, "zone": zone, "max_run_duration": max_run_duration,
        "code_sha": "deadbeef",
        "coordinator": {"name": f"{run_id}-coord", "internal_ip": "10.0.0.2"},
        "workers": workers,
        "artifacts_root_remote": f"artifacts/{run_id}",
        "created_at": created_at,
    }
    md = tmp_path / ".runs" / run_id
    md.mkdir(parents=True, exist_ok=True)
    mf = md / "manifest.json"
    mf.write_text(json.dumps(d, indent=2))
    return mf


def _fake_gcloud(tmp_path: Path) -> None:
    """Logs every call to $GCLOUD_LOG (if set) else stdout; always exits 0."""
    fake = tmp_path / "gcloud"
    fake.write_text('#!/usr/bin/env bash\necho "GCLOUD $*" >> "${GCLOUD_LOG:-/dev/stdout}"\nexit 0\n')
    fake.chmod(0o755)


def _fake_gcloud_scp_creates(tmp_path: Path) -> None:
    """On `compute scp`, creates <dest>/<basename-of-src>/artifact.txt (simulates a pull)."""
    fake = tmp_path / "gcloud"
    fake.write_text(
        '#!/usr/bin/env bash\n'
        'echo "GCLOUD $*" >> "${GCLOUD_LOG:-/dev/stdout}"\n'
        'if [[ "$1 $2" == "compute scp" ]]; then\n'
        '  dest="${@: -1}"; src="${@: -2:1}"; base="${src##*/}"\n'
        '  mkdir -p "$dest/$base"; echo hi > "$dest/$base/artifact.txt"\n'
        'fi\n'
        'exit 0\n'
    )
    fake.chmod(0o755)


def _fake_gcloud_integration(tmp_path: Path, statuses: list[str]) -> None:
    """`ssh ... curl /status` returns statuses[i] (clamped at last), advancing per call;
    `compute scp` creates the staging tree so pull() succeeds."""
    for i, s in enumerate(statuses):
        (tmp_path / f"status_{i}").write_text(s)
    (tmp_path / "status_last").write_text(str(len(statuses) - 1))
    (tmp_path / "idx").write_text("0")
    fake = tmp_path / "gcloud"
    fake.write_text(
        '#!/usr/bin/env bash\n'
        'echo "GCLOUD $*" >> "$GCLOUD_LOG"\n'
        f'd="{tmp_path}"\n'
        'if [[ "$*" == *"curl -fsS"* ]]; then\n'
        '  i="$(cat "$d/idx")"; last="$(cat "$d/status_last")"\n'
        '  [[ "$i" -gt "$last" ]] && i="$last"\n'
        '  cat "$d/status_$i"\n'
        '  echo $(( $(cat "$d/idx") + 1 )) > "$d/idx"\n'
        '  exit 0\n'
        'fi\n'
        'if [[ "$1 $2" == "compute scp" ]]; then\n'
        '  dest="${@: -1}"; src="${@: -2:1}"; base="${src##*/}"\n'
        '  mkdir -p "$dest/$base"; echo hi > "$dest/$base/artifact.txt"\n'
        'fi\n'
        'exit 0\n'
    )
    fake.chmod(0o755)


def test_syntax_ok():
    r = bash("bash -n ./supervise_run.sh")
    assert r.returncode == 0, r.stderr


def test_load_manifest_populates(tmp_path):
    mf = _write_manifest(tmp_path)
    snippet = (
        f'source ./supervise_run.sh; MANIFEST="{mf}"; load_manifest; '
        'echo "RUN_ID=$RUN_ID ZONE=$ZONE COORD=$COORD ART=$ARTIFACTS_ROOT_REMOTE '
        'MRD=$MAX_RUN_DURATION_MF CREATED=$CREATED_AT W0=${WORKERS[0]} W1=${WORKERS[1]} NW=${#WORKERS[@]}"'
    )
    r = bash(snippet)
    assert r.returncode == 0, r.stderr
    assert "RUN_ID=r1" in r.stdout and "ZONE=z1" in r.stdout
    assert "COORD=r1-coord" in r.stdout and "ART=artifacts/r1" in r.stdout
    assert "MRD=12h" in r.stdout
    assert "W0=r1-w-0" in r.stdout and "W1=r1-api-0" in r.stdout and "NW=2" in r.stdout


def test_resolve_hardcap_from_manifest(tmp_path):
    mf = _write_manifest(tmp_path, created_at="2026-07-01T00:00:00Z", max_run_duration="12h")
    snippet = (
        f'source ./supervise_run.sh; MANIFEST="{mf}"; load_manifest; resolve_hardcap; '
        'echo "DEADLINE=$HARDCAP_DEADLINE SRC=$HARDCAP_SRC"'
    )
    r = bash(snippet)
    assert r.returncode == 0, r.stderr
    created = iso_epoch("2026-07-01T00:00:00Z")
    assert f"DEADLINE={created + 43200}" in r.stdout
    assert "SRC=manifest" in r.stdout


def test_resolve_hardcap_override(tmp_path):
    mf = _write_manifest(tmp_path, created_at="2026-07-01T00:00:00Z", max_run_duration="12h")
    snippet = (
        f'source ./supervise_run.sh; MANIFEST="{mf}"; HARDCAP_OVERRIDE="1h"; '
        'load_manifest; resolve_hardcap; echo "DEADLINE=$HARDCAP_DEADLINE SRC=$HARDCAP_SRC"'
    )
    r = bash(snippet)
    assert r.returncode == 0, r.stderr
    created = iso_epoch("2026-07-01T00:00:00Z")
    assert f"DEADLINE={created + 3600}" in r.stdout
    assert "SRC=override" in r.stdout


def test_missing_manifest_fails(tmp_path):
    r = bash(f'source ./supervise_run.sh; MANIFEST="{tmp_path}/nope.json"; load_manifest')
    assert r.returncode != 0
    assert "no manifest" in r.stderr


def test_parse_args_requires_dest():
    r = bash("source ./supervise_run.sh; parse_args --manifest m.json")
    assert r.returncode != 0
    assert "--dest is required" in r.stderr


def test_parse_args_requires_manifest():
    r = bash("source ./supervise_run.sh; parse_args --dest /tmp/out")
    assert r.returncode != 0
    assert "--manifest is required" in r.stderr


def test_parse_args_rejects_nonnumeric_interval():
    r = bash("source ./supervise_run.sh; parse_args --manifest m.json --dest d --interval abc")
    assert r.returncode == 2
    assert "--interval must be a non-negative integer" in r.stderr


def test_parse_args_rejects_nonnumeric_stall_minutes():
    r = bash("source ./supervise_run.sh; parse_args --manifest m.json --dest d --stall-minutes 5m")
    assert r.returncode == 2
    assert "--stall-minutes must be a non-negative integer" in r.stderr


def test_load_manifest_names_missing_key(tmp_path):
    # A manifest missing a required key must fail AND name the offending key.
    bad = tmp_path / "bad.json"
    bad.write_text('{"run_id":"r1","max_run_duration":"12h","coordinator":{"name":"c"},'
                   '"workers":[],"artifacts_root_remote":"artifacts/r1"}')  # no "zone"
    r = bash(f'source ./supervise_run.sh; MANIFEST="{bad}"; load_manifest')
    assert r.returncode != 0
    assert "missing required key" in r.stderr and "zone" in r.stderr


def test_iso_to_epoch_rejects_tz_naive():
    r = bash("source ./supervise_run.sh; iso_to_epoch '2026-07-01T00:00:00'; echo RC=$?")
    assert "RC=1" in r.stdout
    assert "timezone-naive" in r.stderr


def test_main_load_failure_exits_2(tmp_path):
    # A nonexistent manifest is a startup error -> documented exit code 2 (not 1).
    r = bash(f'bash ./supervise_run.sh --manifest "{tmp_path}/nope.json" --dest "{tmp_path}/out"')
    assert r.returncode == 2, r.stderr


def test_state_read_sig_null_returns_empty(tmp_path):
    # A state file with a literal null sig must read back as "" (not "None").
    sf = tmp_path / "s.json"
    sf.write_text('{"sig": null, "ts": 0}')
    r = bash(f'source ./supervise_run.sh; STATE_FILE="{sf}"; echo "SIG=[$(state_read_sig)]"')
    assert "SIG=[]" in r.stdout


def test_resolved_config_logged(tmp_path):
    mf = _write_manifest(tmp_path)
    snippet = (
        f'source ./supervise_run.sh; MANIFEST="{mf}"; DEST="{tmp_path}/out"; '
        'load_manifest; resolve_hardcap; log_resolved_config'
    )
    r = bash(snippet)
    assert r.returncode == 0, r.stderr
    assert "run_id=r1" in r.stderr and "coord=r1-coord" in r.stderr
    assert "deadline_epoch=" in r.stderr and "(manifest)" in r.stderr


def test_probe_parses_counts():
    env = {"SUPERVISOR_TEST_STATUS": _status(6, verified=2, failed=1, running=2, pending=1, progress_total=30)}
    r = bash("source ./supervise_run.sh; probe_status", env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "6 2 1 2 1 30"


def test_probe_missing_progress_defaults_zero():
    s = json.dumps({"unit_count": 6, "units": {"verified": 6}})
    r = bash("source ./supervise_run.sh; probe_status", env={"SUPERVISOR_TEST_STATUS": s})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "6 6 0 0 0 0"


def test_probe_malformed_status_fails():
    r = bash("source ./supervise_run.sh; probe_status", env={"SUPERVISOR_TEST_STATUS": "not json"})
    assert r.returncode != 0


def test_make_sig_format():
    r = bash("source ./supervise_run.sh; make_sig 6 2 1 2 1 30")
    assert r.stdout.strip() == "v:2|f:1|r:2|p:1|prog:30|total:6"


def test_state_write_read_roundtrip(tmp_path):
    sf = tmp_path / "s.json"
    r = bash(f'source ./supervise_run.sh; STATE_FILE="{sf}"; state_write "sigA" 1000; '
             'echo "SIG=$(state_read_sig) TS=$(state_read_ts)"')
    assert r.returncode == 0, r.stderr
    assert "SIG=sigA" in r.stdout and "TS=1000" in r.stdout


def test_state_read_absent_defaults(tmp_path):
    sf = tmp_path / "nope.json"
    r = bash(f'source ./supervise_run.sh; STATE_FILE="{sf}"; echo "SIG=[$(state_read_sig)] TS=$(state_read_ts)"')
    assert "SIG=[]" in r.stdout and "TS=0" in r.stdout


def test_pull_lands_at_exactly_dest(tmp_path):
    _fake_gcloud_scp_creates(tmp_path)
    dest = tmp_path / "out"
    snippet = (
        f'source ./supervise_run.sh; ZONE=z; COORD=c; ARTIFACTS_ROOT_REMOTE="artifacts/r1"; '
        f'DEST="{dest}"; pull; echo RC=$?'
    )
    r = bash(snippet, env={"PATH": f"{tmp_path}:{os.environ['PATH']}"})
    assert "RC=0" in r.stdout, r.stderr
    assert (dest / "artifact.txt").exists()          # landed at exactly --dest
    assert not (dest / "r1").exists()                # NOT nested under $DEST/<run_id>


def test_stop_vms_stops_coord_and_all_workers_no_delete(tmp_path):
    _fake_gcloud(tmp_path)
    glog = tmp_path / "g.log"
    snippet = 'source ./supervise_run.sh; ZONE=z1; COORD=r1-coord; WORKERS=(r1-w-0 r1-api-0); stop_vms'
    r = bash(snippet, env={"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(glog)})
    assert r.returncode == 0, r.stderr
    log = glog.read_text()
    assert "instances stop r1-coord" in log
    assert "instances stop r1-w-0" in log and "instances stop r1-api-0" in log
    assert "instances delete" not in log


def test_stop_vms_skips_empty_worker_names(tmp_path):
    _fake_gcloud(tmp_path)
    glog = tmp_path / "g.log"
    snippet = 'source ./supervise_run.sh; ZONE=z1; COORD=c; WORKERS=(w0 "" w1); stop_vms'
    r = bash(snippet, env={"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(glog)})
    assert r.returncode == 0, r.stderr
    log = glog.read_text()
    assert "instances stop w0" in log and "instances stop w1" in log
    # the empty element must not produce a bare `instances stop --zone` call
    assert "instances stop --zone" not in log


def test_finalize_issues_coordinator_finalize(tmp_path):
    _fake_gcloud(tmp_path)
    glog = tmp_path / "g.log"
    snippet = ('source ./supervise_run.sh; ZONE=z1; COORD=r1-coord; RUN_ID=r1; '
               'ARTIFACTS_ROOT_REMOTE="artifacts/r1"; run_finalize')
    r = bash(snippet, env={"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(glog)})
    assert r.returncode == 0, r.stderr
    log = glog.read_text()
    assert "coordinator-finalize" in log and "--run-set-id r1" in log
    assert "--artifacts-root artifacts/r1" in log


def test_verdict_write_read_code(tmp_path):
    vf = tmp_path / "verdict"
    snippet = (
        f'source ./supervise_run.sh; VERDICT_FILE="{vf}"; LAST_V=6; LAST_F=0; LAST_T=6; '
        'write_verdict complete 10; echo "CODE=$(verdict_code)"; read_verdict'
    )
    r = bash(snippet)
    assert r.returncode == 0, r.stderr
    assert "CODE=10" in r.stdout
    assert "complete code=10 verified=6 failed=0 total=6" in r.stdout


def test_verdict_code_absent_returns_nonzero(tmp_path):
    vf = tmp_path / "nope"
    r = bash(f'source ./supervise_run.sh; VERDICT_FILE="{vf}"; verdict_code; echo RC=$?')
    assert "RC=1" in r.stdout


def test_snapshot_logs_best_effort_never_fails(tmp_path):
    # gcloud that fails every ssh: snapshot_logs must still return 0.
    fake = tmp_path / "gcloud"
    fake.write_text('#!/usr/bin/env bash\nexit 1\n')
    fake.chmod(0o755)
    snippet = (f'source ./supervise_run.sh; ZONE=z; COORD=c; RUN_ID=r1; '
               f'ARTIFACTS_ROOT_REMOTE="artifacts/r1"; WORKERS=(w0); LOGDIR="{tmp_path}/logs"; '
               'snapshot_logs; echo RC=$?')
    r = bash(snippet, env={"PATH": f"{tmp_path}:{os.environ['PATH']}"})
    assert "RC=0" in r.stdout, r.stderr
    assert (tmp_path / "logs").is_dir()


def test_snapshot_logs_coord_uses_log_tail_lines(tmp_path):
    # The coordinator tail must honor $LOG_TAIL_LINES (not a hardcoded 60).
    _fake_gcloud(tmp_path)
    glog = tmp_path / "g.log"
    snippet = (f'source ./supervise_run.sh; ZONE=z; COORD=c; RUN_ID=r1; '
               f'ARTIFACTS_ROOT_REMOTE="artifacts/r1"; WORKERS=(); LOGDIR="{tmp_path}/logs"; '
               'LOG_TAIL_LINES=99; snapshot_logs')
    r = bash(snippet, env={"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(glog)})
    assert r.returncode == 0, r.stderr
    log = glog.read_text()
    assert "tail -n 99" in log and "coordinator-serve.log" in log
    assert "tail -n 60" not in log


def test_pull_returns_1_when_mv_fails(tmp_path):
    # Staging succeeds, but relocating into a read-only parent makes mv fail;
    # pull must report failure (return 1), not a false success.
    _fake_gcloud_scp_creates(tmp_path)
    parent = tmp_path / "ro"
    parent.mkdir()
    parent.chmod(0o500)  # r-x, no write: mv into it fails
    dest = parent / "out"  # dest path does NOT exist; will fail to create in read-only parent
    try:
        snippet = (
            f'source ./supervise_run.sh; ZONE=z; COORD=c; ARTIFACTS_ROOT_REMOTE="artifacts/r1"; '
            f'DEST="{dest}"; pull; echo RC=$?'
        )
        r = bash(snippet, env={"PATH": f"{tmp_path}:{os.environ['PATH']}"})
        assert "RC=1" in r.stdout, r.stderr
    finally:
        parent.chmod(0o700)


def test_graceful_pull_ok_stops_and_writes_verdict(tmp_path):
    vf = tmp_path / "verdict"
    snippet = (
        f'source ./supervise_run.sh; VERDICT_FILE="{vf}"; LAST_V=6; LAST_F=0; LAST_T=6; '
        'run_finalize(){ return 0; }; pull(){ return 0; }; stop_vms(){ echo STOPPED; }; '
        'graceful_terminate complete 10; echo RC=$?'
    )
    r = bash(snippet)
    assert "STOPPED" in r.stdout
    assert "RC=10" in r.stdout
    assert vf.exists() and "complete code=10" in vf.read_text()


def test_graceful_pull_fail_does_not_stop_or_write_verdict(tmp_path):
    vf = tmp_path / "verdict"
    snippet = (
        f'source ./supervise_run.sh; VERDICT_FILE="{vf}"; LAST_V=3; LAST_F=1; LAST_T=6; '
        'run_finalize(){ return 0; }; pull(){ return 1; }; stop_vms(){ echo STOPPED; }; '
        'graceful_terminate complete 10; echo RC=$?'
    )
    r = bash(snippet)
    assert "STOPPED" not in r.stdout            # data not off-box -> do NOT stop
    assert "RC=40" in r.stdout                  # dedicated egress-failed code, not a success code
    assert not vf.exists()                      # NO verdict -> a restart retries
    assert "ALERT" in r.stderr


def test_abnormal_pull_fail_still_stops_and_writes_verdict(tmp_path):
    vf = tmp_path / "verdict"
    snippet = (
        f'source ./supervise_run.sh; VERDICT_FILE="{vf}"; LAST_V=2; LAST_F=0; LAST_T=6; '
        'pull(){ return 1; }; stop_vms(){ echo STOPPED; }; '
        'abnormal_terminate hardcap 30; echo RC=$?'
    )
    r = bash(snippet)
    assert "STOPPED" in r.stdout                # abnormal stops even if pull failed (disk preserved)
    assert "RC=30" in r.stdout
    assert vf.exists() and "hardcap code=30" in vf.read_text()


def test_graceful_finalize_nonzero_still_pulls_and_stops(tmp_path):
    # finalize is best-effort: a nonzero run_finalize must NOT abort the pull+STOP.
    vf = tmp_path / "verdict"
    snippet = (
        f'source ./supervise_run.sh; VERDICT_FILE="{vf}"; LAST_V=6; LAST_F=0; LAST_T=6; '
        'run_finalize(){ return 1; }; pull(){ echo PULLED; return 0; }; stop_vms(){ echo STOPPED; }; '
        'graceful_terminate complete 10; echo RC=$?'
    )
    r = bash(snippet)
    assert "PULLED" in r.stdout and "STOPPED" in r.stdout   # best-effort finalize didn't block egress/STOP
    assert "RC=10" in r.stdout
    assert vf.exists() and "complete code=10" in vf.read_text()


_ONCE_STUBS = (
    'snapshot_logs(){ :; }; pull(){ return 0; }; stop_vms(){ echo STOP; }; run_finalize(){ return 0; }; '
)

def _once_prelude(tmp_path, deadline=99999999999, stall_minutes=1):
    return (
        'source ./supervise_run.sh; ' + _ONCE_STUBS +
        f'STATE_FILE="{tmp_path}/s.json"; VERDICT_FILE="{tmp_path}/verdict"; DEST="{tmp_path}/out"; '
        f'WORKERS=(w0); ARTIFACTS_ROOT_REMOTE="artifacts/r1"; RUN_ID=r1; ZONE=z; COORD=c; '
        f'HARDCAP_DEADLINE={deadline}; STALL_MINUTES={stall_minutes}; '
    )


def test_once_complete_returns_10(tmp_path):
    snippet = _once_prelude(tmp_path) + 'supervise_once; echo RC=$?'
    r = bash(snippet, env={"SUPERVISOR_TEST_STATUS": _status(6, verified=6), "SUPERVISOR_NOW_EPOCH": "1000"})
    assert "RC=10" in r.stdout, r.stderr
    assert "STOP" in r.stdout
    assert (tmp_path / "verdict").exists()   # graceful complete wrote the verdict


def test_once_partial_returns_21(tmp_path):
    # no running, no pending, verified<total -> nothing can progress
    snippet = _once_prelude(tmp_path) + 'supervise_once; echo RC=$?'
    r = bash(snippet, env={"SUPERVISOR_TEST_STATUS": _status(6, verified=4, failed=2, running=0, pending=0),
                           "SUPERVISOR_NOW_EPOCH": "1000"})
    assert "RC=21" in r.stdout, r.stderr
    assert "STOP" in r.stdout                 # partial is graceful -> reaches stop_vms
    assert (tmp_path / "verdict").exists()   # and wrote the verdict


def test_once_running_returns_0(tmp_path):
    snippet = _once_prelude(tmp_path) + 'supervise_once; echo RC=$?'
    r = bash(snippet, env={"SUPERVISOR_TEST_STATUS": _status(6, verified=2, running=3, pending=1),
                           "SUPERVISOR_NOW_EPOCH": "1000"})
    assert "RC=0" in r.stdout, r.stderr


def test_once_hardcap_returns_30(tmp_path):
    snippet = _once_prelude(tmp_path, deadline=500) + 'supervise_once; echo RC=$?'
    r = bash(snippet, env={"SUPERVISOR_NOW_EPOCH": "1000"})   # now(1000) >= deadline(500)
    assert "RC=30" in r.stdout, r.stderr
    assert "STOP" in r.stdout


def test_once_probe_error_returns_3(tmp_path):
    # No SUPERVISOR_TEST_STATUS and gcloud ssh curl fails -> probe returns 1 -> once returns 3.
    fake = tmp_path / "gcloud"; fake.write_text('#!/usr/bin/env bash\nexit 1\n'); fake.chmod(0o755)
    snippet = _once_prelude(tmp_path) + 'supervise_once; echo RC=$?'
    r = bash(snippet, env={"PATH": f"{tmp_path}:{os.environ['PATH']}", "SUPERVISOR_NOW_EPOCH": "1000"})
    assert "RC=3" in r.stdout, r.stderr


def test_once_stall_trips_after_threshold(tmp_path):
    status = _status(6, verified=0, running=2, pending=4, progress_total=10)
    pre = _once_prelude(tmp_path, stall_minutes=1)
    # tick 1 at t=1000 records signature, reports running
    r1 = bash(pre + 'supervise_once; echo RC=$?',
              env={"SUPERVISOR_TEST_STATUS": status, "SUPERVISOR_NOW_EPOCH": "1000"})
    assert "RC=0" in r1.stdout, r1.stderr
    # tick 2 at t=1120 (120s > 60s), identical status -> stalled
    r2 = bash(pre + 'supervise_once; echo RC=$?',
              env={"SUPERVISOR_TEST_STATUS": status, "SUPERVISOR_NOW_EPOCH": "1120"})
    assert "RC=20" in r2.stdout, r2.stderr


def test_once_progress_bump_resets_stall(tmp_path):
    pre = _once_prelude(tmp_path, stall_minutes=1)
    r1 = bash(pre + 'supervise_once; echo RC=$?',
              env={"SUPERVISOR_TEST_STATUS": _status(6, running=2, pending=4, progress_total=10),
                   "SUPERVISOR_NOW_EPOCH": "1000"})
    assert "RC=0" in r1.stdout, r1.stderr
    # same counts but progress advanced -> signature changed -> NOT stalled
    r2 = bash(pre + 'supervise_once; echo RC=$?',
              env={"SUPERVISOR_TEST_STATUS": _status(6, running=2, pending=4, progress_total=25),
                   "SUPERVISOR_NOW_EPOCH": "1120"})
    assert "RC=0" in r2.stdout, r2.stderr


def test_verdict_sentinel_short_circuits_zero_cloud(tmp_path):
    mf = _write_manifest(tmp_path)
    (tmp_path / ".runs" / "r1" / "verdict").write_text("complete code=10 verified=6 failed=0 total=6\n")
    _fake_gcloud(tmp_path)
    glog = tmp_path / "g.log"; glog.write_text("")
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(glog)}
    r = bash(f'bash ./supervise_run.sh --manifest "{mf}" --dest "{tmp_path}/out"', env=env)
    assert r.returncode == 10, r.stderr
    assert glog.read_text().strip() == ""          # NO cloud calls on a resolved run


def test_loop_reaches_complete_pulls_and_stops(tmp_path):
    mf = _write_manifest(tmp_path, created_at="2026-07-01T00:00:00Z", max_run_duration="12h")
    _fake_gcloud_integration(tmp_path, statuses=[
        _status(6, verified=3, running=3), _status(6, verified=6)])
    glog = tmp_path / "g.log"
    now = iso_epoch("2026-07-01T00:00:00Z") + 3600   # 1h in, well before the 12h deadline
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(glog),
           "INTERVAL": "0", "SUPERVISOR_NOW_EPOCH": str(now), "SUPERVISOR_MAX_TICKS": "10"}
    r = bash(f'bash ./supervise_run.sh --manifest "{mf}" --dest "{tmp_path}/out"', env=env)
    assert r.returncode == 10, r.stderr
    log = glog.read_text()
    assert "instances stop r1-coord" in log and "instances stop r1-w-0" in log and "instances stop r1-api-0" in log
    assert "instances delete" not in log
    assert (tmp_path / "out" / "artifact.txt").exists()      # pulled to exactly --dest
    assert (tmp_path / ".runs" / "r1" / "verdict").exists()


def test_loop_consecutive_probe_errors_terminate_2(tmp_path):
    mf = _write_manifest(tmp_path, created_at="2026-07-01T00:00:00Z", max_run_duration="12h")
    fake = tmp_path / "gcloud"
    fake.write_text('#!/usr/bin/env bash\necho "GCLOUD $*" >> "$GCLOUD_LOG"\n'
                    'if [[ "$*" == *"curl -fsS"* ]]; then exit 1; fi\nexit 0\n')
    fake.chmod(0o755)
    glog = tmp_path / "g.log"
    glog.write_text("")   # pre-create so the assertion reads cleanly even if flow changes
    now = iso_epoch("2026-07-01T00:00:00Z") + 3600
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(glog),
           "INTERVAL": "0", "SUPERVISOR_NOW_EPOCH": str(now), "SUPERVISOR_MAX_TICKS": "20"}
    r = bash(f'bash ./supervise_run.sh --manifest "{mf}" --dest "{tmp_path}/out"', env=env)
    assert r.returncode == 2, r.stderr
    assert "instances stop" in glog.read_text()      # abnormal terminate STOPs even on error


def test_loop_max_ticks_exits_nonterminal(tmp_path):
    mf = _write_manifest(tmp_path, created_at="2026-07-01T00:00:00Z", max_run_duration="12h")
    _fake_gcloud_integration(tmp_path, statuses=[_status(6, verified=3, running=3)])  # never completes
    glog = tmp_path / 'g.log'
    now = iso_epoch("2026-07-01T00:00:00Z") + 3600
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(glog),
           "INTERVAL": "0", "SUPERVISOR_NOW_EPOCH": str(now),
           "STALL_MINUTES": "60", "SUPERVISOR_MAX_TICKS": "3"}
    r = bash(f'bash ./supervise_run.sh --manifest "{mf}" --dest "{tmp_path}/out"', env=env)
    assert r.returncode == 0, r.stderr           # bounded loop, non-terminal
    assert glog.read_text().count("curl -fsS") == 3   # exactly MAX_TICKS probe ticks ran


def test_once_flag_single_tick(tmp_path):
    mf = _write_manifest(tmp_path, created_at="2026-07-01T00:00:00Z", max_run_duration="12h")
    _fake_gcloud_integration(tmp_path, statuses=[_status(6, verified=6)])
    now = iso_epoch("2026-07-01T00:00:00Z") + 3600
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(tmp_path / 'g.log'),
           "SUPERVISOR_NOW_EPOCH": str(now)}
    r = bash(f'bash ./supervise_run.sh --once --manifest "{mf}" --dest "{tmp_path}/out"', env=env)
    assert r.returncode == 10, r.stderr
