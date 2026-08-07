"""Bash tests for sweep_run.sh (the reused-fleet batch sequencer).

All cloud/git are stubbed with fakes on PATH; no real GPU/cloud, ever. The
sequencer STOPs and never deletes, and fails closed on empty egress.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def bash(snippet: str, env: dict | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess:
    e = os.environ.copy()
    e.pop("MAX_RUN_DURATION", None)
    if env:
        e.update(env)
    return subprocess.run(["bash", "-c", snippet], capture_output=True, text=True,
                          cwd=str(cwd or REPO), env=e)


def _fake_bin(d: Path, name: str, body: str) -> None:
    p = d / name
    p.write_text("#!/usr/bin/env bash\n" + body)
    p.chmod(0o755)


def _fake_gcloud_logging(tmp_path: Path) -> None:
    _fake_bin(tmp_path, "gcloud", 'echo "GCLOUD $*" >> "$GCLOUD_LOG"\nexit 0\n')


def _fake_gcloud_scp_lands(tmp_path: Path) -> None:
    # Logs every call; on `compute scp` it writes a REAL tarball at the destination
    # (last arg = $dest/egress.tgz) containing landed.txt, so finalize's local
    # `tar xzf` extracts it and the egress gate sees a non-empty dest.
    _fake_bin(tmp_path, "gcloud",
              'echo "GCLOUD $*" >> "$GCLOUD_LOG"\n'
              'if [[ "$1 $2" == "compute scp" ]]; then for a in "$@"; do d="$a"; done; '
              'td=$(mktemp -d); echo x > "$td/landed.txt"; mkdir -p "$(dirname "$d")"; '
              'tar czf "$d" -C "$td" . 2>/dev/null; rm -rf "$td"; fi\n'
              'exit 0\n')


def _write_manifest(runs: Path, sweep_id: str, workers: list[dict]) -> Path:
    d = runs / sweep_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({
        "run_id": sweep_id,
        "zone": "z1",
        "coordinator": {"name": f"{sweep_id}-coord", "internal_ip": "10.0.0.2"},
        "workers": workers,
        "created_at": "2026-07-03T00:00:00Z",
        "max_run_duration": "120h",
        "artifacts_root_remote": f"artifacts/{sweep_id}",
    }))
    return d


FULL_FLEET = [
    {"name": "swp-qwen36-27b-0", "kind": "gpu", "model_group": "qwen36-27b"},
    {"name": "swp-qwen36-27b-1", "kind": "gpu", "model_group": "qwen36-27b"},
    {"name": "swp-qwen36-27b-2", "kind": "gpu", "model_group": "qwen36-27b"},
    {"name": "swp-kimi-api-0", "kind": "api", "model_group": "kimi-api"},
    {"name": "swp-claude-api-0", "kind": "api", "model_group": "claude-api"},
]


def test_syntax_ok():
    assert bash("bash -n ./sweep_run.sh").returncode == 0


def test_teardown_stops_never_deletes(tmp_path):
    _fake_gcloud_logging(tmp_path)
    runs = tmp_path / ".runs"
    _write_manifest(runs, "swp", FULL_FLEET)
    log = tmp_path / "gcloud.log"; log.write_text("")
    r = bash(f'PATH="{tmp_path}:$PATH" GCLOUD_LOG="{log}" ZONE=z1 SWEEP_ID=swp '
             f'RUNS_DIR="{runs}" ./sweep_run.sh teardown', env={"MAX_RUN_DURATION": "120h"})
    assert r.returncode == 0, r.stderr
    calls = log.read_text()
    assert "instances stop" in calls
    assert "instances delete" not in calls               # never delete
    for vm in ["swp-coord", "swp-qwen36-27b-0", "swp-qwen36-27b-1",
               "swp-qwen36-27b-2", "swp-kimi-api-0", "swp-claude-api-0"]:
        assert vm in calls


def test_finalize_fails_closed_when_pull_lands_nothing(tmp_path):
    _fake_gcloud_logging(tmp_path)                        # scp is a no-op -> DEST stays empty
    runs = tmp_path / ".runs"
    _write_manifest(runs, "swp", [])
    log = tmp_path / "gcloud.log"; log.write_text("")
    dest = tmp_path / "egress"
    r = bash(f'PATH="{tmp_path}:$PATH" GCLOUD_LOG="{log}" ZONE=z1 SWEEP_ID=swp '
             f'RUNS_DIR="{runs}" DEST="{dest}" ./sweep_run.sh finalize-batch 7',
             env={"MAX_RUN_DURATION": "120h"})
    assert r.returncode == 40, (r.returncode, r.stderr)   # dedicated egress-failed code


def test_provision_exports_qwen_count_and_inits_state(tmp_path):
    # A fake launcher records the env it was handed and writes a manifest, like
    # the real launch_distributed.sh; provision then initialises tracking state.
    launcher = tmp_path / "fake_launch.sh"
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        'printf "QWC=%s RC=%s MF=%s RID=%s\\n" "$QWEN_WORKER_COUNT" "$RUN_CONFIG" "$MANIFEST" "$RUN_ID" > "$LAUNCH_LOG"\n'
        'mkdir -p "$RUNS_DIR/$RUN_ID"\n'
        'printf "{\\"run_id\\":\\"%s\\",\\"zone\\":\\"z1\\",\\"coordinator\\":{\\"name\\":\\"%s-coord\\"},\\"workers\\":[]}\\n" "$RUN_ID" "$RUN_ID" > "$RUNS_DIR/$RUN_ID/manifest.json"\n'
        "exit 0\n")
    launcher.chmod(0o755)
    runs = tmp_path / ".runs"
    llog = tmp_path / "launch.log"
    r = bash(f'SWEEP_ID=swpP RUNS_DIR="{runs}" SWEEP_LAUNCHER="{launcher}" '
             f'LAUNCH_LOG="{llog}" QWEN_WORKER_COUNT=3 ./sweep_run.sh provision',
             env={"MAX_RUN_DURATION": "120h"})
    assert r.returncode == 0, r.stderr
    assert "QWC=3" in llog.read_text()                   # topology fan-out passed to launcher
    # smoke fixture is the fleet-topology config for provision
    assert "smoke_qwen36_kimi_claude" in llog.read_text()
    st = json.loads((runs / "swpP" / "sweep_state.json").read_text())
    assert len(st["batches"]) == 11                      # smoke + 10 conditional
    assert st["batches"][0]["status"] == "running"       # batch 0 (smoke) started by provision


def test_provision_requires_max_run_duration(tmp_path):
    runs = tmp_path / ".runs"
    r = bash(f'SWEEP_ID=swpP RUNS_DIR="{runs}" ./sweep_run.sh provision')  # no MAX_RUN_DURATION
    assert r.returncode != 0
    assert "MAX_RUN_DURATION" in r.stderr


def test_next_batch_rearms_watchdog_never_creates_or_deletes(tmp_path):
    _fake_gcloud_logging(tmp_path)
    runs = tmp_path / ".runs"
    _write_manifest(runs, "swp", FULL_FLEET)
    log = tmp_path / "gcloud.log"; log.write_text("")
    # BATCH_CAP has no default: a 6h default once killed a legit ~7h massive
    # tail before egress. Batch-starting subcommands refuse to run without it.
    r = bash(f'PATH="{tmp_path}:$PATH" GCLOUD_LOG="{log}" ZONE=z1 SWEEP_ID=swp '
             f'RUNS_DIR="{runs}" ./sweep_run.sh next-batch 2', env={"MAX_RUN_DURATION": "120h"})
    assert r.returncode != 0
    assert "BATCH_CAP is required" in r.stderr

    r = bash(f'PATH="{tmp_path}:$PATH" GCLOUD_LOG="{log}" ZONE=z1 SWEEP_ID=swp '
             f'RUNS_DIR="{runs}" ./sweep_run.sh next-batch 2',
             env={"MAX_RUN_DURATION": "120h", "BATCH_CAP": "9h"})
    assert r.returncode == 0, r.stderr
    calls = log.read_text()
    assert "shutdown -h" in calls                        # watchdog re-armed on the reused VMs
    assert "instances start" in calls                    # the e2 api runners are STARTed
    assert "instances create" not in calls               # never create
    assert "instances delete" not in calls               # never delete


def test_finalize_batch0_reads_sweep_id_artifacts_and_egresses(tmp_path):
    # Regression for the batch-0 egress mismatch: provision starts the smoke under
    # RUN_ID=$SWEEP_ID, so finalize-batch 0 must read artifacts/$SWEEP_ID (NOT the
    # literal batch run_id "smoke"), egress to DEST/smoke, and succeed.
    _fake_gcloud_scp_lands(tmp_path)
    runs = tmp_path / ".runs"
    _write_manifest(runs, "swp", FULL_FLEET)
    log = tmp_path / "gcloud.log"; log.write_text("")
    dest = tmp_path / "egress"
    r = bash(f'PATH="{tmp_path}:$PATH" GCLOUD_LOG="{log}" ZONE=z1 SWEEP_ID=swp '
             f'RUNS_DIR="{runs}" DEST="{dest}" ./sweep_run.sh finalize-batch 0',
             env={"MAX_RUN_DURATION": "120h"})
    assert r.returncode == 0, r.stderr                   # successful smoke egresses (no false 40)
    calls = log.read_text()
    assert "artifacts/swp" in calls                      # reads the $SWEEP_ID namespace (tar -C)
    assert "artifacts/smoke" not in calls                # never the literal run_id path
    assert (dest / "smoke" / "landed.txt").exists()      # dest folder = readable run_id


def test_publish_nothing_to_commit_is_success(tmp_path):
    # A clean index (idempotent re-publish, or a retry after commit-but-push-failed)
    # must exit 0 without committing — publish is durability, a false failure misleads.
    _fake_bin(tmp_path, "git",
              'echo "GIT $*" >> "$GIT_LOG"\n'
              'if [[ "$1" == "diff" ]]; then exit 0; fi\n'   # clean tree / clean index
              'exit 0\n')
    dest = tmp_path / "egress"; (dest / "cond_prompt").mkdir(parents=True)
    (dest / "cond_prompt" / "episode_runs.jsonl").write_text('{"x":1}\n')
    results = tmp_path / "Multinet-v2-results"; results.mkdir()
    glog = tmp_path / "git.log"; glog.write_text("")
    r = bash(f'PATH="{tmp_path}:$PATH" GIT_LOG="{glog}" SWEEP_ID=swpX DEST="{dest}" '
             f'RESULTS_REPO="{results}" ./sweep_run.sh publish cond_prompt',
             env={"MAX_RUN_DURATION": "120h"})
    assert r.returncode == 0, r.stderr
    gl = glog.read_text()
    assert "add -A" in gl                                # add ran
    assert "commit" not in gl                            # ...but commit did NOT (nothing new)


def test_stop_apis_stops_only_api_vms(tmp_path):
    _fake_gcloud_logging(tmp_path)
    runs = tmp_path / ".runs"
    _write_manifest(runs, "swp", FULL_FLEET)
    log = tmp_path / "gcloud.log"; log.write_text("")
    r = bash(f'PATH="{tmp_path}:$PATH" GCLOUD_LOG="{log}" ZONE=z1 SWEEP_ID=swp '
             f'RUNS_DIR="{runs}" ./sweep_run.sh stop-apis', env={"MAX_RUN_DURATION": "120h"})
    assert r.returncode == 0, r.stderr
    stops = [ln for ln in log.read_text().splitlines() if "instances stop" in ln]
    assert len(stops) == 2                               # exactly the two api-kind VMs
    assert any("swp-kimi-api-0" in ln for ln in stops)
    assert any("swp-claude-api-0" in ln for ln in stops)
    assert not any("qwen36-27b" in ln for ln in stops)   # gpu VMs never stopped here
    assert not any("swp-coord" in ln for ln in stops)    # coordinator never stopped here


def test_publish_excludes_png_and_runs_git_only_in_results_repo(tmp_path):
    # `diff --cached --quiet` -> exit 1 (dirty index / something staged) so the
    # nothing-to-commit guard does NOT short-circuit and the commit+push path runs.
    _fake_bin(tmp_path, "git",
              'echo "GIT[$(pwd)] $*" >> "$GIT_LOG"\n'
              'if [[ "$1" == "diff" ]]; then exit 1; fi\n'
              'exit 0\n')
    dest = tmp_path / "egress"
    (dest / "cond_prompt").mkdir(parents=True)
    (dest / "cond_prompt" / "episode_runs.jsonl").write_text('{"x":1}\n')
    (dest / "cond_prompt" / "maze.png").write_text("PNG-BYTES")
    results = tmp_path / "Multinet-v2-results"; results.mkdir()
    glog = tmp_path / "git.log"; glog.write_text("")
    r = bash(f'PATH="{tmp_path}:$PATH" GIT_LOG="{glog}" SWEEP_ID=swpX DEST="{dest}" '
             f'RESULTS_REPO="{results}" ./sweep_run.sh publish cond_prompt',
             env={"MAX_RUN_DURATION": "120h"})
    assert r.returncode == 0, r.stderr
    landed = results / "swpX" / "cond_prompt"
    assert (landed / "episode_runs.jsonl").exists()      # data mirrored
    assert not (landed / "maze.png").exists()            # *.png excluded
    gl = glog.read_text()
    assert "commit" in gl and "push" in gl
    git_lines = [ln for ln in gl.splitlines() if ln.startswith("GIT[")]
    assert git_lines
    assert all(ln.startswith(f"GIT[{results}]") for ln in git_lines)  # never the code repo


def test_results_repo_is_gitignored():
    # The results repo is a SEPARATE git repo; the code repo must never track it.
    assert bash("git check-ignore Multinet-v2-results/").returncode == 0


def test_status_renders_table(tmp_path):
    runs = tmp_path / ".runs"
    (runs / "swpS").mkdir(parents=True)
    bash(f'python3 -c "from scripts.sweep_state import init_state, save_state; '
         f"save_state('{runs}/swpS/sweep_state.json', init_state('swpS', '2026-07-03T00:00:00Z'))\"")
    r = bash(f'SWEEP_ID=swpS RUNS_DIR="{runs}" ./sweep_run.sh status', env={"MAX_RUN_DURATION": "120h"})
    assert r.returncode == 0, r.stderr
    assert "cond_baseline_thinking" in r.stdout          # a row per batch
