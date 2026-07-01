"""Bash tests for lib/distributed_start.sh (coordinator/worker start recipes).
All cloud is stubbed: a fake `gcloud` on PATH records --command args and the
heredoc stdin. No real cloud or API calls."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def bash(snippet: str, env: dict | None = None) -> subprocess.CompletedProcess:
    e = os.environ.copy()
    if env:
        e.update(env)
    return subprocess.run(["bash", "-c", snippet], capture_output=True, text=True, cwd=REPO, env=e)


def _fake_gcloud_capture(tmp_path: Path) -> None:
    """Records `ARGS: ...` (all argv) and the heredoc stdin between STDIN markers."""
    fake = tmp_path / "gcloud"
    fake.write_text(
        '#!/usr/bin/env bash\n'
        'echo "ARGS: $*" >> "$GCLOUD_LOG"\n'
        'echo "STDIN_BEGIN" >> "$GCLOUD_LOG"\n'
        'cat >> "$GCLOUD_LOG"\n'
        'echo "STDIN_END" >> "$GCLOUD_LOG"\n'
        'exit 0\n'
    )
    fake.chmod(0o755)


def test_coordinator_prepares_serves_healthchecks(tmp_path):
    _fake_gcloud_capture(tmp_path)
    glog = tmp_path / "g.log"; glog.write_text("")
    snippet = (
        'source ./lib/distributed_start.sh; '
        'COORD=r1-coord; ZONE=z1; RUN_ID=r1; '
        'RUN_CONFIG=gridworld/fixtures/run_config.smoke_claude_sonnet.json; '
        'MANIFEST=gridworld/fixtures/manifest.smoke_eval.json; '
        'start_coordinator'
    )
    r = bash(snippet, env={"PATH": f"{tmp_path}:{os.environ['PATH']}", "GCLOUD_LOG": str(glog)})
    assert r.returncode == 0, r.stderr
    log = glog.read_text()
    # env injected on the --command; recipe body in the heredoc stdin
    assert "RUN_ID='r1'" in log
    assert "coordinator-prepare" in log
    assert "--run-config" in log and "run_config.smoke_claude_sonnet.json" in log
    assert "--manifest" in log and "manifest.smoke_eval.json" in log
    # the quoted heredoc keeps $RUN_ID literal (expanded remotely from the injected env)
    assert "--run-set-id" in log and "artifacts/$RUN_ID" in log
    assert "coordinator-serve" in log and "--host 0.0.0.0 --port 8765" in log
    assert "127.0.0.1:8765/status" in log
