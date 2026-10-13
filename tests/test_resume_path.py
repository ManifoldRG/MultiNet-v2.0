"""next-run-fixes item 1: a resumable unit resumes on a worker through the
replay-verify gate and finishes as one episode with resume provenance.

The worker gets the killed run's archive, replays every recorded action
against a fresh backend (refusing on divergence), and continues with the agent.
It refuses automatically across code SHAs: replay checks the environment, not
the prompts, so that stays a manual resume_from_archive job. Locally,
run_pipeline resumes killed episodes only with --resume-killed.
"""

from __future__ import annotations

import base64
import io
import json
import tarfile
from pathlib import Path

import pytest

from interface.episode_log import load_query_messages
from interface.loader import default_maze_path
from interface.smoke_tests.plans import v01_empty_room_trajectory
from pipeline import resume as resume_mod
from scorer.io import load_json
from scripts.distributed_run_pipeline import (
    CoordinatorStore,
    package_run_archive,
    prepare_job,
    run_assigned_unit,
)
from scripts.run_pipeline import run_from_config

_REPO = Path(__file__).resolve().parents[1]
_MANIFEST = _REPO / "gridworld" / "fixtures" / "manifest.json"
PLAN = v01_empty_room_trajectory()
QUERIES = len(PLAN) - 1  # stepping onto the goal ends the episode: DONE is never asked for
KILL_AT = 3  # the third query dies: two steps survive


def _rc(tmp_path: Path) -> Path:
    cfg = tmp_path / "rc.json"
    cfg.write_text(json.dumps({
        "experiment_config": {"progress_stall_k": 30},
        "models": {"stub": {"provider": "claude", "model": "stub-model", "group": "stub",
                            "tasks": [str(default_maze_path("V01_empty_room.json"))]}},
    }), encoding="utf-8")
    return cfg


class _Agent:
    def __init__(self, actions, fail_at=None):
        self._actions = iter(actions)
        self._fail_at = fail_at
        self.calls = 0
        self.last_usage = None

    def __call__(self, messages):
        self.calls += 1
        if self._fail_at is not None and self.calls >= self._fail_at:
            raise RuntimeError("connection reset by peer")
        self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
        return f"FINAL_OUTPUT: {next(self._actions, 'DONE')}"


def _killed_unit(tmp_path: Path) -> tuple[CoordinatorStore, dict]:
    store = _kill_one(tmp_path)
    resumed = store.assign("w2", {"model_group": "stub"})["unit"]
    assert "resume" in resumed
    return store, resumed


def _kill_one(tmp_path: Path) -> CoordinatorStore:
    artifacts = tmp_path / "coordinator"
    prepare_job(run_config_path=_rc(tmp_path), manifest_path=_MANIFEST, seeds=[0], conditions=None,
                artifacts_root=artifacts, run_set_id="res", difficulty_max_static_score=1000.0)
    store = CoordinatorStore(artifacts)
    store.register({"worker_id": "w1", "capabilities": {"model_group": "stub"}})
    unit = store.assign("w1", {"model_group": "stub"})["unit"]
    root = tmp_path / "w1"
    run_assigned_unit(unit, artifacts_root=root, agent_factory=lambda k, c: (_Agent(PLAN, KILL_AT), "stub-model"))
    assert store.upload("w1", unit["unit_id"], package_run_archive(unit, artifacts_root=root).read_bytes())["status"] == "resumable"
    return store


def test_a_resumable_unit_resumes_on_another_worker_and_verifies(tmp_path):
    store, unit = _killed_unit(tmp_path)
    agent = _Agent(PLAN[KILL_AT - 1:])
    root = tmp_path / "w2"
    run_assigned_unit(unit, artifacts_root=root, agent_factory=lambda k, c: (agent, "stub-model"))
    assert agent.calls == QUERIES - (KILL_AT - 1)  # only the rest of the episode is paid for

    run_dir = root / unit["run_dir"]
    episode = load_json(run_dir / "episode.json")
    assert episode["success"] is True and episode["end_reason"] == "success"
    assert episode["steps_used"] == QUERIES and episode["query_count"] == QUERIES
    assert episode["resumed_from"]["killed_at_query"] == KILL_AT - 1
    assert episode["resumed_from"]["end_reason"].startswith("agent_error:RuntimeError")
    resume = episode["resume"]
    assert resume["replay_verified"] is True and resume["resume_query_index"] == KILL_AT - 1
    assert resume["as_run_sha"] == resume["resume_sha"] == resume_mod.current_code_sha()
    assert resume["resume_count"] == 1
    # The killed run's images survive: query_001's logged request rebuilds with its image.
    sent = load_query_messages(run_dir / "queries" / "query_001")
    blocks = [b for m in sent if isinstance(m["content"], list) for b in m["content"]]
    assert any(b.get("type") == "image_url" for b in blocks)
    assert not any("missing from source archive" in str(b.get("text", "")) for b in blocks)
    side = load_json(run_dir / "run_inputs.json")
    assert side["inputs_hash"] == unit["episode_inputs_hash"] and side["resume"] == resume

    store.upload("w2", unit["unit_id"], package_run_archive(unit, artifacts_root=root).read_bytes())
    state = store.load_state()["units"][unit["unit_id"]]
    assert state["status"] == "verified" and store.status()["infra_killed"] == []


def test_resume_refuses_another_code_sha_and_says_it_is_a_manual_job(tmp_path, monkeypatch):
    _store, unit = _killed_unit(tmp_path)
    monkeypatch.setattr(resume_mod, "current_code_sha", lambda: "f" * 40)
    agent = _Agent(PLAN)
    with pytest.raises(resume_mod.ResumeRefused) as exc:
        run_assigned_unit(unit, artifacts_root=tmp_path / "w2", agent_factory=lambda k, c: (agent, "stub-model"))
    message = str(exc.value)
    assert "f" * 40 in message and "scripts/resume_from_archive.py" in message
    assert agent.calls == 0


def _tamper(unit: dict, edit) -> None:
    raw = base64.b64decode(unit["resume"]["archive_b64"])
    out = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as src, tarfile.open(fileobj=out, mode="w:gz") as dst:
        for member in src.getmembers():
            data = src.extractfile(member).read() if member.isfile() else None
            if member.name == "episode.json":
                episode = json.loads(data)
                edit(episode)
                data = json.dumps(episode).encode()
                member.size = len(data)
            dst.addfile(member, io.BytesIO(data) if data is not None else None)
    unit["resume"]["archive_b64"] = base64.b64encode(out.getvalue()).decode()


def test_resume_refuses_a_replay_that_diverges(tmp_path):
    _store, unit = _killed_unit(tmp_path)

    def move_the_agent(episode):
        step = next(r for r in episode["transcript"] if r.get("kind") == "step")
        step["position_after_row_col"] = [5, 5]

    _tamper(unit, move_the_agent)
    agent = _Agent(PLAN)
    with pytest.raises(resume_mod.ResumeRefused, match="diverge"):
        run_assigned_unit(unit, artifacts_root=tmp_path / "w2", agent_factory=lambda k, c: (agent, "stub-model"))
    assert agent.calls == 0


def test_a_refused_resume_fails_the_unit_terminally(tmp_path, monkeypatch):
    store, unit = _killed_unit(tmp_path)
    monkeypatch.setattr(resume_mod, "current_code_sha", lambda: "f" * 40)
    store.fail("w2", unit["unit_id"], "pipeline.resume.ResumeRefused: other sha")
    assert store.assign("w3", {"model_group": "stub"})["unit"] is None
    [row] = store.status()["infra_killed"]
    assert row["status"] == "failed"


def _local(tmp_path: Path, agent, **kwargs) -> Path:
    artifacts = tmp_path / "local"
    run_from_config(run_config_path=_rc(tmp_path), manifest_path=_MANIFEST, seeds=[0], artifacts_root=artifacts,
                    run_set_id="loc", agent_factory=lambda n, c: (agent, c["model"]),
                    difficulty_max_static_score=1000.0, **kwargs)
    [episode] = artifacts.rglob("episode.json")
    return episode


def test_local_rerun_keeps_a_killed_episode_unless_resume_killed(tmp_path):
    episode_path = _local(tmp_path, _Agent(PLAN, KILL_AT))
    assert load_json(episode_path)["end_reason"].startswith("agent_error:")

    idle = _Agent(PLAN)
    _local(tmp_path, idle)  # default: the cached killed episode stands, nothing is spent
    assert idle.calls == 0 and load_json(episode_path)["end_reason"].startswith("agent_error:")

    rest = _Agent(PLAN[KILL_AT - 1:])
    _local(tmp_path, rest, resume_killed=True)
    episode = load_json(episode_path)
    assert episode["end_reason"] == "success" and rest.calls == QUERIES - (KILL_AT - 1)
    assert episode["resume"]["replay_verified"] is True


class _BatchAgent:
    def __init__(self, actions):
        self._actions = iter(actions)
        self.calls = 0

    def generate_batch(self, batch):
        from interface.agents.reply import Reply

        self.calls += len(batch)
        usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
        return [Reply(text=f"FINAL_OUTPUT: {next(self._actions, 'DONE')}", usage=usage) for _ in batch]


def test_a_lockstep_worker_resumes_a_unit_killed_on_a_serial_worker(tmp_path):
    from scripts.distributed_run_pipeline import run_lockstep_worker

    store = _kill_one(tmp_path)
    agent = _BatchAgent(PLAN[KILL_AT - 1:])
    result = run_lockstep_worker(artifacts_root=tmp_path / "lockstep", capabilities={"model_group": "stub"},
                                 max_batches=1, client=store, agent_factory=lambda k, c: (agent, "stub-model"))
    assert result["completed"] == 1 and not result["failed"]
    assert agent.calls == QUERIES - (KILL_AT - 1)
    [unit] = store.load_plan()["units"]
    assert store.load_state()["units"][unit["unit_id"]]["status"] == "verified"
    episode = load_json(store.artifacts_root / unit["run_dir"] / "episode.json")
    assert episode["end_reason"] == "success" and episode["resume"]["replay_verified"] is True
    assert load_json(store.artifacts_root / unit["run_dir"] / "run_inputs.json")["resume"] == episode["resume"]


def test_a_lockstep_worker_fails_a_refused_resume_terminally(tmp_path, monkeypatch):
    from scripts.distributed_run_pipeline import run_lockstep_worker

    store = _kill_one(tmp_path)
    monkeypatch.setattr(resume_mod, "current_code_sha", lambda: "f" * 40)
    agent = _BatchAgent(PLAN)
    result = run_lockstep_worker(artifacts_root=tmp_path / "lockstep", capabilities={"model_group": "stub"},
                                 max_batches=1, client=store, agent_factory=lambda k, c: (agent, "stub-model"))
    assert result["completed"] == 0 and len(result["failed"]) == 1 and agent.calls == 0
    [row] = store.status()["infra_killed"]
    assert row["status"] == "failed"
    assert store.assign("w9", {"model_group": "stub"})["unit"] is None
