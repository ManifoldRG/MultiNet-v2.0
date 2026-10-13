"""Fixes from the fresh-context review of the resume/billing branch.

Each test pins a failure the first version allowed: billing on the real batch
transports, finalize scoring a killed partial mid-resume, a re-prepare wiping
resume state, an ungated crash checkpoint, set-ordered state breaking replay,
late parked uploads re-parking a provider, and the smaller ones below.
"""

from __future__ import annotations

import base64
import io
import json
import tarfile
from pathlib import Path

import pytest

from gridworld.baselines import plan_bfs_path
from interface.agents.billing import BillingExhausted, BillingParked
from interface.config import ExperimentConfig
from interface.loader import default_maze_path
from interface.parser import ACTION_ORDER
from interface.smoke_tests.plans import v01_empty_room_trajectory
from pipeline import resume as resume_mod
from pipeline.run_stage3 import run_episode
from scorer.io import load_json, task_spec_from_payload
from scripts import distributed_run_pipeline as drp
from scripts.distributed_run_pipeline import (
    CoordinatorStore,
    finalize_job,
    package_run_archive,
    prepare_job,
    run_assigned_unit,
    run_lockstep_worker,
)

_REPO = Path(__file__).resolve().parents[1]
_MANIFEST = _REPO / "gridworld" / "fixtures" / "manifest.json"
PLAN = v01_empty_room_trajectory()
V01 = default_maze_path("V01_empty_room.json")


class _Agent:
    def __init__(self, actions, fail_at=None, error=None):
        self._actions = iter(actions)
        self._fail_at, self._error = fail_at, error
        self.calls = 0
        self.last_usage = None

    def __call__(self, messages):
        self.calls += 1
        if self._fail_at is not None and self.calls >= self._fail_at:
            raise self._error
        self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
        return f"FINAL_OUTPUT: {next(self._actions, 'DONE')}"


def _job(tmp_path: Path, seeds=(0,), task=V01, run_config_name="rc.json", **extra) -> Path:
    cfg = tmp_path / run_config_name
    cfg.write_text(json.dumps({"models": {"stub": {"provider": "claude", "model": "stub-model", "group": "stub",
                                                    "tasks": [str(task)]}}, **extra}), encoding="utf-8")
    artifacts = tmp_path / "coordinator"
    prepare_job(run_config_path=cfg, manifest_path=_MANIFEST, seeds=list(seeds), conditions=None,
                artifacts_root=artifacts, run_set_id="fix", difficulty_max_static_score=1000.0)
    return cfg


def _kill(store: CoordinatorStore, tmp_path: Path, worker: str, agent: _Agent) -> dict:
    store.register({"worker_id": worker, "capabilities": {"model_group": "stub"}})
    unit = store.assign(worker, {"model_group": "stub"})["unit"]
    root = tmp_path / worker
    run_assigned_unit(unit, artifacts_root=root, agent_factory=lambda k, c: (agent, "stub-model"))
    store.upload(worker, unit["unit_id"], package_run_archive(unit, artifacts_root=root).read_bytes())
    return unit


# --------------------------------------------------------------------------- #
# 1 (critical): billing on the real batch transports
# --------------------------------------------------------------------------- #
def test_anthropic_batch_low_credit_past_the_wait_is_billing(monkeypatch):
    from interface.agents import anthropic_batch

    def refuse(*_args, **_kwargs):
        raise anthropic_batch.AnthropicBatchHTTPError(400, "Your credit balance is too low to access the Anthropic API.")

    monkeypatch.setattr(anthropic_batch, "_request_json", refuse)
    with pytest.raises(BillingExhausted) as exc:
        anthropic_batch._create_batch("u", [], api_key="k", credit_wait_s=0.0, credit_poll_s=0.0)
    assert exc.value.provider == "claude"


def test_moonshot_batch_insufficient_balance_is_billing(monkeypatch):
    import urllib.error

    from interface.agents import moonshot_batch

    def urlopen(*_args, **_kwargs):
        raise urllib.error.HTTPError("u", 429, "err", {}, io.BytesIO(b'{"error":{"message":"insufficient balance"}}'))

    monkeypatch.setattr(moonshot_batch.urllib.request, "urlopen", urlopen)
    with pytest.raises(BillingExhausted) as exc:
        moonshot_batch._http("https://api.moonshot.ai/v1/batches", api_key="k", method="GET", max_attempts=1)
    assert exc.value.provider == "kimi"


def test_kimi_sync_fanout_lets_billing_through(monkeypatch):
    from interface.agents.kimi_k26 import KimiK26Agent, KimiK26Config

    monkeypatch.setenv("KIMI_BATCH_TRANSPORT", "sync")
    agent = KimiK26Agent(config=KimiK26Config(max_tokens=8), api_key="k")

    def broke(messages):
        raise BillingExhausted("kimi", "insufficient balance")

    monkeypatch.setattr(agent, "generate", broke)
    with pytest.raises(BillingExhausted):
        agent.generate_batch([[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]])


# --------------------------------------------------------------------------- #
# 2: finalize never scores a killed partial, whatever the unit's status now
# --------------------------------------------------------------------------- #
def test_finalize_refuses_while_a_killed_unit_is_being_resumed(tmp_path):
    _job(tmp_path)
    store = CoordinatorStore(tmp_path / "coordinator")
    unit = _kill(store, tmp_path, "w1", _Agent(PLAN, 3, RuntimeError("reset")))
    assert store.assign("w2", {"model_group": "stub"})["unit"]["unit_id"] == unit["unit_id"]  # now "assigned"
    with pytest.raises(RuntimeError, match="infra-killed"):
        finalize_job(artifacts_root=tmp_path / "coordinator", run_set_id="fix")
    result = finalize_job(artifacts_root=tmp_path / "coordinator", run_set_id="fix", allow_partial=True)
    assert result["run_count"] == 0 and [r["unit_id"] for r in result["infra_killed"]] == [unit["unit_id"]]
    assert store.load_state()["units"][unit["unit_id"]]["status"] != "verified"


# --------------------------------------------------------------------------- #
# 3: a same-job re-prepare keeps resume state and parked providers
# --------------------------------------------------------------------------- #
def test_a_reprepare_keeps_resumable_units_and_parked_providers(tmp_path):
    cfg = _job(tmp_path, seeds=(0, 1))
    store = CoordinatorStore(tmp_path / "coordinator")
    killed = _kill(store, tmp_path, "w1", _Agent(PLAN, 3, RuntimeError("reset")))
    in_flight = store.assign("w2", {"model_group": "stub"})["unit"]  # the other unit: fresh work first
    assert in_flight["unit_id"] != killed["unit_id"]
    resuming = store.assign("w3", {"model_group": "stub"})["unit"]
    assert resuming["unit_id"] == killed["unit_id"]
    state = store.load_state()
    state["parked_providers"] = {"claude": {"groups": ["stub"], "since": "t", "unit_id": "x", "reason": "r"}}
    store.save_state(state)

    prepare_job(run_config_path=cfg, manifest_path=_MANIFEST, seeds=[0, 1], conditions=None,
                artifacts_root=tmp_path / "coordinator", run_set_id="fix", difficulty_max_static_score=1000.0)
    after = CoordinatorStore(tmp_path / "coordinator").load_state()
    kept = after["units"][killed["unit_id"]]
    assert kept["status"] == "resumable" and kept["infra_kill"]["steps_used"] == 2 and kept["infra_kills"] == 1
    assert "resume_pending" not in kept
    assert after["units"][in_flight["unit_id"]]["status"] == "pending"  # no kill: a fresh rerun, as before
    assert list(after["parked_providers"]) == ["claude"]


# --------------------------------------------------------------------------- #
# 4: the crash checkpoint is gated, and a bad one never aborts the rerun
# --------------------------------------------------------------------------- #
def _crash(out: Path, config: ExperimentConfig, die_at: int = 4) -> None:
    with pytest.raises(KeyboardInterrupt):
        run_episode(V01, config, _Agent(PLAN, die_at, KeyboardInterrupt()), 0, out)
    assert (out / "checkpoint.json").exists()


def test_a_checkpoint_from_other_inputs_is_discarded_not_resumed(tmp_path):
    out = tmp_path / "run"
    _crash(out, ExperimentConfig(context_window="last_n"))
    fresh = _Agent(PLAN)
    episode = run_episode(V01, ExperimentConfig(context_window="text_summary"), fresh, 0, out)
    assert "resumed_from" not in episode and fresh.calls == len(PLAN) - 1


def test_a_checkpoint_from_other_code_is_discarded(tmp_path, monkeypatch):
    out = tmp_path / "run"
    _crash(out, ExperimentConfig())
    monkeypatch.setattr(resume_mod, "current_code_sha", lambda: "f" * 40)
    episode = run_episode(V01, ExperimentConfig(), _Agent(PLAN), 0, out)
    assert "resumed_from" not in episode


def test_a_checkpoint_that_does_not_replay_is_discarded_not_fatal(tmp_path):
    out = tmp_path / "run"
    _crash(out, ExperimentConfig())
    data = json.loads((out / "checkpoint.json").read_text())
    step = next(r for r in data["transcript"] if r.get("kind") == "step")
    step["position_after_row_col"] = [5, 5]
    (out / "checkpoint.json").write_text(json.dumps(data))
    episode = run_episode(V01, ExperimentConfig(), _Agent(PLAN), 0, out)
    assert episode["end_reason"] == "success" and "resumed_from" not in episode


def test_force_discards_a_leftover_checkpoint(tmp_path):
    out = tmp_path / "run"
    _crash(out, ExperimentConfig())
    episode = run_episode(V01, ExperimentConfig(), _Agent(PLAN), 0, out, fresh=True)
    assert "resumed_from" not in episode


def test_replay_ignores_the_order_of_set_valued_state_fields(tmp_path):
    """collected_keys is serialized from a set, so its order follows
    PYTHONHASHSEED. A replay on another process must not call that a divergence."""
    task = default_maze_path("V08_chain_kk.json")
    plan = plan_bfs_path(task_spec_from_payload(json.loads(Path(task).read_text())))
    actions = [ACTION_ORDER[a] for a in plan.actions]
    _job(tmp_path, task=task)
    store = CoordinatorStore(tmp_path / "coordinator")
    unit = _kill(store, tmp_path, "w1", _Agent(actions, len(actions) - 1, RuntimeError("reset")))
    resumed = store.assign("w2", {"model_group": "stub"})["unit"]
    raw = base64.b64decode(resumed["resume"]["archive_b64"])
    out = io.BytesIO()
    reordered = 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as src, tarfile.open(fileobj=out, mode="w:gz") as dst:
        for member in src.getmembers():
            data = src.extractfile(member).read() if member.isfile() else None
            if member.name == "episode.json":
                episode = json.loads(data)
                for rec in episode["transcript"]:
                    keys = (rec.get("state_after") or {}).get("collected_keys")
                    if keys and len(keys) > 1:
                        rec["state_after"]["collected_keys"] = list(reversed(keys))
                        reordered += 1
                data = json.dumps(episode).encode()
                member.size = len(data)
            dst.addfile(member, io.BytesIO(data) if data is not None else None)
    assert reordered > 0, "the plan must collect two keys before the kill"
    resumed["resume"]["archive_b64"] = base64.b64encode(out.getvalue()).decode()
    run_assigned_unit(resumed, artifacts_root=tmp_path / "w2",
                      agent_factory=lambda k, c: (_Agent(actions[len(actions) - 2:]), "stub-model"))
    episode = load_json(tmp_path / "w2" / unit["run_dir"] / "episode.json")
    assert episode["resume"]["replay_verified"] is True


# --------------------------------------------------------------------------- #
# 5: a late parked upload does not re-park an unparked provider
# --------------------------------------------------------------------------- #
def test_a_late_parked_upload_does_not_repark_the_provider(tmp_path):
    _job(tmp_path, seeds=(0, 1))
    store = CoordinatorStore(tmp_path / "coordinator")
    _kill(store, tmp_path, "w1", _Agent(PLAN, 3, BillingExhausted("claude", "credit balance too low")))
    assert list(store.status()["parked_providers"]) == ["claude"]
    store.unpark("claude")
    late = _kill(store, tmp_path, "w2", _Agent(PLAN, 3, BillingParked("claude")))  # in flight during the park
    assert store.load_state()["units"][late["unit_id"]]["status"] == "resumable"
    assert store.status()["parked_providers"] == {}


# --------------------------------------------------------------------------- #
# Minors
# --------------------------------------------------------------------------- #
def test_a_failed_checkpoint_or_ledger_write_never_ends_the_episode(tmp_path, monkeypatch):
    from interface import episode_checkpoint

    def disk_full(*_args, **_kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(episode_checkpoint, "save_checkpoint", disk_full)
    out = tmp_path / "run"
    out.mkdir()
    (out / "queries.jsonl").mkdir()  # the ledger cannot be appended to either
    episode = run_episode(V01, ExperimentConfig(), _Agent(PLAN), 0, out)
    assert episode["end_reason"] == "success" and episode["query_count"] == len(PLAN) - 1


def test_a_resumable_unit_whose_archive_is_gone_reruns_fresh(tmp_path):
    _job(tmp_path)
    store = CoordinatorStore(tmp_path / "coordinator")
    unit = _kill(store, tmp_path, "w1", _Agent(PLAN, 3, RuntimeError("reset")))
    (drp._uploads_root(tmp_path / "coordinator") / f"{unit['unit_id']}.tar.gz").unlink()
    again = store.assign("w2", {"model_group": "stub"})["unit"]
    assert again["unit_id"] == unit["unit_id"] and "resume" not in again


def test_a_stale_fail_does_not_clobber_a_resumable_unit(tmp_path):
    _job(tmp_path)
    store = CoordinatorStore(tmp_path / "coordinator")
    unit = _kill(store, tmp_path, "w1", _Agent(PLAN, 3, RuntimeError("reset")))
    store.fail("w1", unit["unit_id"], "upload timed out")  # the worker gave up after the upload landed
    assert store.load_state()["units"][unit["unit_id"]]["status"] == "resumable"


def test_api_client_groups_include_resumable_units(tmp_path):
    _job(tmp_path)
    store = CoordinatorStore(tmp_path / "coordinator")
    _kill(store, tmp_path, "w1", _Agent(PLAN, 3, RuntimeError("reset")))
    assert drp._api_client_groups(store.load_plan(), store.load_state()) == ["stub"]


def test_a_lockstep_worker_moves_on_past_a_refused_first_resume(tmp_path, monkeypatch):
    _job(tmp_path, seeds=(0, 1))
    store = CoordinatorStore(tmp_path / "coordinator")
    _kill(store, tmp_path, "w1", _Agent(PLAN, 3, RuntimeError("reset")))
    _kill(store, tmp_path, "w2", _Agent(PLAN, 3, RuntimeError("reset")))
    monkeypatch.setattr(resume_mod, "current_code_sha", lambda: "f" * 40)

    class _Batch:
        def generate_batch(self, batch):
            raise AssertionError("no model call: every resume is refused")

    result = run_lockstep_worker(artifacts_root=tmp_path / "lock", capabilities={"model_group": "stub"},
                                 max_batches=1, client=store, agent_factory=lambda k, c: (_Batch(), "stub-model"))
    assert len(result["failed"]) == 2


def test_dirty_trees_with_different_changes_get_different_shas():
    assert resume_mod._sha_with_dirt("abc", "diff one") != resume_mod._sha_with_dirt("abc", "diff two")
    assert resume_mod._sha_with_dirt("abc", "") == "abc"


def test_preflight_sums_the_kimi_caps_against_one_balance():
    from scripts import preflight_api

    rows = preflight_api.preflight(
        {"models": {"a": {"provider": "kimi", "spend_cap_usd": 20}, "b": {"provider": "kimi", "spend_cap_usd": 20}}},
        probe=lambda p, c: None, balance=lambda: 30.0)
    assert {r["status"] for r in rows} == {"failed"}
    assert "40.00" in rows[0]["detail"]
