"""Billing exhaustion parks the provider instead of
ending the run's episodes.

Agents raise BillingExhausted for an exhausted account or a spend cap. The
killed unit is uploaded as resumable (billing never counts against
MAX_RESUMES) and the coordinator parks every model group on that provider:
no new units, and workers park their in-flight episodes at the next query
boundary as resumable. Other providers keep running. ``coordinator-unpark``
releases a provider after a top-up.
"""

from __future__ import annotations

import io
import json
import threading
import urllib.error
from pathlib import Path

import pytest

from interface.agents.billing import BillingExhausted
from interface.agents.reply import Reply
from interface.loader import default_maze_path
from interface.smoke_tests.plans import v01_empty_room_trajectory
from scripts import distributed_run_pipeline as drp
from scripts.distributed_run_pipeline import CoordinatorStore, prepare_job, run_lockstep_worker, run_worker_loop

_REPO = Path(__file__).resolve().parents[1]
_MANIFEST = _REPO / "gridworld" / "fixtures" / "manifest.json"
PLAN = v01_empty_room_trajectory()


# --------------------------------------------------------------------------- #
# Agents classify billing errors
# --------------------------------------------------------------------------- #
def _http_error(code: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://api.example", code, "err", {}, io.BytesIO(body.encode()))


def _raising_urlopen(error):
    def urlopen(*_args, **_kwargs):
        raise error
    return urlopen


def test_claude_low_credit_is_billing(monkeypatch):
    from interface.agents import claude

    body = '{"type":"error","error":{"type":"invalid_request_error","message":"Your credit balance is too low to access the Anthropic API."}}'
    monkeypatch.setattr(claude.urllib.request, "urlopen", _raising_urlopen(_http_error(400, body)))
    with pytest.raises(BillingExhausted) as exc:
        claude._post_messages("k", model="claude-opus-5-5", max_tokens=8, temperature=0.0, system=None,
                              messages=[{"role": "user", "content": "hi"}], timeout=5, max_attempts=1)
    assert exc.value.provider == "claude"


def test_claude_other_400s_are_not_billing(monkeypatch):
    from interface.agents import claude

    body = '{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long"}}'
    monkeypatch.setattr(claude.urllib.request, "urlopen", _raising_urlopen(_http_error(400, body)))
    with pytest.raises(RuntimeError) as exc:
        claude._post_messages("k", model="claude-opus-5-5", max_tokens=8, temperature=0.0, system=None,
                              messages=[{"role": "user", "content": "hi"}], timeout=5, max_attempts=1)
    assert not isinstance(exc.value, BillingExhausted)


def test_kimi_insufficient_balance_is_billing(monkeypatch):
    from interface.agents import kimi_k26

    body = '{"error":{"message":"Your account is suspended due to insufficient balance","type":"exceeded_current_quota_error"}}'
    monkeypatch.setattr(kimi_k26.urllib.request, "urlopen", _raising_urlopen(_http_error(429, body)))
    with pytest.raises(BillingExhausted) as exc:
        kimi_k26._post_chat_completions("k", model="kimi-k2.6", max_tokens=8, temperature=1.0,
                                        messages=[{"role": "user", "content": "hi"}], timeout=5,
                                        enable_thinking=False, max_attempts=1)
    assert exc.value.provider == "kimi"


def test_openai_quota_and_spend_cap_are_billing():
    from interface.agents.openai_agent import OpenAISpendCapReached

    assert issubclass(OpenAISpendCapReached, BillingExhausted)
    assert OpenAISpendCapReached("cap $5 reached").provider == "openai"


def test_a_billing_error_ends_the_episode_with_a_billing_end_reason():
    from interface.agents.openai_agent import OpenAISpendCapReached
    from pipeline.run_stage3 import build_episode_runner
    from interface.config import ExperimentConfig

    runner = build_episode_runner(default_maze_path("V01_empty_room.json"), ExperimentConfig(), 0)

    def agent(messages):
        raise OpenAISpendCapReached("cap $5 reached")

    result = runner.run(agent, verbose=False)
    assert result["end_reason"].startswith("agent_error:BillingExhausted: openai: cap $5 reached")


# --------------------------------------------------------------------------- #
# Coordinator parks the provider, unpark releases it
# --------------------------------------------------------------------------- #
def _two_provider_job(tmp_path: Path, seeds=(0,)) -> Path:
    cfg = tmp_path / "rc.json"
    task = str(default_maze_path("V01_empty_room.json"))
    cfg.write_text(json.dumps({"models": {
        "claude_a": {"provider": "claude", "model": "claude-stub", "group": "claude-api", "tasks": [task]},
        "kimi_a": {"provider": "kimi", "model": "kimi-stub", "group": "kimi-api", "tasks": [task]},
    }}), encoding="utf-8")
    artifacts = tmp_path / "coordinator"
    prepare_job(run_config_path=cfg, manifest_path=_MANIFEST, seeds=list(seeds), conditions=None,
                artifacts_root=artifacts, run_set_id="park", difficulty_max_static_score=1000.0)
    return artifacts


class _Agent:
    def __init__(self, fail_at=None, error=None, gate=None):
        self._actions = iter(PLAN)
        self._fail_at, self._error, self._gate = fail_at, error, gate
        self.calls = 0
        self.last_usage = None

    def __call__(self, messages):
        self.calls += 1
        if self._gate is not None:
            self._gate(self.calls)
        if self._fail_at is not None and self.calls >= self._fail_at:
            raise self._error
        self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
        return f"FINAL_OUTPUT: {next(self._actions, 'DONE')}"


def _kill_claude_unit_with_billing(store: CoordinatorStore, tmp_path: Path) -> dict:
    store.register({"worker_id": "c1", "capabilities": {"model_group": "claude-api"}})
    unit = store.assign("c1", {"model_group": "claude-api"})["unit"]
    agent = _Agent(3, BillingExhausted("claude", "credit balance too low"))
    drp.run_assigned_unit(unit, artifacts_root=tmp_path / "c1", agent_factory=lambda k, c: (agent, "claude-stub"))
    result = store.upload("c1", unit["unit_id"], drp.package_run_archive(unit, artifacts_root=tmp_path / "c1").read_bytes())
    assert result["status"] == "resumable" and result["parked"] is True
    return unit


def test_a_billing_kill_parks_its_provider_and_only_its_provider(tmp_path):
    store = CoordinatorStore(_two_provider_job(tmp_path))
    _kill_claude_unit_with_billing(store, tmp_path)
    parked = store.status()["parked_providers"]
    assert list(parked) == ["claude"] and parked["claude"]["groups"] == ["claude-api"]
    assert "credit balance" in parked["claude"]["reason"]

    blocked = store.assign("c2", {"model_group": "claude-api"})
    assert blocked["unit"] is None and blocked["parked"] is True
    assert store.heartbeat("c2")["park"] is True
    kimi = store.assign("k1", {"model_group": "kimi-api"})["unit"]
    assert kimi is not None and kimi["provider"] == "kimi"
    assert store.heartbeat("k1", kimi["unit_id"])["park"] is False


def test_unpark_releases_the_provider_and_its_resumable_units(tmp_path):
    store = CoordinatorStore(_two_provider_job(tmp_path))
    unit = _kill_claude_unit_with_billing(store, tmp_path)
    assert store.unpark("claude")["groups"] == ["claude-api"]
    assert store.status()["parked_providers"] == {}
    again = store.assign("c2", {"model_group": "claude-api"})["unit"]
    assert again["unit_id"] == unit["unit_id"] and "resume" in again


def test_coordinator_unpark_role(tmp_path, capsys):
    artifacts = _two_provider_job(tmp_path)
    store = CoordinatorStore(artifacts)
    _kill_claude_unit_with_billing(store, tmp_path)
    from scripts.run_pipeline import main

    main(["--distributed-role", "coordinator-unpark", "--artifacts-root", str(artifacts), "--provider", "claude"])
    assert CoordinatorStore(artifacts).status()["parked_providers"] == {}
    assert "claude" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Workers park
# --------------------------------------------------------------------------- #
class _StoreClient:
    """The in-process store behind the worker loop's client API (upload takes a path)."""

    def __init__(self, store: CoordinatorStore):
        self._store = store

    def __getattr__(self, name):
        return getattr(self._store, name)

    def upload(self, worker_id, unit_id, archive_path):
        return self._store.upload(worker_id, unit_id, Path(archive_path).read_bytes())


def test_a_serial_worker_stops_taking_units_once_its_provider_is_parked(tmp_path):
    artifacts = _two_provider_job(tmp_path, seeds=(0, 1))
    store = CoordinatorStore(artifacts)
    agents = iter([_Agent(3, BillingExhausted("claude", "credit balance too low")), _Agent()])
    completed = run_worker_loop(coordinator_url="unused", artifacts_root=tmp_path / "w",
                                capabilities={"model_group": "claude-api"},
                                worker_state_path=tmp_path / "w" / "state.json", client=_StoreClient(store),
                                agent_factory=lambda k, c: (next(agents), "claude-stub"),
                                poll_interval_seconds=0.01, heartbeat_interval_seconds=0.01, drain=True)
    statuses = sorted(s["status"] for uid, s in store.load_state()["units"].items()
                      if uid in {u["unit_id"] for u in store.load_plan()["units"] if u["provider"] == "claude"})
    assert statuses == ["pending", "resumable"]  # parked after the first: the second was never taken
    assert completed is True


def test_an_in_flight_episode_parks_at_its_next_query_when_the_provider_is_parked(tmp_path):
    store = CoordinatorStore(_two_provider_job(tmp_path, seeds=(0, 1)))
    store.register({"worker_id": "c2", "capabilities": {"model_group": "claude-api"}})
    unit = store.assign("c2", {"model_group": "claude-api"})["unit"]
    park = drp._Park()

    def gate(call):
        if call == 3:
            park.event.set()  # another worker's billing kill reached us via heartbeat

    agent = _Agent(gate=gate)
    drp.run_assigned_unit(unit, artifacts_root=tmp_path / "c2", agent_factory=lambda k, c: (agent, "claude-stub"),
                          park=park)
    episode = json.loads((tmp_path / "c2" / unit["run_dir"] / "episode.json").read_text())
    assert episode["end_reason"].startswith("agent_error:BillingExhausted: claude: parked")
    assert episode["query_count"] == 3  # the park arrived during query 3; query 4 is refused


class _BatchAgent:
    def __init__(self, fail_round=None, error=None):
        self._fail_round, self._error = fail_round, error
        self.rounds = 0
        self._cursor: dict[int, int] = {}

    def generate_batch(self, batch):
        self.rounds += 1
        if self._fail_round is not None and self.rounds >= self._fail_round:
            raise self._error
        usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
        replies = []
        for i, _ in enumerate(batch):
            step = self.rounds - 1
            replies.append(Reply(text=f"FINAL_OUTPUT: {PLAN[step] if step < len(PLAN) else 'DONE'}", usage=usage))
        return replies


def test_a_lockstep_batch_billing_error_uploads_partials_as_resumable_and_parks(tmp_path):
    artifacts = _two_provider_job(tmp_path, seeds=(0, 1, 2))
    store = CoordinatorStore(artifacts)
    agent = _BatchAgent(3, BillingExhausted("claude", "credit balance too low"))
    result = run_lockstep_worker(artifacts_root=tmp_path / "lock", capabilities={"model_group": "claude-api"},
                                 max_batches=2, client=store, agent_factory=lambda k, c: (agent, "claude-stub"))
    claude_ids = [u["unit_id"] for u in store.load_plan()["units"] if u["provider"] == "claude"]
    statuses = sorted(store.load_state()["units"][uid]["status"] for uid in claude_ids)
    assert statuses == ["pending", "resumable", "resumable"]  # two parked mid-episode, the third never started
    assert sorted(result["parked"]) == sorted(uid for uid in claude_ids
                                              if store.load_state()["units"][uid]["status"] == "resumable")
    assert list(store.status()["parked_providers"]) == ["claude"]
    for uid in claude_ids:
        state = store.load_state()["units"][uid]
        if state["status"] == "resumable":
            assert state["infra_kill"]["query_count"] == 2
