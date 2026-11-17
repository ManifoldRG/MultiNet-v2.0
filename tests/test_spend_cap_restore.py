"""A spend cap survives a process restart.

The spend ledger lives in process memory. Before this fix a worker that exited
on its cap park (or crashed, or was restarted after an unpark) came back with a
$0 total and the full cap to spend again. Now the first time a process builds a
capped model's agent, it restores that model's total from the priced usage
already recorded in its run directories: finished episodes, crash checkpoints
and per-query usage ledgers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interface.agents import spend_cap
from interface.agents.spend_cap import SPEND
from interface.loader import default_maze_path
from scripts import distributed_run_pipeline as drp
from scripts.distributed_run_pipeline import CoordinatorStore, prepare_job

_REPO = Path(__file__).resolve().parents[1]
_MANIFEST = _REPO / "gridworld" / "fixtures" / "manifest.json"

# Opus 5.5 at (4.00, 0.20, 20.00) per 1M tokens: 1000 in + 400 out = $0.012.
_USAGE = {"input_tokens": 1000, "output_tokens": 400}
_ONE_QUERY_USD = (1000 * 4.00 + 400 * 20.00) / 1_000_000
_OPUS = {"provider": "claude", "model": "claude-opus-5-5", "spend_cap_usd": 0.01}


@pytest.fixture(autouse=True)
def _fresh_ledger():
    SPEND.reset()
    yield
    SPEND.reset()


class _Message:
    headers: dict = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps({
            "content": [{"type": "text", "text": "FINAL_OUTPUT: MOVE_FORWARD"}],
            "stop_reason": "end_turn",
            "usage": _USAGE,
        }).encode()


def _counting_urlopen(monkeypatch) -> list:
    import urllib.request

    calls: list = []

    def urlopen(req, timeout=None):
        calls.append(req)
        return _Message()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-never-sent")
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return calls


def test_a_restarted_worker_does_not_get_its_cap_back(tmp_path, monkeypatch):
    """Opus spends past its cap, parks, and the worker exits. After an unpark the
    worker starts again (a new process: empty ledger) on the same disk. It must
    refuse before sending a request, not spend a second cap."""
    calls = _counting_urlopen(monkeypatch)
    cfg = tmp_path / "rc.json"
    cfg.write_text(json.dumps({"models": {
        "opus": {**_OPUS, "group": "opus", "tasks": [str(default_maze_path("V01_empty_room.json"))]},
    }}), encoding="utf-8")
    artifacts = tmp_path / "coordinator"
    prepare_job(run_config_path=cfg, manifest_path=_MANIFEST, seeds=[0], conditions=None,
                artifacts_root=artifacts, run_set_id="restart", difficulty_max_static_score=1000.0)
    store = CoordinatorStore(artifacts)
    worker_root = tmp_path / "o1"

    unit = store.assign("o1", {"model_group": "opus"})["unit"]
    drp.run_assigned_unit(unit, artifacts_root=worker_root)  # one query, then the cap refuses
    up = store.upload("o1", unit["unit_id"], drp.package_run_archive(unit, artifacts_root=worker_root).read_bytes())
    assert up["status"] == "resumable" and up["parked"] is True
    assert len(calls) == 1

    # The parked worker exits; the operator unparks Opus and starts it again.
    SPEND.reset()                                   # a new process starts with an empty ledger
    store.unpark("claude", model="claude-opus-5-5")
    again = store.assign("o1", {"model_group": "opus"})["unit"]
    assert again is not None and again["unit_id"] == unit["unit_id"]
    drp.run_assigned_unit(again, artifacts_root=worker_root)

    assert len(calls) == 1, "the restarted worker spent past its cap again"
    assert SPEND.total("claude-opus-5-5") == pytest.approx(_ONE_QUERY_USD)
    up = store.upload("o1", again["unit_id"], drp.package_run_archive(again, artifacts_root=worker_root).read_bytes())
    assert up["status"] == "resumable" and up["parked"] is True


def test_a_cap_raised_after_the_restart_lets_the_worker_continue(tmp_path, monkeypatch):
    """The restored total counts against the cap in force: raising the cap is how
    an operator grants more budget, and the worker then spends only the difference."""
    calls = _counting_urlopen(monkeypatch)
    run = tmp_path / "runs" / "t" / "minigrid" / "claude-opus-5-5" / "seed_0" / "default"
    _write_episode(run, queries=1)
    spend_cap.restore_spend({**_OPUS, "spend_cap_usd": 1.0}, [run], batch=False)
    from interface.agents.claude import ClaudeAnthropicAgent, ClaudeAnthropicConfig

    agent = ClaudeAnthropicAgent(ClaudeAnthropicConfig(model="claude-opus-5-5", spend_cap_usd=1.0))
    agent.generate([{"role": "user", "content": "hi"}])
    assert len(calls) == 1
    assert SPEND.total("claude-opus-5-5") == pytest.approx(2 * _ONE_QUERY_USD)


# --------------------------------------------------------------------------- #
# What a run directory contributes
# --------------------------------------------------------------------------- #
def _query(i: int, usage=_USAGE) -> dict:
    return {"kind": "query", "query_index": i, "usage": dict(usage)}


def _write_episode(run: Path, *, queries: int, tier: str | None = None) -> None:
    run.mkdir(parents=True, exist_ok=True)
    transcript = [_query(i) for i in range(queries)] + [{"kind": "step"}]
    (run / "episode.json").write_text(json.dumps({"transcript": transcript}), encoding="utf-8")
    inputs = {"model_id": "claude-opus-5-5"}
    if tier:
        inputs["pricing_tier"] = tier
    (run / "run_inputs.json").write_text(json.dumps(inputs), encoding="utf-8")


def test_a_finished_episode_counts_every_query(tmp_path):
    run = tmp_path / "run"
    _write_episode(run, queries=3)
    assert spend_cap.recorded_spend_usd(run, _OPUS, batch=False) == pytest.approx(3 * _ONE_QUERY_USD)


def test_the_run_inputs_pricing_tier_wins_over_the_caller_default(tmp_path):
    run = tmp_path / "run"
    _write_episode(run, queries=2, tier="batch")
    assert spend_cap.recorded_spend_usd(run, _OPUS, batch=False) == pytest.approx(2 * _ONE_QUERY_USD * 0.5)
    _write_episode(run, queries=2, tier="standard")
    assert spend_cap.recorded_spend_usd(run, _OPUS, batch=True) == pytest.approx(2 * _ONE_QUERY_USD)


def test_a_partial_counts_the_larger_of_its_checkpoint_and_its_usage_ledger(tmp_path):
    """A process killed mid-episode leaves a checkpoint (applied queries) and a
    usage ledger (every reply, including one whose step was never applied)."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "checkpoint.json").write_text(json.dumps({"transcript": [_query(0), _query(1)]}), encoding="utf-8")
    with (run / "queries.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(3):
            fh.write(json.dumps({"query_index": i + 1, "usage": _USAGE}) + "\n")
    assert spend_cap.recorded_spend_usd(run, _OPUS, batch=False) == pytest.approx(3 * _ONE_QUERY_USD)
    (run / "queries.jsonl").unlink()
    assert spend_cap.recorded_spend_usd(run, _OPUS, batch=True) == pytest.approx(2 * _ONE_QUERY_USD * 0.5)


def test_a_finished_episode_is_not_counted_twice_with_its_leftovers(tmp_path):
    run = tmp_path / "run"
    _write_episode(run, queries=2)
    (run / "checkpoint.json").write_text(json.dumps({"transcript": [_query(0)]}), encoding="utf-8")
    assert spend_cap.recorded_spend_usd(run, _OPUS, batch=False) == pytest.approx(2 * _ONE_QUERY_USD)


def test_unreadable_files_and_empty_dirs_count_nothing(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "episode.json").write_text("{not json", encoding="utf-8")
    (run / "queries.jsonl").write_text('{"usage": {"input_tokens": 1000, "output_tokens": 400}}\nnot json\n',
                                       encoding="utf-8")
    assert spend_cap.recorded_spend_usd(run, _OPUS, batch=False) == pytest.approx(_ONE_QUERY_USD)
    assert spend_cap.recorded_spend_usd(tmp_path / "missing", _OPUS, batch=False) == 0.0


def test_restore_never_lowers_a_running_total(tmp_path):
    run = tmp_path / "run"
    _write_episode(run, queries=1)
    SPEND.add("claude-opus-5-5", 5.0)
    assert spend_cap.restore_spend(_OPUS, [run], batch=False) == pytest.approx(5.0)


def test_restore_prices_kimi_and_openai_like_their_agents(tmp_path):
    from interface.agents.kimi_k26 import KIMI_PRICES, kimi_usage_cost_usd
    from interface.agents.openai_agent import price_for, usage_cost_usd

    run = tmp_path / "run"
    run.mkdir()
    usage = {"input_tokens": 2000, "cache_read_input_tokens": 500, "cache_creation_input_tokens": 300,
             "output_tokens": 700}
    (run / "episode.json").write_text(json.dumps({"transcript": [_query(0, usage)]}), encoding="utf-8")
    kimi = {"provider": "kimi", "model": "kimi-k3", "prompt_cache_ttl": "1h", "spend_cap_usd": 5}
    assert spend_cap.recorded_spend_usd(run, kimi, batch=False) == pytest.approx(
        kimi_usage_cost_usd(usage, KIMI_PRICES["kimi-k3"], cache_ttl="1h"))
    assert spend_cap.recorded_spend_usd(run, kimi, batch=True) == pytest.approx(
        kimi_usage_cost_usd(usage, KIMI_PRICES["kimi-k3"], cache_ttl=None))
    astra = {"provider": "openai", "model": "gpt-6-astra", "service_tier": "flex", "spend_cap_usd": 5}
    assert spend_cap.recorded_spend_usd(run, astra, batch=False) == pytest.approx(
        usage_cost_usd(usage, price_for("gpt-6-astra", "flex")))


def test_a_local_rerun_with_resume_killed_keeps_the_cap(tmp_path, monkeypatch):
    """The local pipeline restores from the run dirs it is about to run."""
    from scripts.run_pipeline import run_from_config

    calls = _counting_urlopen(monkeypatch)
    cfg = tmp_path / "rc.json"
    cfg.write_text(json.dumps({"models": {
        "opus": {**_OPUS, "tasks": [str(default_maze_path("V01_empty_room.json"))]},
    }}), encoding="utf-8")
    kwargs = dict(run_config_path=cfg, manifest_path=_MANIFEST, seeds=[0], conditions=None,
                  artifacts_root=tmp_path / "artifacts", run_set_id="local",
                  difficulty_max_static_score=1000.0)
    run_from_config(**kwargs)
    assert len(calls) == 1
    SPEND.reset()                                   # the next invocation is a new process
    run_from_config(**kwargs, resume_killed=True)
    assert len(calls) == 1, "the rerun spent past its cap again"


def test_resume_from_archive_requires_its_own_cap_for_kimi_too():
    from scripts.resume_from_archive import _PER_PROCESS_SPEND_PROVIDERS, build_live_agent

    assert "kimi" in _PER_PROCESS_SPEND_PROVIDERS
    with pytest.raises(SystemExit, match="--spend-cap-usd is required"):
        build_live_agent({"provider": "kimi", "model": "kimi-k3", "spend_cap_usd": 1300}, log=lambda *_: None)


def test_a_worker_restores_only_its_models_dirs_once_per_process(tmp_path):
    """The lockstep worker's restore: an in-flight partial (checkpoint, no
    run_inputs yet) is priced at batch rates; another model's dirs don't count;
    a second unit of the same model does not restore again."""
    root = tmp_path / "worker"
    mine = root / "runs" / "t1" / "minigrid" / "claude-opus-5-5" / "seed_0" / "default"
    mine.mkdir(parents=True)
    (mine / "checkpoint.json").write_text(json.dumps({"transcript": [_query(0), _query(1)]}), encoding="utf-8")
    other = root / "runs" / "t1" / "minigrid" / "kimi-k3" / "seed_0" / "default"
    _write_episode(other, queries=5)
    unit = {"model_config": dict(_OPUS), "run_dir": "runs/t2/minigrid/claude-opus-5-5/seed_0/default"}

    drp._restore_capped_spend(unit, root, batch=True)
    assert SPEND.total("claude-opus-5-5") == pytest.approx(2 * _ONE_QUERY_USD * 0.5)
    _write_episode(mine.parent / "other_condition", queries=3)        # written later by this process
    drp._restore_capped_spend(unit, root, batch=True)
    assert SPEND.total("claude-opus-5-5") == pytest.approx(2 * _ONE_QUERY_USD * 0.5)


def test_an_uncapped_model_is_not_restored(tmp_path):
    root = tmp_path / "worker"
    _write_episode(root / "runs" / "t" / "minigrid" / "claude-opus-5-5" / "seed_0" / "default", queries=2)
    unit = {"model_config": {"provider": "claude", "model": "claude-opus-5-5"},
            "run_dir": "runs/t/minigrid/claude-opus-5-5/seed_0/default"}
    drp._restore_capped_spend(unit, root, batch=False)
    assert SPEND.total("claude-opus-5-5") == 0.0


def test_an_uncapped_kimi_run_still_resumes_without_a_cap(monkeypatch):
    from interface.agents.kimi_k26 import KimiK26Agent
    from scripts.resume_from_archive import build_live_agent

    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    agent = build_live_agent({"provider": "kimi", "model": "kimi-k2.6"}, log=lambda *_: None)
    assert isinstance(agent, KimiK26Agent) and agent.config.spend_cap_usd is None
    capped = build_live_agent({"provider": "kimi", "model": "kimi-k3", "enable_thinking": True, "spend_cap_usd": 1300},
                              spend_cap_usd=20, log=lambda *_: None)
    assert capped.config.spend_cap_usd == 20
