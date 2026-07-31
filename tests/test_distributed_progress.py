from __future__ import annotations

import json
from pathlib import Path

from scripts.distributed_run_pipeline import CoordinatorStore, plan_path


def _write_plan(root: Path, n_units: int, group: str = "qwen36-27b", max_in_flight: int = 2) -> dict:
    units = [
        {
            "unit_id": f"u{i}",
            "task_id": f"t{i}",
            "model_group": group,
            "hardware_profile": "local-gpu",
            "max_in_flight": max_in_flight,
        }
        for i in range(n_units)
    ]
    plan = {
        "job_id": "job_test",
        "units": units,
        "scorer_config": {},
        "difficulty_max_static_score": 1000.0,
    }
    p = plan_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(plan))
    return plan


def _caps() -> dict:
    return {"model_groups": ["qwen36-27b"], "hardware_profile": "local-gpu"}


def _store(tmp_path: Path, monkeypatch, n_units: int) -> CoordinatorStore:
    _write_plan(tmp_path, n_units)
    store = CoordinatorStore(tmp_path)
    # assign() builds a payload that reads per-task files; we only care about
    # which unit is selected, so return the raw unit dict instead.
    monkeypatch.setattr(store, "_unit_payload", lambda plan, unit: dict(unit))
    return store


def test_work_stealing_two_workers_three_units(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch, n_units=3)

    a = store.assign("A", _caps())["unit"]["unit_id"]
    b = store.assign("B", _caps())["unit"]["unit_id"]
    assert a != b                      # parallel pickup of two distinct units

    # group is at max_in_flight=2 -> a fresh worker gets nothing yet
    assert store.assign("C", _caps())["unit"] is None

    # A finishes its unit -> frees an in-flight slot
    state = store.load_state()
    state["units"][a]["status"] = "verified"
    store.save_state(state)

    third = store.assign("A", _caps())["unit"]["unit_id"]
    assert third not in {a, b}         # work-stealing: A claims the 3rd unit

    # all units now assigned/verified -> empty queue -> clean None
    assert store.assign("C", _caps())["unit"] is None


def test_heartbeat_records_progress_monotonic(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch, n_units=1)
    store.assign("A", _caps())                       # u0 -> A (assigned)
    store.heartbeat("A", "u0", progress=5)
    assert store.load_state()["units"]["u0"]["progress"] == 5
    store.heartbeat("A", "u0", progress=3)            # lower -> ignored
    assert store.load_state()["units"]["u0"]["progress"] == 5
    store.heartbeat("A", "u0", progress=9)
    assert store.load_state()["units"]["u0"]["progress"] == 9


def test_status_progress_total_sums(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch, n_units=2)
    store.assign("A", _caps())
    store.assign("B", _caps())
    store.heartbeat("A", "u0", progress=4)
    store.heartbeat("B", "u1", progress=6)
    assert store.status()["progress_total"] == 10


def test_heartbeat_without_progress_keeps_total_zero(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch, n_units=1)
    store.assign("A", _caps())
    store.heartbeat("A", "u0")                        # no progress kwarg
    assert store.status()["progress_total"] == 0


def test_counting_agent_increments_and_delegates():
    from scripts.distributed_run_pipeline import ProgressCounter, _CountingAgent

    class Inner:
        some_attr = 42
        def __call__(self, messages):
            return "ok"

    pc = ProgressCounter()
    agent = _CountingAgent(Inner(), pc)
    assert agent([{"role": "user"}]) == "ok"
    assert pc.count == 1
    assert agent.some_attr == 42          # non-call attrs delegate to inner
    agent([{"role": "user"}])
    assert pc.count == 2


def test_counting_agent_counts_generate():
    """ExperimentRunner.run prefers agent.generate(messages); the wrapper must
    count those calls too or per-unit progress heartbeats freeze at 0."""
    from scripts.distributed_run_pipeline import ProgressCounter, _CountingAgent

    class Reply:
        def __init__(self, text):
            self.text = text

    class Inner:
        def generate(self, messages):
            return Reply("ok")

    pc = ProgressCounter()
    agent = _CountingAgent(Inner(), pc)
    assert hasattr(agent, "generate")          # inner has generate -> wrapper does
    r = agent.generate([{"role": "user"}])
    assert r.text == "ok"
    assert pc.count == 1
    agent.generate([{"role": "user"}])
    assert pc.count == 2


def test_counting_agent_no_generate_when_inner_lacks_it():
    """A legacy __call__-only double must NOT sprout a generate attribute, so the
    runner's ``hasattr(agent, "generate")`` check still routes it to __call__."""
    from scripts.distributed_run_pipeline import ProgressCounter, _CountingAgent

    class Inner:
        def __call__(self, messages):
            return "ok"

    agent = _CountingAgent(Inner(), ProgressCounter())
    assert not hasattr(agent, "generate")


def test_counting_agent_setattr_reaches_inner():
    """The runner resets ``last_usage`` on the agent before each call and reads it
    back after. The wrapper must forward both the write and the read to the inner
    agent, or usage telemetry is silently dropped for agents (Kimi/Claude) that
    have no ``reset_usage()`` method and rely on the ``setattr`` fallback."""
    from scripts.distributed_run_pipeline import ProgressCounter, _CountingAgent

    class Inner:
        last_usage = None

        def __call__(self, messages):
            self.last_usage = {"tokens": 7}   # inner records usage during the call
            return "ok"

    inner = Inner()
    agent = _CountingAgent(inner, ProgressCounter())
    setattr(agent, "last_usage", None)        # runner's pre-call reset
    agent([{"role": "user"}])
    # Read-back must see the inner's real usage, not a None shadow on the wrapper.
    assert getattr(agent, "last_usage") == {"tokens": 7}
    assert inner.last_usage == {"tokens": 7}


def test_run_assigned_unit_wraps_agent_for_progress(tmp_path, monkeypatch):
    import scripts.distributed_run_pipeline as drp
    from scripts.distributed_run_pipeline import ProgressCounter, _CountingAgent

    seen = {}

    def fake_run_one_unit(row, agent, model_id, **kw):
        seen["agent"] = agent
        agent([{"role": "user"}])          # simulate one maze turn
        return ({"ok": True}, 1.0)

    monkeypatch.setattr(drp.pipeline, "_run_one_unit", fake_run_one_unit)
    monkeypatch.setattr(drp, "materialize_worker_inputs", lambda unit, root: {})
    monkeypatch.setattr(drp.ScorerConfig, "from_dict", staticmethod(lambda d: object()))

    class FakeAgent:
        def __call__(self, messages):
            return "x"

    pc = ProgressCounter()
    unit = {
        "unit_id": "u0", "model_key": "m", "model_config": {}, "model_id": "mid",
        "task_id": "t", "seed": 0, "prompt_variant": "default", "scorer_config": {},
        "difficulty_max_static_score": 1000.0,
        "task_artifacts": {"scored_static": {}},
        "experiment_config": {},
    }
    drp.run_assigned_unit(
        unit, artifacts_root=tmp_path,
        agent_factory=lambda k, c: (FakeAgent(), "label"), progress=pc,
    )
    assert isinstance(seen["agent"], _CountingAgent)
    assert pc.count == 1
