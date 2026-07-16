"""R1 batch-API validation-smoke fixtures + budget-gated driver (task A10).

Binding checks (from the task brief):

Fixtures
  * ``manifest.r1_smoke_batch.json`` loads via the real pipeline loader and has
    exactly 5 tasks, each with a source that exists on disk;
  * none of the 5 task sources appear in ``manifest.r1_balanced_03.json``
    (the smoke panel is disjoint from the scored R1 panel);
  * ``run_config.r1_smoke_batch.json`` passes ``check_run_config_expectations``
    against the smoke manifest, and declares equal 64k token caps for both
    models (the equal-caps guard);
  * every fixture beats + BFS-solves (``validate_fixtures.validate_manifest``),
    and each ``expected_mechanisms`` equals the on-path door/gate ids that
    ``plan_bfs_path`` actually opens.

Driver
  * ``run_batch_smoke.main(--dry-run)`` builds the whole plan, prints it, writes
    no report, and spends nothing — a mocked agent factory whose agents raise if
    ever invoked proves no query was issued.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_pipeline as pipeline

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _REPO_ROOT / "gridworld" / "fixtures"
_SMOKE_MANIFEST = _FIXTURES / "manifest.r1_smoke_batch.json"
_SMOKE_RUN_CONFIG = _FIXTURES / "run_config.r1_smoke_batch.json"
_BALANCED_03 = _FIXTURES / "manifest.r1_balanced_03.json"

# The 5 mazes locked in docs/batch-api-lockstep-runner-design.md §Validation smoke.
_EXPECTED_SOURCES = {
    "ogbench/ogbench/procgen/maze_jsons/S4/10x10_dense_0.json",
    "ogbench/ogbench/procgen/maze_jsons/D1/10x10_dense_wrong_ky_kr_sg_1.json",
    "ogbench/ogbench/procgen/maze_jsons/M5/10x10_corridor_kr_kb_1.json",
    "ogbench/ogbench/procgen/maze_jsons/M6/10x10_dense_kr_sg_kb_0.json",
    "ogbench/ogbench/procgen/maze_jsons/D1/14x14_dense_wrong_ky_kr_0.json",
}


# --------------------------------------------------------------------------- #
# Fixtures: manifest
# --------------------------------------------------------------------------- #
def test_manifest_loads_five_tasks_via_pipeline_loader():
    rows = pipeline.load_manifest(_SMOKE_MANIFEST)
    assert len(rows) == 5
    assert all(r["experiment"] == "r1_smoke" for r in rows)
    assert all(r["task_id"].startswith("r1smoke_") for r in rows)
    assert {r["source"] for r in rows} == _EXPECTED_SOURCES


def test_manifest_sources_exist_on_disk():
    rows = pipeline.load_manifest(_SMOKE_MANIFEST)
    for row in rows:
        resolved = pipeline._resolve_path(row["source"], _SMOKE_MANIFEST)
        assert resolved is not None and resolved.exists(), row["source"]


def test_smoke_sources_disjoint_from_balanced_03():
    balanced = {r["source"] for r in pipeline.load_manifest(_BALANCED_03)}
    smoke = {r["source"] for r in pipeline.load_manifest(_SMOKE_MANIFEST)}
    assert smoke.isdisjoint(balanced), smoke & balanced


def test_expected_mechanisms_match_on_path_bfs_events():
    from gridworld import baselines as B
    from gridworld.baselines import MiniGridActions
    from gridworld.task_spec import TaskSpecification

    def on_path_mechanisms(spec) -> list[str]:
        ctx = B.TaskPlanningContext(spec)
        state = ctx.initial_state()
        open_doors = set(state.open_doors)
        open_gates = set(state.open_gates)
        seq: list[str] = []
        for action in B._bfs_actions(spec):
            if action == int(MiniGridActions.DONE):
                break
            trans = next(
                (c for c in B._successors(ctx, state) if c.action == action), None
            )
            if trans is None:
                break
            for d in trans.next_state.open_doors:
                if d not in open_doors:
                    open_doors.add(d)
                    seq.append(d)
            for g in trans.next_state.open_gates:
                if g not in open_gates:
                    open_gates.add(g)
                    seq.append(g)
            state = trans.next_state
        return seq

    for row in pipeline.load_manifest(_SMOKE_MANIFEST):
        resolved = pipeline._resolve_path(row["source"], _SMOKE_MANIFEST)
        spec = TaskSpecification.from_json(str(resolved))
        assert on_path_mechanisms(spec) == row["expected_mechanisms"], row["task_id"]


def test_manifest_beats_and_bfs_solves():
    from scripts.validate_fixtures import validate_manifest

    _data, errors = validate_manifest(_SMOKE_MANIFEST)
    assert errors == [], errors


# --------------------------------------------------------------------------- #
# Fixtures: run config
# --------------------------------------------------------------------------- #
def test_run_config_passes_expectations_against_smoke_manifest():
    rc = pipeline.load_run_config(_SMOKE_RUN_CONFIG)
    # No raise: manifest matches, caps equal, no conditions mismatch.
    pipeline.check_run_config_expectations(rc, _SMOKE_MANIFEST, None)


def test_run_config_equal_token_caps_and_no_qwen():
    rc = pipeline.load_run_config(_SMOKE_RUN_CONFIG)
    models = rc["models"]
    assert set(models) == {"claude_opus", "kimi_k26"}
    assert {m["max_tokens"] for m in models.values()} == {64000}
    assert all(m["max_in_flight"] == 8 for m in models.values())
    assert not any(
        (m.get("provider") or "").startswith("qwen") for m in models.values()
    )
    # The declared manifest must point at the smoke manifest.
    assert Path(rc["manifest"]).name == "manifest.r1_smoke_batch.json"


# --------------------------------------------------------------------------- #
# Driver: --dry-run builds the plan and spends nothing
# --------------------------------------------------------------------------- #
class _NoSpendAgent:
    """Any generation call is a budget violation in --dry-run."""

    def generate_batch(self, batch, *args, **kwargs):  # noqa: D401
        raise AssertionError("generate_batch called during --dry-run (spent money)")

    def generate(self, messages, *args, **kwargs):
        raise AssertionError("generate called during --dry-run (spent money)")

    def __call__(self, *args, **kwargs):
        raise AssertionError("__call__ used during --dry-run (spent money)")


def _no_spend_factory(model_key, model_config):
    return _NoSpendAgent(), "mock-model"


def test_driver_dry_run_builds_plan_and_spends_nothing(tmp_path, capsys):
    from scripts import run_batch_smoke

    artifacts = tmp_path / "art"
    code = run_batch_smoke.main(
        [
            "--models",
            "claude_opus,kimi_k26",
            "--max-batches",
            "5",
            "--control-episodes",
            "1",
            "--artifacts-root",
            str(artifacts),
            "--dry-run",
        ],
        agent_factory=_no_spend_factory,
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "claude_opus" in out and "kimi_k26" in out
    # 5 smoke mazes appear in each model's plan.
    assert "r1smoke_S4_10x10_dense_0" in out
    # A dry run must not write a spend report.
    assert not (artifacts / "smoke_report.json").exists()


def test_driver_refuses_without_budget_ack(tmp_path, monkeypatch):
    from scripts import run_batch_smoke

    monkeypatch.delenv("SMOKE_BUDGET_ACK", raising=False)
    code = run_batch_smoke.main(
        [
            "--models",
            "claude_opus",
            "--artifacts-root",
            str(tmp_path / "art"),
        ],
        agent_factory=_no_spend_factory,
    )
    assert code != 0
