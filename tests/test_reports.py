"""Unit tests for Stage-5 report aggregators (pipeline.reports)."""

from __future__ import annotations

from pipeline import reports


def _row(**kw):
    base = {
        "task_id": "t",
        "experiment": "test1",
        "condition": "default",
        "prompt_variant": "default",
        "agent_or_model": "m",
        "seed": 0,
        "success": True,
        "optimality_ratio": 1.0,
        "path_choice": None,
        "mechanism_interaction_order": [],
        "failure_point": None,
    }
    base.update(kw)
    return base


def test_scoring_calibration_summary_groups_and_correlates():
    rows = [
        _row(task_id="a", success=True, optimality_ratio=1.0),
        _row(task_id="b", success=False, optimality_ratio=0.0),
    ]
    composites = {
        ("a", "m", 0, "default", "default"): 0.2,
        ("b", "m", 0, "default", "default"): 0.8,
    }
    static_by_task = {
        "a": {"static_score": 1.0, "dimensions_12": {"grid_size": 9.0, "optimal_path_length": 3.0}},
        "b": {"static_score": 5.0, "dimensions_12": {"grid_size": 25.0, "optimal_path_length": 9.0}},
    }
    summary = reports.scoring_calibration_summary(rows, composites, static_by_task)

    assert summary["run_count"] == 2
    assert summary["task_count"] == 2
    assert summary["success_rate_by_task"]["a"]["success_rate"] == 1.0
    assert summary["success_rate_by_task"]["b"]["success_rate"] == 0.0
    # Only successful runs feed optimality.
    assert summary["optimality_ratio_mean"] == 1.0
    # Two tasks with variance -> correlation defined for the populated dims.
    assert summary["dimension_correlation"]["grid_size"] is not None
    assert "p33" in summary["tier_boundary_candidates"]


def test_prompt_variants_do_not_collide():
    # Same task + manifest condition, two prompt variants: their composites and
    # success must stay distinct (regression for the setdefault collapse bug).
    rows = [
        _row(task_id="a", prompt_variant="step_by_step", success=True),
        _row(task_id="a", prompt_variant="bulk", success=False),
    ]
    composites = {
        ("a", "m", 0, "default", "step_by_step"): 0.9,
        ("a", "m", 0, "default", "bulk"): 0.1,
    }
    static_by_task = {"a": {"static_score": 1.0, "dimensions_12": {}}}
    summary = reports.scoring_calibration_summary(rows, composites, static_by_task)

    by_variant = summary["success_rate_by_prompt_variant"]
    assert by_variant["step_by_step"]["success_rate"] == 1.0
    assert by_variant["bulk"]["success_rate"] == 0.0
    # Both per-variant composites are reachable (neither overwrote the other).
    assert summary["run_count"] == 2


def test_complexity_distance_summary_counts_path_choice():
    rows = [
        _row(experiment="test2", task_id="T2", condition="shortcut", path_choice="short_mech", success=True),
        _row(experiment="test2", task_id="T2", condition="shortcut", path_choice="long_open", success=False),
        _row(experiment="test2", task_id="T2", condition="shortcut", path_choice="short_mech", success=True),
    ]
    summary = reports.complexity_distance_summary(rows)
    assert summary["run_count"] == 3
    assert summary["path_choice_overall"]["short_mech"] == 2
    assert summary["path_choice_overall"]["long_open"] == 1
    assert summary["success_rate_by_path_choice"]["short_mech"] == 1.0
    assert summary["success_rate_by_path_choice"]["long_open"] == 0.0


def test_mechanism_ordering_pairs_paired_delta():
    rows = [
        _row(experiment="test3", task_id="k", condition="key_first", success=True,
             mechanism_interaction_order=["kB", "s1"]),
        _row(experiment="test3", task_id="s", condition="switch_first", success=False,
             mechanism_interaction_order=["s1"], failure_point={"mechanism": "kB"}),
    ]
    manifest = [
        {"task_id": "k", "pair_id": "corridor", "expected_mechanisms": ["kB", "s1"]},
        {"task_id": "s", "pair_id": "corridor", "expected_mechanisms": ["s1", "kB"]},
    ]
    summary = reports.mechanism_ordering_pairs(rows, manifest)
    pair = summary["pairs"]["corridor"]
    assert pair["conditions"]["key_first"]["success_rate"] == 1.0
    assert pair["conditions"]["switch_first"]["success_rate"] == 0.0
    assert pair["conditions"]["key_first"]["expected_order_match_rate"] == 1.0
    assert pair["conditions"]["switch_first"]["failure_point_counts"]["kB"] == 1
    # sorted conditions: ["key_first", "switch_first"] -> delta = 1.0 - 0.0
    assert pair["paired_success_delta"]["success_delta"] == 1.0


def test_model_report_aggregates_per_model():
    rows = [
        _row(task_id="a", agent_or_model="m1", experiment="test1",
             success=True, optimality_ratio=1.0, steps=3, optimal_steps=3, tokens=10),
        _row(task_id="b", agent_or_model="m1", experiment="test1",
             success=False, optimality_ratio=0.0, steps=9, optimal_steps=3, tokens=20),
        _row(task_id="a", agent_or_model="m2", experiment="test1",
             success=True, optimality_ratio=0.5, steps=6, optimal_steps=3, tokens=5),
    ]
    composites = {
        ("a", "m1", 0, "default", "default"): 0.4,
        ("b", "m1", 0, "default", "default"): 0.0,
        ("a", "m2", 0, "default", "default"): 0.2,
    }

    rep = reports.model_report(rows, composites, "m1", "rs")
    assert rep["schema_version"] == "0.1.0"
    assert rep["model_id"] == "m1"
    assert rep["run_set_id"] == "rs"
    assert rep["provisional"] is True
    assert rep["run_count"] == 2
    assert rep["task_count"] == 2
    assert rep["overall"]["success_rate"] == 0.5
    assert rep["overall"]["optimality_ratio_mean"] == 1.0  # successful runs only
    assert rep["overall"]["tokens_total"] == 30.0
    assert rep["overall"]["composite_mean"] == 0.2  # mean(0.4, 0.0)
    assert "test1" in rep["by_experiment"]
    assert "default" in rep["by_prompt_variant"]
    assert len(rep["tasks"]) == 2
    assert {t["task_id"] for t in rep["tasks"]} == {"a", "b"}

    # A second model is fully independent (no collision).
    rep2 = reports.model_report(rows, composites, "m2", "rs")
    assert rep2["run_count"] == 1
    assert rep2["overall"]["success_rate"] == 1.0


def test_scoring_calibration_summary_lists_ineligible_tasks():
    rows = [_row(task_id="a", success=True)]
    composites = {("a", "m", 0, "default", "default"): 0.5}
    static_by_task = {
        "a": {"static_score": 1.0, "dimensions_12": {}, "is_beatable": True},
        "dead": {"static_score": 0.0, "dimensions_12": {}, "is_beatable": False},
    }
    summary = reports.scoring_calibration_summary(rows, composites, static_by_task)
    assert summary["ineligible_tasks"] == ["dead"]
