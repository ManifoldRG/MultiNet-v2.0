"""Stage 5 — thin aggregation reports for tests 1-3.

Pure functions over in-memory run rows (Appendix A.3 dicts), per-run composites,
static-score artifacts, and the manifest. These produce calibration *evidence*,
not a final MultiNet score.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Iterable, Optional

import numpy as np

from scorer.config import DIMENSION_NAMES


def _run_key(row: dict[str, Any]) -> tuple:
    return (
        row.get("task_id"),
        row.get("agent_or_model"),
        row.get("seed"),
        row.get("condition"),
        row.get("prompt_variant"),
    )


def _mean(values: list[float]) -> Optional[float]:
    return float(statistics.fmean(values)) if values else None


def _median(values: list[float]) -> Optional[float]:
    return float(statistics.median(values)) if values else None


def _group_success(rows: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key))].append(bool(row.get("success")))
    return {
        name: {"n": len(flags), "success_rate": _mean([float(f) for f in flags])}
        for name, flags in buckets.items()
    }


def scoring_calibration_summary(
    rows: list[dict[str, Any]],
    composites: dict[tuple, float],
    static_by_task: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Test 1: success rates, optimality, and 12-dimension correlation evidence."""
    successful_opt = [
        float(r["optimality_ratio"])
        for r in rows
        if r.get("success") and r.get("optimality_ratio") is not None
    ]

    # Per-task mean composite, for correlating static dimensions against difficulty.
    comp_by_task: dict[str, list[float]] = defaultdict(list)
    succ_by_task: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        comp = composites.get(_run_key(r))
        if comp is not None:
            comp_by_task[r["task_id"]].append(float(comp))
        succ_by_task[r["task_id"]].append(float(bool(r.get("success"))))

    tasks = [t for t in static_by_task if t in comp_by_task]
    correlation: dict[str, Optional[float]] = {}
    point_weight_candidates: dict[str, Optional[float]] = {}
    if len(tasks) >= 2:
        dim_matrix = np.array(
            [
                [float((static_by_task[t].get("dimensions_12") or {}).get(name, 0.0)) for name in DIMENSION_NAMES]
                for t in tasks
            ],
            dtype=float,
        )
        target = np.array([_mean(comp_by_task[t]) or 0.0 for t in tasks], dtype=float)
        for idx, name in enumerate(DIMENSION_NAMES):
            col = dim_matrix[:, idx]
            if np.std(col) == 0 or np.std(target) == 0:
                correlation[name] = None
            else:
                correlation[name] = float(np.corrcoef(col, target)[0, 1])
        abs_corr = {n: abs(c) for n, c in correlation.items() if c is not None}
        total = sum(abs_corr.values())
        for name in DIMENSION_NAMES:
            point_weight_candidates[name] = (
                abs_corr[name] / total if total > 0 and name in abs_corr else None
            )

    static_scores = [
        float(static_by_task[t]["static_score"])
        for t in static_by_task
        if static_by_task[t].get("static_score") is not None
    ]
    tier_boundary_candidates = (
        {
            "p33": float(np.percentile(static_scores, 33)),
            "p66": float(np.percentile(static_scores, 66)),
        }
        if static_scores
        else {}
    )

    return {
        "experiment": "test1",
        "run_count": len(rows),
        "task_count": len(static_by_task),
        "success_rate_by_task": _group_success(rows, "task_id"),
        "success_rate_by_condition": _group_success(rows, "condition"),
        "success_rate_by_prompt_variant": _group_success(rows, "prompt_variant"),
        "success_rate_by_model": _group_success(rows, "agent_or_model"),
        "optimality_ratio_mean": _mean(successful_opt),
        "optimality_ratio_median": _median(successful_opt),
        "dimension_correlation": correlation,
        "point_weight_candidates": point_weight_candidates,
        "tier_boundary_candidates": tier_boundary_candidates,
    }


def complexity_distance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Test 2: path-choice counts (short mechanistic vs long open route)."""
    test2 = [r for r in rows if r.get("experiment") == "test2"]
    by_group: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    overall: dict[str, int] = defaultdict(int)
    for r in test2:
        choice = r.get("path_choice") or "none"
        group = (
            f"{r.get('task_id')}|{r.get('condition')}|"
            f"{r.get('prompt_variant')}|{r.get('agent_or_model')}"
        )
        by_group[group][choice] += 1
        overall[choice] += 1
    return {
        "experiment": "test2",
        "run_count": len(test2),
        "path_choice_overall": dict(overall),
        "path_choice_by_group": {g: dict(c) for g, c in by_group.items()},
        "success_rate_by_path_choice": {
            choice: _mean(
                [float(bool(r.get("success"))) for r in test2 if (r.get("path_choice") or "none") == choice]
            )
            for choice in set((r.get("path_choice") or "none") for r in test2)
        },
    }


def mechanism_ordering_pairs(
    rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Test 3: paired success deltas across matched mechanism-ordering pairs."""
    pair_of = {m.get("task_id"): m.get("pair_id") for m in manifest_rows}
    expected_of = {m.get("task_id"): list(m.get("expected_mechanisms", []) or []) for m in manifest_rows}

    test3 = [r for r in rows if r.get("experiment") == "test3"]
    pairs: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in test3:
        pid = pair_of.get(r.get("task_id"))
        if pid is None:
            continue
        pairs[pid][str(r.get("condition"))].append(r)

    pair_reports: dict[str, Any] = {}
    for pid, conditions in pairs.items():
        cond_stats = {}
        for cond, cond_rows in conditions.items():
            failures: dict[str, int] = defaultdict(int)
            order_match = []
            for r in cond_rows:
                if not r.get("success"):
                    fp = r.get("failure_point") or {}
                    failures[str(fp.get("mechanism"))] += 1
                expected = expected_of.get(r.get("task_id"), [])
                # The interaction order also carries downstream effects (opened
                # doors/gates) that are not in expected_mechanisms; compare only
                # the actuated mechanisms' relative order so a correct solve matches.
                expected_set = set(expected)
                engaged_order = [
                    m for m in (r.get("mechanism_interaction_order") or []) if m in expected_set
                ]
                order_match.append(
                    float(engaged_order == expected) if expected else 0.0
                )
            cond_stats[cond] = {
                "n": len(cond_rows),
                "success_rate": _mean([float(bool(r.get("success"))) for r in cond_rows]),
                "failure_point_counts": dict(failures),
                "expected_order_match_rate": _mean(order_match),
            }
        sorted_conds = sorted(cond_stats)
        delta = None
        if len(sorted_conds) == 2:
            a, b = sorted_conds
            sr_a, sr_b = cond_stats[a]["success_rate"], cond_stats[b]["success_rate"]
            if sr_a is not None and sr_b is not None:
                delta = {"conditions": [a, b], "success_delta": sr_a - sr_b}
        pair_reports[pid] = {"conditions": cond_stats, "paired_success_delta": delta}

    return {
        "experiment": "test3",
        "run_count": len(test3),
        "pairs": pair_reports,
    }
