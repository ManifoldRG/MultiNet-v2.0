"""Stage 5 aggregation reports for scorer artifacts and experiments."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from .io import dump_json


def build_experiment_reports(
    static_scores: Iterable[dict[str, Any]],
    runtime_scores: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build the Stage 5 report payloads required by experiments 1-3."""
    static_rows = [_static_row(score) for score in static_scores]
    runtime_rows = [_runtime_row(score) for score in runtime_scores]
    return {
        "leaderboard.json": _leaderboard(runtime_rows),
        "tier_breakdown.json": _tier_breakdown(static_rows, runtime_rows),
        "scoring_calibration_summary.json": _scoring_calibration_summary(
            static_rows, runtime_rows
        ),
        "complexity_distance_summary.json": _complexity_distance_summary(runtime_rows),
        "mechanism_ordering_pairs.json": _mechanism_ordering_pairs(runtime_rows),
    }


def write_experiment_reports(
    static_scores: Iterable[dict[str, Any]],
    runtime_scores: Iterable[dict[str, Any]],
    output_dir: str | Path,
) -> list[Path]:
    """Write Stage 5 report payloads and return their paths."""
    output_root = Path(output_dir)
    reports = build_experiment_reports(static_scores, runtime_scores)
    paths = []
    for filename, payload in reports.items():
        path = output_root / filename
        dump_json(path, payload)
        paths.append(path)
    return paths


def _static_row(score: dict[str, Any]) -> dict[str, Any]:
    dimensions = score.get("dimensions_12", score.get("dimensions", {}))
    return {
        "task_id": str(score.get("task_id", "")),
        "difficulty_tier": score.get("difficulty_tier"),
        "static_score": float(score.get("static_score", score.get("composite", 0.0))),
        "dimensions": {
            str(key): float(value)
            for key, value in dimensions.items()
        }
        if isinstance(dimensions, dict)
        else {},
    }


def _runtime_row(score: dict[str, Any]) -> dict[str, Any]:
    signals = score.get("signals", score)
    if not isinstance(signals, dict):
        signals = {}
    ratio = signals.get("optimality_ratio", signals.get("step_ratio"))
    return {
        "task_id": str(score.get("task_id", "")),
        "experiment": str(score.get("experiment", "")),
        "condition": str(score.get("condition", "")),
        "variant": str(score.get("variant", "")),
        "pair_id": str(score.get("pair_id", score.get("pair", ""))),
        "model_id": str(
            score.get("model_id", score.get("agent_or_model", score.get("adapter", "")))
        ),
        "seed": score.get("seed"),
        "success": bool(signals.get("success", score.get("success", False))),
        "composite": float(score.get("composite", 0.0)),
        "optimality_ratio": float(ratio) if ratio is not None else None,
        "path_choice": signals.get("path_choice", score.get("path_choice")),
        "mechanism_interaction_order": signals.get(
            "mechanism_interaction_order",
            score.get("mechanism_interaction_order"),
        ),
        "failure_point": signals.get("failure_point", score.get("failure_point")),
        "has_mechanism_interaction_order": (
            "mechanism_interaction_order" in signals
            or "mechanism_interaction_order" in score
        ),
        "has_failure_point": "failure_point" in signals or "failure_point" in score,
    }


def _leaderboard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = _group_rows(rows, ("model_id",))
    return {
        "run_count": len(rows),
        "models": [
            _group_summary(("model_id",), key, grouped)
            for key, grouped in groups.items()
        ],
    }


def _tier_breakdown(
    static_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    tiers = {row["task_id"]: row["difficulty_tier"] for row in static_rows}
    rows = [{**row, "difficulty_tier": tiers.get(row["task_id"])} for row in runtime_rows]
    groups = _group_rows(rows, ("difficulty_tier", "model_id"))
    return {
        "run_count": len(rows),
        "groups": [
            _group_summary(("difficulty_tier", "model_id"), key, grouped)
            for key, grouped in groups.items()
        ],
    }


def _scoring_calibration_summary(
    static_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    names = sorted(
        {
            name
            for row in static_rows
            for name in row["dimensions"]
        }
    )
    vectors = {
        name: [row["dimensions"].get(name) for row in static_rows]
        for name in names
    }
    correlations = {
        left: {
            right: _correlation(vectors[left], vectors[right])
            for right in names
        }
        for left in names
    }
    tier_scores: dict[int, list[float]] = defaultdict(list)
    for row in static_rows:
        if row["difficulty_tier"] is not None:
            tier_scores[int(row["difficulty_tier"])].append(row["static_score"])

    return {
        "static_task_count": len(static_rows),
        "runtime_run_count": len(runtime_rows),
        "success_rate_by_task_condition_agent": _success_rate_groups(runtime_rows),
        "dimension_correlation_matrix": correlations,
        "point_weight_candidates": {
            "method": "inverse_max_abs",
            "weights": {
                name: _inverse_max_abs(vectors[name])
                for name in names
            },
        },
        "tier_boundary_candidates": _tier_boundaries(tier_scores),
    }


def _complexity_distance_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if _is_test_two(row["experiment"])]
    if any(row["path_choice"] is None for row in selected):
        raise ValueError("Test 2 aggregation requires path_choice telemetry")
    path_counts = Counter(
        (row["condition"], row["model_id"], str(row["path_choice"]))
        for row in selected
    )
    return {
        "run_count": len(selected),
        "success_rate_by_task_condition_agent": _success_rate_groups(selected),
        "path_choice_counts": [
            {
                "condition": condition,
                "model_id": model_id,
                "path_choice": path_choice,
                "count": count,
            }
            for (condition, model_id, path_choice), count in sorted(path_counts.items())
        ],
    }


def _mechanism_ordering_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if _is_test_three(row["experiment"])]
    if any(not row["pair_id"] or not row["condition"] for row in selected):
        raise ValueError("Test 3 aggregation requires pair_id and condition metadata")
    if any(
        not row["has_mechanism_interaction_order"] or not row["has_failure_point"]
        for row in selected
    ):
        raise ValueError(
            "Test 3 aggregation requires mechanism_interaction_order and "
            "failure_point telemetry"
        )
    grouped = _group_rows(selected, ("pair_id", "model_id", "seed"))
    paired_deltas = []
    for key, pair_rows in grouped.items():
        pair_id, model_id, seed = key
        by_condition = {
            row["condition"]: 1.0 if row["success"] else 0.0
            for row in pair_rows
        }
        if len(by_condition) != len(pair_rows):
            raise ValueError(
                "Test 3 aggregation received duplicate rows for the same "
                "pair_id, model_id, seed, and condition"
            )
        conditions = sorted(by_condition)
        if len(conditions) < 2:
            continue
        deltas = {
            f"{right}_minus_{left}": by_condition[right] - by_condition[left]
            for index, left in enumerate(conditions)
            for right in conditions[index + 1 :]
        }
        paired_deltas.append(
            {
                "pair_id": pair_id,
                "model_id": model_id,
                "seed": seed,
                "success_by_condition": by_condition,
                "paired_success_deltas": deltas,
            }
        )

    return {
        "run_count": len(selected),
        "success_rate_by_task_condition_agent": _success_rate_groups(selected),
        "paired_deltas": paired_deltas,
        "runs": [
            {
                key: row[key]
                for key in (
                    "task_id",
                    "pair_id",
                    "condition",
                    "model_id",
                    "seed",
                    "success",
                    "mechanism_interaction_order",
                    "failure_point",
                )
            }
            for row in selected
        ],
    }


def _success_rate_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = _group_rows(rows, ("task_id", "condition", "model_id"))
    return [
        _group_summary(("task_id", "condition", "model_id"), key, grouped)
        for key, grouped in groups.items()
    ]


def _group_rows(
    rows: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    return dict(sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])))


def _group_summary(
    fields: tuple[str, ...],
    key: tuple[Any, ...],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ratios = [row["optimality_ratio"] for row in rows if row["optimality_ratio"] is not None]
    return {
        **dict(zip(fields, key)),
        "run_count": len(rows),
        "success_rate": mean(1.0 if row["success"] else 0.0 for row in rows),
        "mean_composite": mean(row["composite"] for row in rows),
        "mean_optimality_ratio": mean(ratios) if ratios else None,
        "median_optimality_ratio": median(ratios) if ratios else None,
    }


def _correlation(left: list[float | None], right: list[float | None]) -> float | None:
    pairs = [
        (float(x), float(y))
        for x, y in zip(left, right)
        if x is not None and y is not None
    ]
    if len(pairs) < 2:
        return None
    left_mean = mean(pair[0] for pair in pairs)
    right_mean = mean(pair[1] for pair in pairs)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in pairs)
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x, _ in pairs))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for _, y in pairs))
    if left_norm == 0 or right_norm == 0:
        return None
    return numerator / (left_norm * right_norm)


def _inverse_max_abs(values: list[float | None]) -> float:
    max_abs = max((abs(float(value)) for value in values if value is not None), default=0.0)
    return 1.0 / max_abs if max_abs else 1.0


def _tier_boundaries(tier_scores: dict[int, list[float]]) -> list[dict[str, Any]]:
    tiers = sorted(tier_scores)
    return [
        {
            "lower_tier": lower,
            "upper_tier": upper,
            "candidate": (max(tier_scores[lower]) + min(tier_scores[upper])) / 2.0,
        }
        for lower, upper in zip(tiers, tiers[1:])
    ]


def _is_test_two(experiment: str) -> bool:
    normalized = experiment.lower().replace("-", "_")
    return "test2" in normalized or "complexity_distance" in normalized


def _is_test_three(experiment: str) -> bool:
    normalized = experiment.lower().replace("-", "_")
    return "test3" in normalized or "mechanism_order" in normalized
