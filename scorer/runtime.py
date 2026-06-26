"""Runtime scoring for run and episode JSON artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import CanonicalPathReport, RuntimeScoreArtifact, StaticScoreArtifact
from .config import SCORER_VERSION, ScorerConfig
from .io import dump_json, load_json, stable_hash


def _artifact_dict(value: dict[str, Any] | StaticScoreArtifact | CanonicalPathReport) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return value.to_dict()  # type: ignore[no-any-return]
    return value


def _lookup_path(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _extract_task_id(run: dict[str, Any], fallback: str = "") -> str:
    return str(
        run.get("task_id")
        or _lookup_path(run, "task_spec", "task_id")
        or _lookup_path(run, "episode", "task_id")
        or fallback
    )


def _extract_bool(run: dict[str, Any], *keys: str, default: bool = False) -> bool:
    for key in keys:
        value = run.get(key)
        if value is not None:
            return bool(value)
    return default


def _extract_steps(run: dict[str, Any]) -> int:
    for key in ("steps", "steps_taken", "steps_used"):
        if run.get(key) is not None:
            return int(run[key])
    signal_steps = _lookup_path(run, "signals", "steps")
    if signal_steps is not None:
        return int(signal_steps)
    final_step = _lookup_path(run, "final_state", "step_count")
    if final_step is not None:
        return int(final_step)
    return 0


def _extract_token_count(run: dict[str, Any]) -> int | None:
    for key in ("total_tokens", "token_count", "tokens"):
        if run.get(key) is not None:
            return int(run[key])
    signal_tokens = _lookup_path(run, "signals", "token_count")
    if signal_tokens is not None:
        return int(signal_tokens)

    trajectory_total = _sum_record_tokens(run.get("trajectory", []))
    if trajectory_total is not None:
        return trajectory_total
    return _sum_record_tokens(run.get("transcript", []), kind="query")


def _sum_record_tokens(records: Any, kind: str | None = None) -> int | None:
    if not isinstance(records, list):
        return None
    total = 0
    found = False
    for item in records:
        if not isinstance(item, dict):
            continue
        if kind is not None and item.get("kind") != kind:
            continue
        item_tokens = _token_count_from_record(item)
        if item_tokens is not None:
            total += item_tokens
            found = True
    return total if found else None


def _token_count_from_record(record: dict[str, Any]) -> int | None:
    for container in (record, record.get("info"), record.get("metadata")):
        if not isinstance(container, dict):
            continue
        for key in ("total_tokens", "token_count", "tokens", "model_tokens"):
            if container.get(key) is not None:
                return int(container[key])
        usage = container.get("usage")
        if isinstance(usage, dict):
            if usage.get("total_tokens") is not None:
                return int(usage["total_tokens"])
            prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
            completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
            if prompt_tokens is not None or completion_tokens is not None:
                return int(prompt_tokens or 0) + int(completion_tokens or 0)
    return None


def _state_position(state: Any) -> tuple[int, int] | None:
    if not isinstance(state, dict):
        return None
    raw = state.get("agent_position") or state.get("position")
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return int(raw[0]), int(raw[1])
    return None


def _extract_run_positions(run: dict[str, Any]) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []

    initial_pos = _state_position(run.get("initial_state"))
    if initial_pos is not None:
        positions.append(initial_pos)

    for item in run.get("trajectory", []):
        if not isinstance(item, dict):
            continue
        pos = _state_position(item.get("state"))
        if pos is not None:
            positions.append(pos)

    for item in run.get("transcript", []):
        if not isinstance(item, dict):
            continue
        if item.get("kind") == "reset":
            pos = _state_position(item.get("state"))
        else:
            pos = _state_position(item.get("state_after"))
            if pos is None:
                raw = item.get("position_after")
                pos = (int(raw[0]), int(raw[1])) if isinstance(raw, list) and len(raw) >= 2 else None
        if pos is not None:
            positions.append(pos)

    final_pos = _state_position(run.get("final_state"))
    if final_pos is not None:
        positions.append(final_pos)

    deduped: list[tuple[int, int]] = []
    for pos in positions:
        if not deduped or deduped[-1] != pos:
            deduped.append(pos)
    return deduped


def _extract_canonical_positions(
    canonical_paths: dict[str, Any],
    agent: str = "bfs",
) -> list[tuple[int, int]]:
    path = canonical_paths.get(agent, canonical_paths if agent == "bfs" else {})
    if not isinstance(path, dict):
        return []
    positions = []
    for pos in path.get("positions", []):
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            positions.append((int(pos[0]), int(pos[1])))
    return positions


def _cell_overlap(run_positions: list[tuple[int, int]], canonical_positions: list[tuple[int, int]]) -> float:
    canonical_cells = set(canonical_positions)
    if not canonical_cells:
        return 0.0
    return len(set(run_positions) & canonical_cells) / len(canonical_cells)


def _extract_static_score(static_score: dict[str, Any]) -> float:
    return float(static_score.get("static_score", static_score.get("composite", 0.0)))


def _extract_greedy_solvability(static_score: dict[str, Any]) -> float:
    value = _lookup_path(static_score, "canonical_agent_features", "greedy_solvability")
    if value is None:
        raise ValueError("Runtime scoring requires evaluated canonical_agent_features.greedy_solvability")
    solvability = float(value)
    if not 0.0 <= solvability <= 1.0:
        raise ValueError("greedy_solvability must be between 0.0 and 1.0")
    return solvability


def _runtime_weighted_average(signals: dict[str, float], weights: dict[str, float]) -> float:
    numerator = 0.0
    denominator = 0.0
    for key in ("step_ratio", "cell_overlap_bfs", "token_efficiency"):
        weight = float(weights.get(key, 0.0))
        numerator += signals[key] * weight
        denominator += weight
    return numerator / denominator if denominator else 0.0


def compute_runtime_score(
    run: dict[str, Any],
    static_score: dict[str, Any] | StaticScoreArtifact,
    canonical_paths: dict[str, Any] | CanonicalPathReport,
    config: ScorerConfig | None = None,
    difficulty_max_static_score: float | None = None,
) -> RuntimeScoreArtifact:
    """Compute the Stage 4 runtime score for one run JSON payload."""
    scorer_config = config or ScorerConfig.default()
    static_data = _artifact_dict(static_score)
    canonical_data = _artifact_dict(canonical_paths)

    task_id = _extract_task_id(run, fallback=str(static_data.get("task_id", "")))
    success = _extract_bool(run, "success", default=bool(_lookup_path(run, "signals", "success") or False))
    steps = _extract_steps(run)
    token_count = _extract_token_count(run)
    canonical_positions = _extract_canonical_positions(canonical_data)
    greedy_positions = _extract_canonical_positions(canonical_data, agent="greedy")
    run_positions = _extract_run_positions(run)

    optimal_steps = int(
        _lookup_path(canonical_data, "bfs", "optimal_steps")
        or canonical_data.get("optimal_steps")
        or static_data.get("optimal_steps", 0)
    )
    step_ratio = 0.0
    if success and optimal_steps > 0:
        step_ratio = optimal_steps / max(float(steps), float(optimal_steps), 1.0)

    cell_overlap_bfs = _cell_overlap(run_positions, canonical_positions)
    cell_overlap_greedy = (
        _cell_overlap(run_positions, greedy_positions)
        if isinstance(canonical_data.get("greedy"), dict)
        else None
    )
    token_efficiency = 1.0
    if token_count is not None:
        token_efficiency = min(1.0, scorer_config.baseline_tokens / max(float(token_count), 1.0))

    static_composite = _extract_static_score(static_data)
    normalizer = (
        difficulty_max_static_score
        if difficulty_max_static_score is not None
        else scorer_config.difficulty_max_static_score
    )
    if normalizer is None:
        raise ValueError(
            "Runtime scoring requires difficulty_max_static_score from the task suite "
            "or scorer config"
        )
    if normalizer <= 0:
        raise ValueError("difficulty_max_static_score must be greater than zero")
    if static_composite > normalizer:
        raise ValueError("difficulty_max_static_score must be at least the task static score")
    difficulty_weight = static_composite / normalizer
    success_factor = 1.0 if success else 0.0
    efficiency_signals = {
        "step_ratio": step_ratio,
        "cell_overlap_bfs": cell_overlap_bfs,
        "token_efficiency": token_efficiency,
    }
    efficiency_factor = _runtime_weighted_average(
        efficiency_signals,
        scorer_config.runtime_weights,
    )
    greedy_solvability = _extract_greedy_solvability(static_data)
    greedy_penalty = (
        scorer_config.runtime_weights.get("greedy_penalty", 0.0)
        * greedy_solvability
        * success_factor
    )
    composite = max(0.0, success_factor * efficiency_factor * difficulty_weight - greedy_penalty)

    signals: dict[str, Any] = {
        "success": success,
        "steps": steps,
        "terminated": _extract_bool(run, "terminated", default=False),
        "truncated": _extract_bool(run, "truncated", default=False),
        "terminated_reason": run.get("terminated_reason") or run.get("end_reason") or ("success" if success else "unknown"),
        "reward": run.get("reward", run.get("total_reward")),
        "token_count": token_count,
        "optimal_steps": optimal_steps,
        "step_ratio": step_ratio,
        "cell_overlap_bfs": cell_overlap_bfs,
        "cell_overlap_greedy": cell_overlap_greedy,
        "token_efficiency": token_efficiency,
        "difficulty_weight": difficulty_weight,
        "efficiency_factor": efficiency_factor,
        "greedy_penalty": greedy_penalty,
    }
    for key in (
        "distractor_interactions",
        "irreversible_failures",
        "path_choice",
        "mechanism_interaction_order",
        "failure_point",
    ):
        if run.get(key) is not None:
            signals[key] = run[key]

    inputs_hash = stable_hash(
        {
            "run": run,
            "static_score": static_data,
            "canonical_paths": canonical_data,
            "config": scorer_config.to_dict(),
            "scorer_version": SCORER_VERSION,
        }
    )

    return RuntimeScoreArtifact(
        task_id=task_id,
        backend=str(run.get("backend", "")),
        adapter=str(run.get("adapter", run.get("agent_or_model", ""))),
        model_id=str(run.get("model_id", run.get("model_name", run.get("agent_or_model", "")))),
        seed=int(run["seed"]) if run.get("seed") is not None else None,
        signals=signals,
        composite=composite,
        calibration_version=scorer_config.version,
        inputs_hash=inputs_hash,
    )


def score_runtime_file(
    run_path: str | Path,
    static_score_path: str | Path,
    canonical_paths_path: str | Path,
    output_path: str | Path | None = None,
    config: ScorerConfig | None = None,
    difficulty_max_static_score: float | None = None,
) -> RuntimeScoreArtifact:
    """Score one run JSON file and optionally write run_score.json."""
    run = load_json(run_path)
    static_score = load_json(static_score_path)
    canonical_paths = load_json(canonical_paths_path)
    score = compute_runtime_score(
        run,
        static_score=static_score,
        canonical_paths=canonical_paths,
        config=config,
        difficulty_max_static_score=difficulty_max_static_score,
    )
    if output_path is not None:
        dump_json(output_path, score.to_dict())
    return score
