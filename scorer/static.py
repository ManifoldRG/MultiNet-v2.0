"""Static task scoring and Stage 2 artifact generation."""

from __future__ import annotations

from pathlib import Path

from gridworld.baselines import PlannedPath, plan_bfs_path, plan_greedy_path
from gridworld.task_spec import TaskSpecification
from gridworld.task_validator import DifficultyReport, TaskValidator, compute_difficulty

from .artifacts import ScoredDifficulty, StaticScoreArtifact
from .config import (
    DEFAULT_DISTRACTOR_TYPE_WEIGHTS,
    DIMENSION_NAMES,
    GREEDY_SOLVABILITY_FEATURE,
    SCORER_VERSION,
    ScorerConfig,
)
from .io import dump_json, load_json, stable_hash, task_spec_from_payload
from .solvers import compute_canonical_paths, compute_greedy_solvability, require_scorable_spec


def _count_backtracking(solution: list[tuple[int, int]] | None) -> float:
    if not solution:
        return 0.0
    seen = set()
    revisits = 0
    previous_pos = None
    for pos in solution:
        if pos == previous_pos:
            continue
        if pos in seen:
            revisits += 1
        seen.add(pos)
        previous_pos = pos
    return float(revisits)


def _dependency_variety(spec: TaskSpecification) -> float:
    if spec.dependency_chain is not None:
        return float(len({step.type for step in spec.dependency_chain.sequence}))

    variety = 0
    if spec.mechanisms.keys and spec.mechanisms.doors:
        variety += 1
    if spec.mechanisms.switches and spec.mechanisms.gates:
        variety += 1
    if spec.mechanisms.blocks:
        variety += 1
    if spec.mechanisms.teleporters:
        variety += 1
    if spec.mechanisms.hazards:
        variety += 1
    return float(variety)


def _distractor_quality(
    spec: TaskSpecification,
    distractor_type_weights: dict[str, float] | None = None,
) -> float:
    if not spec.distractors:
        return 0.0
    weights = distractor_type_weights or DEFAULT_DISTRACTOR_TYPE_WEIGHTS
    return float(sum(weights.get(d.type, 1.0) for d in spec.distractors))


def _partial_observability(spec: TaskSpecification) -> float:
    mapping = {"full": 0.0, "view_cone": 1.0, "fog_of_war": 2.0}
    return mapping.get(spec.rules.observability, 0.0)


def _irreversibility(spec: TaskSpecification) -> float:
    score = 0.0
    if spec.rules.key_consumption:
        score += float(len(spec.mechanisms.doors))
    score += float(sum(1 for switch in spec.mechanisms.switches if switch.switch_type == "one_shot"))
    score += float(sum(1 for tp in spec.mechanisms.teleporters if not tp.bidirectional))
    return score


def compute_12d_score(
    spec: TaskSpecification,
    solver_output: DifficultyReport | None = None,
    weights: list[float] | None = None,
    config: ScorerConfig | None = None,
    validator: TaskValidator | None = None,
    bfs_path: PlannedPath | None = None,
) -> ScoredDifficulty:
    """
    Compute the 12-dimension static benchmark score.

    This remains compatible with the older gridworld scoring API while moving
    calibration and artifact generation into the standalone scorer package.
    """
    require_scorable_spec(spec)
    scorer_config = config or ScorerConfig.default()
    task_validator = validator or TaskValidator(spec)
    if solver_output is None:
        solver_output = compute_difficulty(spec, validator=task_validator)
    if bfs_path is None:
        bfs_path = plan_bfs_path(spec)

    fragility = task_validator.compute_fragility()
    fragility_value = 0.0 if fragility.min_steps_to_break == -1 else 1.0 / fragility.min_steps_to_break

    width, height = spec.maze.dimensions
    grid_size = float(width * height)
    wall_density = float(len(spec.maze.walls) / grid_size) if grid_size else 0.0

    dimensions = [
        float(len(bfs_path.action_labels) if bfs_path.success else 0),
        float(bfs_path.states_explored),
        _count_backtracking(bfs_path.positions),
        fragility_value,
        float(spec.dependency_chain.depth if spec.dependency_chain is not None else solver_output.dependency_depth),
        _dependency_variety(spec),
        float(len(spec.distractors or [])),
        _distractor_quality(spec, scorer_config.distractor_type_weights),
        grid_size,
        wall_density,
        _partial_observability(spec),
        _irreversibility(spec),
    ]

    weight_vector = (
        scorer_config.static_weight_list()
        if weights is None
        else [float(weight) for weight in weights]
    )
    if len(weight_vector) != len(dimensions):
        raise ValueError(f"Expected {len(dimensions)} static weights, got {len(weight_vector)}")
    composite = float(sum(d * w for d, w in zip(dimensions, weight_vector)))
    return ScoredDifficulty(
        dimensions=dimensions,
        dimension_names=DIMENSION_NAMES.copy(),
        composite=composite,
        weights=weight_vector,
    )


def compute_static_score_artifact(
    spec: TaskSpecification,
    config: ScorerConfig | None = None,
    solver_output: DifficultyReport | None = None,
    validator: TaskValidator | None = None,
    validation_result: tuple[bool, list[tuple[int, int]] | None, str] | None = None,
    bfs_path: PlannedPath | None = None,
    greedy_path: PlannedPath | None = None,
) -> StaticScoreArtifact:
    """Compute the Stage 2 static score artifact for one task."""
    require_scorable_spec(spec)
    scorer_config = config or ScorerConfig.default()
    schema_valid, schema_errors = spec.validate()
    task_validator = validator or TaskValidator(spec)
    if validation_result is None:
        validation_result = task_validator.validate()
    is_beatable, _, message = validation_result
    if solver_output is None:
        solver_output = compute_difficulty(
            spec,
            validator=task_validator,
            validation_result=validation_result,
        )
    if bfs_path is None:
        bfs_path = plan_bfs_path(spec)
    if is_beatable != bfs_path.success:
        raise ValueError(
            "Task validator and canonical BFS disagree on beatability for "
            f"{spec.task_id!r}"
        )
    score = compute_12d_score(
        spec,
        solver_output=solver_output,
        config=scorer_config,
        validator=task_validator,
        bfs_path=bfs_path,
    )

    mechanism_necessity_violations: list[str] = []
    distractor_safety_violations: list[str] = []
    chain_ordering_valid = True
    if schema_valid:
        mechanism_necessity_violations = task_validator.validate_mechanism_necessity()
        distractor_safety_violations = task_validator.validate_distractor_safety(
            base_beatable=is_beatable
        )
        chain_ordering_valid = task_validator.validate_chain_ordering()

    dimensions = score.dimensions_by_name
    static_score_unweighted = float(sum(dimensions.values()))
    inputs_hash = stable_hash(
        {
            "task": spec.to_dict(),
            "config": scorer_config.to_dict(),
            "scorer_version": SCORER_VERSION,
        }
    )

    return StaticScoreArtifact(
        task_id=spec.task_id,
        is_beatable=is_beatable,
        message=message,
        dimensions=dimensions,
        static_score_unweighted=static_score_unweighted,
        static_score=score.composite,
        weights=dict(scorer_config.static_dimension_weights),
        validation={
            "schema_valid": schema_valid,
            "schema_errors": schema_errors,
            "mechanism_necessity_violations": mechanism_necessity_violations,
            "distractor_safety_violations": distractor_safety_violations,
            "chain_ordering_valid": chain_ordering_valid,
        },
        canonical_agent_features={
            GREEDY_SOLVABILITY_FEATURE: (
                compute_greedy_solvability(spec, greedy_path=greedy_path)
                if schema_valid
                else None
            ),
        },
        calibration_version=scorer_config.version,
        inputs_hash=inputs_hash,
    )


def score_task_file(
    task_path: str | Path,
    output_dir: str | Path | None = None,
    config: ScorerConfig | None = None,
):
    """Score a task JSON file and optionally write canonical score artifacts."""
    spec = task_spec_from_payload(load_json(task_path))
    require_scorable_spec(spec)
    validator = TaskValidator(spec)
    validation_result = validator.validate()
    difficulty = compute_difficulty(
        spec,
        validator=validator,
        validation_result=validation_result,
    )
    bfs_path = plan_bfs_path(spec)
    greedy_path = plan_greedy_path(spec)
    canonical_paths = compute_canonical_paths(
        spec,
        bfs_path=bfs_path,
        greedy_path=greedy_path,
    )
    static_score = compute_static_score_artifact(
        spec,
        config=config,
        solver_output=difficulty,
        validator=validator,
        validation_result=validation_result,
        bfs_path=bfs_path,
        greedy_path=greedy_path,
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dump_json(out / "canonical_paths.json", canonical_paths.to_dict())
        dump_json(out / "scored_static.json", static_score.to_dict())

    return canonical_paths, static_score
