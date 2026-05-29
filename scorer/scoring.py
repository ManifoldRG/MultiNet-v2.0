"""Configurable scoring artifacts for gridworld tasks and run outputs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from gridworld.task_spec import TaskSpecification
from gridworld.task_validator import (
    DifficultyReport,
    TaskValidator,
    ValidatorState,
    compute_difficulty,
)


SCORER_VERSION = "0.1.0"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("scorer_config.json")

DIMENSION_NAMES = [
    "optimal_path_length",
    "search_space_size",
    "backtracking_required",
    "fragility",
    "dependency_depth",
    "dependency_variety",
    "distractor_count",
    "distractor_quality",
    "grid_size",
    "wall_density",
    "partial_observability",
    "irreversibility",
]

CANONICAL_AGENT_FEATURE_NAMES = [
    "greedy_solvability",
]

DEFAULT_DISTRACTOR_TYPE_WEIGHTS = {
    "wrong_color_key": 1.0,
    "inactive_switch": 2.0,
    "decoy_door": 2.0,
    "distractor_chain": 3.0,
}

DEFAULT_RUNTIME_WEIGHTS = {
    "step_ratio": 1.0,
    "cell_overlap_bfs": 1.0,
    "token_efficiency": 1.0,
    "greedy_penalty": 0.5,
}


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
        f.write("\n")


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    import hashlib

    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _coerce_float_mapping(
    values: dict[str, Any] | list[Any] | None,
    names: list[str],
    default: float = 1.0,
) -> dict[str, float]:
    if values is None:
        return {name: default for name in names}
    if isinstance(values, list):
        result = {name: default for name in names}
        for name, value in zip(names, values):
            result[name] = float(value)
        return result
    return {name: float(values.get(name, default)) for name in names}


@dataclass
class ScorerConfig:
    """Weights and runtime coefficients used by the standalone scorer."""

    version: str = "default"
    static_dimension_weights: dict[str, float] = field(
        default_factory=lambda: {name: 1.0 for name in DIMENSION_NAMES}
    )
    distractor_type_weights: dict[str, float] = field(
        default_factory=lambda: DEFAULT_DISTRACTOR_TYPE_WEIGHTS.copy()
    )
    runtime_weights: dict[str, float] = field(
        default_factory=lambda: DEFAULT_RUNTIME_WEIGHTS.copy()
    )
    baseline_tokens: float = 1000.0
    difficulty_max_static_score: float | None = None

    @classmethod
    def default(cls) -> "ScorerConfig":
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScorerConfig":
        static_weights = data.get("static_dimension_weights", data.get("static_weights"))
        runtime_weights = data.get("runtime_weights")
        distractor_weights = data.get("distractor_type_weights", data.get("distractor_weights"))

        difficulty_max = data.get("difficulty_max_static_score")
        return cls(
            version=str(data.get("version", "default")),
            static_dimension_weights=_coerce_float_mapping(static_weights, DIMENSION_NAMES),
            distractor_type_weights={
                **DEFAULT_DISTRACTOR_TYPE_WEIGHTS,
                **{k: float(v) for k, v in (distractor_weights or {}).items()},
            },
            runtime_weights={
                **DEFAULT_RUNTIME_WEIGHTS,
                **{k: float(v) for k, v in (runtime_weights or {}).items()},
            },
            baseline_tokens=float(data.get("baseline_tokens", 1000.0)),
            difficulty_max_static_score=(
                None if difficulty_max is None else float(difficulty_max)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "static_dimension_weights": dict(self.static_dimension_weights),
            "distractor_type_weights": dict(self.distractor_type_weights),
            "runtime_weights": dict(self.runtime_weights),
            "baseline_tokens": self.baseline_tokens,
            "difficulty_max_static_score": self.difficulty_max_static_score,
        }

    def static_weight_list(self) -> list[float]:
        return [self.static_dimension_weights.get(name, 1.0) for name in DIMENSION_NAMES]


def load_scorer_config(path: str | Path | None = None) -> ScorerConfig:
    """Load scorer weights from JSON, or return defaults if no file exists."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return ScorerConfig.default()
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "YAML scorer configs require PyYAML. Use JSON or install PyYAML."
            ) from exc
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Expected a YAML object in {config_path}")
        return ScorerConfig.from_dict(data)
    return ScorerConfig.from_dict(_load_json(config_path))


@dataclass
class ScoredDifficulty:
    """Backward-compatible 12-dimension score report."""

    dimensions: list[float]
    dimension_names: list[str] = field(default_factory=lambda: DIMENSION_NAMES.copy())
    composite: float = 0.0
    weights: list[float] = field(default_factory=lambda: [1.0] * len(DIMENSION_NAMES))

    @property
    def dimensions_by_name(self) -> dict[str, float]:
        return dict(zip(self.dimension_names, self.dimensions))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimensions": self.dimensions,
            "dimension_names": self.dimension_names,
            "dimensions_by_name": self.dimensions_by_name,
            "composite": self.composite,
            "weights": self.weights,
        }


@dataclass
class CanonicalPathReport:
    """Canonical solver trace artifact for a task."""

    task_id: str
    success: bool
    actions: list[str]
    positions: list[tuple[int, int]]
    optimal_steps: int
    states_explored: int
    message: str
    producer_version: str = SCORER_VERSION

    @property
    def bfs(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "actions": self.actions,
            "positions": [list(pos) for pos in self.positions],
            "optimal_steps": self.optimal_steps,
            "states_explored": self.states_explored,
            "message": self.message,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "bfs": self.bfs,
            "producer_version": self.producer_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CanonicalPathReport":
        bfs = data.get("bfs", data)
        return cls(
            task_id=str(data.get("task_id", "")),
            success=bool(bfs.get("success", False)),
            actions=[str(action) for action in bfs.get("actions", [])],
            positions=[
                (int(pos[0]), int(pos[1]))
                for pos in bfs.get("positions", [])
                if isinstance(pos, (list, tuple)) and len(pos) >= 2
            ],
            optimal_steps=int(bfs.get("optimal_steps", 0)),
            states_explored=int(bfs.get("states_explored", 0)),
            message=str(bfs.get("message", "")),
            producer_version=str(data.get("producer_version", SCORER_VERSION)),
        )


@dataclass
class StaticScoreArtifact:
    """Stage 2 static score artifact."""

    task_id: str
    is_beatable: bool
    message: str
    dimensions: dict[str, float]
    static_score_unweighted: float
    static_score: float
    weights: dict[str, float]
    validation: dict[str, Any]
    canonical_agent_features: dict[str, float | None]
    calibration_version: str
    inputs_hash: str
    producer_version: str = SCORER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "is_beatable": self.is_beatable,
            "message": self.message,
            "dimensions_12": dict(self.dimensions),
            "dimensions": dict(self.dimensions),
            "static_score_unweighted": self.static_score_unweighted,
            "static_score": self.static_score,
            "composite": self.static_score,
            "weights": dict(self.weights),
            "validation": dict(self.validation),
            "canonical_agent_features": dict(self.canonical_agent_features),
            "calibration_version": self.calibration_version,
            "inputs_hash": self.inputs_hash,
            "producer_version": self.producer_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StaticScoreArtifact":
        dimensions = data.get("dimensions_12", data.get("dimensions", {}))
        if isinstance(dimensions, list):
            dimensions = dict(zip(DIMENSION_NAMES, dimensions))
        return cls(
            task_id=str(data.get("task_id", "")),
            is_beatable=bool(data.get("is_beatable", False)),
            message=str(data.get("message", "")),
            dimensions={str(k): float(v) for k, v in dimensions.items()},
            static_score_unweighted=float(data.get("static_score_unweighted", 0.0)),
            static_score=float(data.get("static_score", data.get("composite", 0.0))),
            weights={str(k): float(v) for k, v in data.get("weights", {}).items()},
            validation=dict(data.get("validation", {})),
            canonical_agent_features=dict(data.get("canonical_agent_features", {})),
            calibration_version=str(data.get("calibration_version", "unknown")),
            inputs_hash=str(data.get("inputs_hash", "")),
            producer_version=str(data.get("producer_version", SCORER_VERSION)),
        )


@dataclass
class RuntimeScoreArtifact:
    """Stage 4 runtime score artifact for one run."""

    task_id: str
    backend: str
    adapter: str
    model_id: str
    seed: int | None
    signals: dict[str, Any]
    composite: float
    calibration_version: str
    inputs_hash: str
    producer_version: str = SCORER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "backend": self.backend,
            "adapter": self.adapter,
            "model_id": self.model_id,
            "seed": self.seed,
            "signals": dict(self.signals),
            "composite": self.composite,
            "calibration_version": self.calibration_version,
            "inputs_hash": self.inputs_hash,
            "producer_version": self.producer_version,
        }


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


def _compute_greedy_solvability(spec: TaskSpecification) -> float | None:
    """Run the optional reporting-branch greedy baseline when it is available."""
    try:
        import numpy as np

        from model_interface import ModelInput

        from gridworld.actions import ACTION_NAMES
        from gridworld.backends.minigrid_backend import MiniGridBackend
        from gridworld.baselines import GreedyModelInterface
    except ImportError:
        return None

    backend = MiniGridBackend(render_mode="rgb_array")
    model = GreedyModelInterface()
    try:
        backend.configure(spec)
        obs, state, _ = backend.reset(seed=spec.seed)
        for step_number in range(1, spec.max_steps + 1):
            image = obs if isinstance(obs, np.ndarray) else np.zeros((64, 64, 3), dtype=np.uint8)
            model_input = ModelInput(
                image=image,
                text_prompt=backend.get_mission_text(),
                action_space=ACTION_NAMES,
                step_number=step_number,
                max_steps=spec.max_steps,
                task_spec=spec,
                grid_state=state,
            )
            output = model.predict(model_input)
            obs, reward, terminated, truncated, state, _ = backend.step(int(output.action))
            if terminated and reward > 0:
                return 1.0
            if terminated or truncated:
                break
    finally:
        backend.close()
    return 0.0


def compute_12d_score(
    spec: TaskSpecification,
    solver_output: DifficultyReport | None = None,
    weights: list[float] | None = None,
    config: ScorerConfig | None = None,
) -> ScoredDifficulty:
    """
    Compute the full 12-dimension benchmark score.

    This is intentionally backward-compatible with the original API while
    allowing callers to load scorer weights from a config file.
    """
    scorer_config = config or ScorerConfig.default()
    validator = TaskValidator(spec)
    _, solution, _ = validator.validate()
    if solver_output is None:
        solver_output = compute_difficulty(spec)

    fragility = validator.compute_fragility()
    fragility_value = 0.0 if fragility.min_steps_to_break == -1 else 1.0 / fragility.min_steps_to_break

    width, height = spec.maze.dimensions
    grid_size = float(width * height)
    wall_density = float(len(spec.maze.walls) / grid_size) if grid_size else 0.0

    dimensions = [
        float(solver_output.optimal_steps),
        float(solver_output.states_explored),
        float(solver_output.backtrack_count if hasattr(solver_output, "backtrack_count") else _count_backtracking(solution)),
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

    weight_vector = weights or scorer_config.static_weight_list()
    composite = float(sum(d * w for d, w in zip(dimensions, weight_vector)))
    return ScoredDifficulty(
        dimensions=dimensions,
        composite=composite,
        weights=weight_vector,
    )


def _initial_validator_state(validator: TaskValidator) -> ValidatorState:
    initial_block_pos = frozenset(
        (bid, pos[0], pos[1]) for pos, bid in validator.blocks.items()
    )
    initial_open_doors = frozenset(
        door["id"] for _, door in validator.doors.items() if not door["locked"]
    )
    initial_active_switches = frozenset(
        sw["id"] for sw in validator.switches.values() if sw.get("initial_state") == "on"
    )
    initial_used_switches = frozenset(
        sw["id"]
        for sw in validator.switches.values()
        if sw.get("initial_state") == "on" and sw.get("switch_type") == "one_shot"
    )
    return ValidatorState(
        agent_pos=validator.start,
        agent_dir=0,
        carrying_key=None,
        collected_keys=frozenset(),
        active_switches=initial_active_switches,
        used_switches=initial_used_switches,
        open_gates=validator._recompute_open_gates(initial_active_switches),
        open_doors=initial_open_doors,
        block_positions=initial_block_pos,
    )


def compute_canonical_paths(
    spec: TaskSpecification,
    max_states: int = 500_000,
) -> CanonicalPathReport:
    """Emit the canonical BFS trace used by static and runtime scoring."""
    validator = TaskValidator(spec)
    initial_state = _initial_validator_state(validator)
    queue = deque([(initial_state, [initial_state.agent_pos], [])])
    visited: set[ValidatorState] = {initial_state}
    states_explored = 0

    while queue:
        if states_explored >= max_states:
            return CanonicalPathReport(
                task_id=spec.task_id,
                success=False,
                actions=[],
                positions=[],
                optimal_steps=0,
                states_explored=states_explored,
                message=f"State space exceeded {max_states} states without finding solution",
            )

        state, positions, actions = queue.popleft()
        states_explored += 1
        if state.agent_pos == validator.goal:
            return CanonicalPathReport(
                task_id=spec.task_id,
                success=True,
                actions=actions,
                positions=positions,
                optimal_steps=len(actions),
                states_explored=states_explored,
                message=f"Solution found in {len(actions)} steps ({states_explored} states explored)",
            )

        for transition in validator._successors(state):
            if transition.next_state in visited:
                continue
            visited.add(transition.next_state)
            queue.append(
                (
                    transition.next_state,
                    positions + [transition.next_pos],
                    actions + [transition.action_label],
                )
            )

    return CanonicalPathReport(
        task_id=spec.task_id,
        success=False,
        actions=[],
        positions=[],
        optimal_steps=0,
        states_explored=states_explored,
        message=f"No solution found ({states_explored} states explored, all reachable states checked)",
    )


def compute_static_score_artifact(
    spec: TaskSpecification,
    config: ScorerConfig | None = None,
    solver_output: DifficultyReport | None = None,
) -> StaticScoreArtifact:
    """Compute the Stage 2 static score artifact for one task."""
    scorer_config = config or ScorerConfig.default()
    schema_valid, schema_errors = spec.validate()
    solver_output = solver_output or compute_difficulty(spec)
    score = compute_12d_score(spec, solver_output=solver_output, config=scorer_config)
    validator = TaskValidator(spec)
    is_beatable, _, message = validator.validate()

    mechanism_necessity_violations: list[str] = []
    distractor_safety_violations: list[str] = []
    chain_ordering_valid = True
    if schema_valid:
        mechanism_necessity_violations = validator.validate_mechanism_necessity()
        distractor_safety_violations = validator.validate_distractor_safety()
        chain_ordering_valid = validator.validate_chain_ordering()

    dimensions = score.dimensions_by_name
    static_score_unweighted = float(sum(dimensions.values()))
    inputs_hash = _stable_hash(
        {
            "task": spec.to_dict(),
            "config": scorer_config.to_dict(),
            "scorer_version": SCORER_VERSION,
        }
    )

    greedy_solvability = _compute_greedy_solvability(spec) if schema_valid else None

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
            "greedy_solvability": greedy_solvability,
        },
        calibration_version=scorer_config.version,
        inputs_hash=inputs_hash,
    )


def _task_spec_from_payload(data: dict[str, Any]) -> TaskSpecification:
    if "task_spec" in data and isinstance(data["task_spec"], dict):
        return TaskSpecification.from_dict(data["task_spec"])
    if "TaskSpecification" in data and isinstance(data["TaskSpecification"], dict):
        return TaskSpecification.from_dict(data)
    required_fields = {"task_id", "maze", "goal", "max_steps"}
    if not required_fields.issubset(data):
        raise ValueError(
            "Input JSON is not a task artifact. Expected task fields or a nested task_spec."
        )
    return TaskSpecification.from_dict(data)


def score_task_file(
    task_path: str | Path,
    output_dir: str | Path | None = None,
    config: ScorerConfig | None = None,
) -> tuple[CanonicalPathReport, StaticScoreArtifact]:
    """Score a task JSON file and optionally write canonical score artifacts."""
    spec = _task_spec_from_payload(_load_json(task_path))
    canonical_paths = compute_canonical_paths(spec)
    static_score = compute_static_score_artifact(spec, config=config)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _dump_json(out / "canonical_paths.json", canonical_paths.to_dict())
        _dump_json(out / "scored_static.json", static_score.to_dict())

    return canonical_paths, static_score


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

    total = 0
    found = False
    for item in run.get("trajectory", []):
        if not isinstance(item, dict):
            continue
        for key in ("tokens", "token_count"):
            if item.get(key) is not None:
                total += int(item[key])
                found = True
        info = item.get("info")
        if isinstance(info, dict):
            for key in ("tokens", "token_count", "model_tokens"):
                if info.get(key) is not None:
                    total += int(info[key])
                    found = True
    return total if found else None


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


def _extract_canonical_positions(canonical_paths: dict[str, Any]) -> list[tuple[int, int]]:
    bfs = canonical_paths.get("bfs", canonical_paths)
    positions = []
    for pos in bfs.get("positions", []):
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
        return 0.0
    return float(value)


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
    token_efficiency = 1.0
    if token_count is not None:
        token_efficiency = min(1.0, scorer_config.baseline_tokens / max(float(token_count), 1.0))

    static_composite = _extract_static_score(static_data)
    normalizer = (
        difficulty_max_static_score
        or scorer_config.difficulty_max_static_score
        or static_composite
    )
    difficulty_weight = static_composite / normalizer if normalizer and normalizer > 0 else 0.0
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
        "cell_overlap_greedy": 0.0,
        "token_efficiency": token_efficiency,
        "distractor_interactions": int(run.get("distractor_interactions", 0)),
        "irreversible_failures": int(run.get("irreversible_failures", 0)),
        "difficulty_weight": difficulty_weight,
        "efficiency_factor": efficiency_factor,
        "greedy_penalty": greedy_penalty,
        "path_choice": run.get("path_choice"),
        "mechanism_interaction_order": run.get("mechanism_interaction_order", []),
        "failure_point": run.get("failure_point"),
    }

    inputs_hash = _stable_hash(
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
    run = _load_json(run_path)
    static_score = _load_json(static_score_path)
    canonical_paths = _load_json(canonical_paths_path)
    score = compute_runtime_score(
        run,
        static_score=static_score,
        canonical_paths=canonical_paths,
        config=config,
        difficulty_max_static_score=difficulty_max_static_score,
    )
    if output_path is not None:
        _dump_json(output_path, score.to_dict())
    return score
