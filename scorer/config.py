"""Scorer configuration and calibration defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io import load_json


SCORER_VERSION = "0.3.0"
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

GREEDY_SOLVABILITY_FEATURE = "greedy_solvability"

CANONICAL_AGENT_FEATURE_NAMES = [
    GREEDY_SOLVABILITY_FEATURE,
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


def _coerce_float_mapping(
    values: dict[str, Any] | list[Any] | None,
    names: list[str],
    default: float = 1.0,
) -> dict[str, float]:
    if values is None:
        return {name: default for name in names}
    if isinstance(values, list):
        if len(values) != len(names):
            raise ValueError(f"Expected {len(names)} weights, got {len(values)}")
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
        if path is not None:
            raise FileNotFoundError(f"Scorer config not found: {config_path}")
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
    return ScorerConfig.from_dict(load_json(config_path))
