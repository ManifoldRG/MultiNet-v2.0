"""Dataclasses for scorer artifact payloads."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .config import DIMENSION_NAMES, SCORER_VERSION


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
            "dimensions": list(self.dimensions),
            "dimension_names": list(self.dimension_names),
            "composite": self.composite,
            "weights": list(self.weights),
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
    greedy: dict[str, Any] | None = None
    producer_version: str = SCORER_VERSION

    @property
    def bfs(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "actions": list(self.actions),
            "positions": [list(pos) for pos in self.positions],
            "optimal_steps": self.optimal_steps,
            "states_explored": self.states_explored,
            "message": self.message,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "task_id": self.task_id,
            "bfs": self.bfs,
            "producer_version": self.producer_version,
        }
        if self.greedy is not None:
            payload["greedy"] = copy.deepcopy(self.greedy)
        return payload

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
            greedy=copy.deepcopy(data.get("greedy")),
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
            "static_score_unweighted": self.static_score_unweighted,
            "static_score": self.static_score,
            "weights": dict(self.weights),
            "validation": copy.deepcopy(self.validation),
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
            "signals": copy.deepcopy(self.signals),
            "composite": self.composite,
            "calibration_version": self.calibration_version,
            "inputs_hash": self.inputs_hash,
            "producer_version": self.producer_version,
        }
