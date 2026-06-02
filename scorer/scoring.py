"""Public scorer interface for static and runtime analysis."""

from __future__ import annotations

from .aggregate import build_experiment_reports, write_experiment_reports
from .artifacts import (
    CanonicalPathReport,
    RuntimeScoreArtifact,
    ScoredDifficulty,
    StaticScoreArtifact,
)
from .config import (
    CANONICAL_AGENT_FEATURE_NAMES,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DISTRACTOR_TYPE_WEIGHTS,
    DEFAULT_RUNTIME_WEIGHTS,
    DIMENSION_NAMES,
    SCORER_VERSION,
    ScorerConfig,
    load_scorer_config,
)
from .runtime import compute_runtime_score, score_runtime_file
from .solvers import compute_canonical_paths, compute_greedy_solvability
from .static import compute_12d_score, compute_static_score_artifact, score_task_file

__all__ = [
    "CANONICAL_AGENT_FEATURE_NAMES",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_DISTRACTOR_TYPE_WEIGHTS",
    "DEFAULT_RUNTIME_WEIGHTS",
    "DIMENSION_NAMES",
    "SCORER_VERSION",
    "CanonicalPathReport",
    "RuntimeScoreArtifact",
    "ScoredDifficulty",
    "ScorerConfig",
    "StaticScoreArtifact",
    "compute_12d_score",
    "compute_canonical_paths",
    "compute_greedy_solvability",
    "compute_runtime_score",
    "compute_static_score_artifact",
    "load_scorer_config",
    "score_runtime_file",
    "score_task_file",
    "build_experiment_reports",
    "write_experiment_reports",
]
