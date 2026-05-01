"""Gridworld domain for MultiNet-v2.0.

This module provides task schema, validation, scoring, parsing, and backend
integration utilities for gridworld puzzle specifications.
"""

from .bootstrap import disable_gymnasium_env_plugins

disable_gymnasium_env_plugins()

from .task_spec import (
    Position,
    KeySpec,
    DoorSpec,
    SwitchSpec,
    GateSpec,
    BlockSpec,
    HazardSpec,
    TeleporterSpec,
    DependencyStep,
    DependencyChain,
    Distractor,
    MazeLayout,
    MechanismSet,
    Rules,
    GoalSpec,
    TaskSpecification,
)
from .task_validator import (
    DifficultyReport,
    FragilityReport,
    TaskValidator,
    compute_difficulty,
)
from .scoring import ScoredDifficulty, compute_12d_score
from .task_parser import TaskParser
from .actions import MiniGridActions, ACTION_NAMES, ACTION_DESCRIPTIONS


__all__ = [
    # Task specification
    "Position",
    "KeySpec",
    "DoorSpec",
    "SwitchSpec",
    "GateSpec",
    "BlockSpec",
    "HazardSpec",
    "TeleporterSpec",
    "DependencyStep",
    "DependencyChain",
    "Distractor",
    "MazeLayout",
    "MechanismSet",
    "Rules",
    "GoalSpec",
    "TaskSpecification",
    # Validation and scoring
    "TaskValidator",
    "DifficultyReport",
    "FragilityReport",
    "compute_difficulty",
    "ScoredDifficulty",
    "compute_12d_score",
    # Parser
    "TaskParser",
    # Actions
    "MiniGridActions",
    "ACTION_NAMES",
    "ACTION_DESCRIPTIONS",
]
