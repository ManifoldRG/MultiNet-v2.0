"""Gridworld domain for MultiNet-v2.0.

This module provides task schema and validation utilities for gridworld
puzzle specifications.
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
]
