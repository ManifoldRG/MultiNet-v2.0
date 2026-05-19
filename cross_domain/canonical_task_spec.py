"""
Canonical Task Specification

Domain-agnostic representation of tasks that can be mapped to supported
domains. Uses normalized [0,1] coordinates for cross-domain compatibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CanonicalGoal:
    """Domain-agnostic goal specification."""
    goal_type: str  # "reach", "collect", "arrange", "survive"
    target: tuple[float, ...] | None = None  # Normalized position
    target_ids: list[str] = field(default_factory=list)
    target_positions: list[tuple[float, ...]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "goal_type": self.goal_type,
            "target": list(self.target) if self.target else None,
            "target_ids": self.target_ids,
            "target_positions": [list(pos) for pos in self.target_positions],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CanonicalGoal":
        return cls(
            goal_type=d["goal_type"],
            target=tuple(d["target"]) if d.get("target") else None,
            target_ids=d.get("target_ids", []),
            target_positions=[
                tuple(pos) for pos in d.get("target_positions", [])
            ],
        )


@dataclass
class CanonicalObject:
    """Domain-agnostic object specification."""
    id: str
    obj_type: str   # "barrier", "collectible", "interactive", "hazard", "portal"
    position: tuple[float, ...]  # Normalized [0,1] coordinates
    properties: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "obj_type": self.obj_type,
            "position": list(self.position),
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CanonicalObject":
        return cls(
            id=d["id"],
            obj_type=d["obj_type"],
            position=tuple(d["position"]),
            properties=d.get("properties", {}),
        )


@dataclass
class CanonicalRules:
    """Domain-agnostic rule flags shared by gridworld-like tasks."""
    key_consumption: bool = True
    switch_type: str = "toggle"
    hidden_mechanisms: list[str] = field(default_factory=list)
    observability: str = "full"
    view_size: int = 7

    def to_dict(self) -> dict:
        return {
            "key_consumption": self.key_consumption,
            "switch_type": self.switch_type,
            "hidden_mechanisms": self.hidden_mechanisms,
            "observability": self.observability,
            "view_size": self.view_size,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CanonicalRules":
        return cls(
            key_consumption=d.get("key_consumption", True),
            switch_type=d.get("switch_type", "toggle"),
            hidden_mechanisms=d.get("hidden_mechanisms", []),
            observability=d.get("observability", "full"),
            view_size=d.get("view_size", 7),
        )


@dataclass
class CanonicalTaskSpec:
    """
    Domain-agnostic task specification.

    All positions are normalized to [0,1] for cross-domain compatibility.
    Domain-specific extensions go in domain_config.
    """
    task_id: str
    seed: int
    difficulty: int  # 1-5
    dimensions: tuple[float, ...]     # Normalized [0,1]
    agent_start: tuple[float, ...]    # Normalized
    goal: CanonicalGoal               # Domain-agnostic goal
    objects: list[CanonicalObject]     # Domain-agnostic objects
    max_steps: int
    description: str = ""
    rules: CanonicalRules = field(default_factory=CanonicalRules)
    dependency_chain: dict | None = None
    distractors: list[dict] = field(default_factory=list)
    domain_config: dict = field(default_factory=dict)  # Domain-specific extensions

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "difficulty": self.difficulty,
            "dimensions": list(self.dimensions),
            "agent_start": list(self.agent_start),
            "goal": self.goal.to_dict(),
            "objects": [obj.to_dict() for obj in self.objects],
            "max_steps": self.max_steps,
            "description": self.description,
            "rules": self.rules.to_dict(),
            "dependency_chain": self.dependency_chain,
            "distractors": self.distractors,
            "domain_config": self.domain_config,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CanonicalTaskSpec":
        return cls(
            task_id=d["task_id"],
            seed=d["seed"],
            difficulty=d["difficulty"],
            dimensions=tuple(d["dimensions"]),
            agent_start=tuple(d["agent_start"]),
            goal=CanonicalGoal.from_dict(d["goal"]),
            objects=[CanonicalObject.from_dict(o) for o in d.get("objects", [])],
            max_steps=d["max_steps"],
            description=d.get("description", ""),
            rules=CanonicalRules.from_dict(d.get("rules", {})),
            dependency_chain=d.get("dependency_chain"),
            distractors=d.get("distractors", []),
            domain_config=d.get("domain_config", {}),
        )

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "CanonicalTaskSpec":
        with open(path) as f:
            return cls.from_dict(json.load(f))
