from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass
class ExperimentConfig:
    """Selects one implementation along each experimental axis."""

    prompting: Literal["minimal", "standard", "verbose"] = "standard"
    observation: Literal["text_only", "image_text", "image_only"] = "text_only"
    context_window: Literal["current", "last3"] = "current"
    querying: Literal["step_by_step", "subgoal", "full_trajectory"] = "step_by_step"
    chat_history: Literal["stateless", "rolling", "full"] = "rolling"
    chat_turns_max: int = 3

    def to_dict(self) -> dict:
        return asdict(self)
