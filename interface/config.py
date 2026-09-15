from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional


@dataclass
class ExperimentConfig:
    """Selects one implementation along each experimental axis."""

    prompting: Literal["minimal", "standard", "verbose", "text_initial_maze"] = "standard"
    observation: Literal["text_only", "image_text", "image_only"] = "image_text"
    include_current_observation_description: bool = True
    observation_text_includes_facing: bool = True
    context_window: Literal[
        "current", "last3", "text_summary", "text_summary_and_last3", "full"
    ] = "last3"
    max_history_tokens: int = 100_000
    querying: Literal["step_by_step", "subgoal", "full_trajectory"] = "step_by_step"
    chat_history: Literal["stateless", "rolling", "full"] = "stateless"
    chat_turns_max: int = 3
    max_parse_retries: int = 3
    in_context_learning: Literal["zero_shot", "one_shot"] = "one_shot"
    action_space: Literal["egocentric", "cardinal"] = "egocentric"
    progress_stall_k: Optional[int] = None

    def __post_init__(self) -> None:
        k = self.progress_stall_k
        if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k <= 0):
            raise ValueError(
                f"progress_stall_k must be None or a positive int, got {k!r}"
            )
        if (
            isinstance(self.max_history_tokens, bool)
            or not isinstance(self.max_history_tokens, int)
            or self.max_history_tokens <= 0
        ):
            raise ValueError(
                "max_history_tokens must be a positive int, "
                f"got {self.max_history_tokens!r}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(**d)
