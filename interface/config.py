from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Literal, Optional, get_args, get_origin, get_type_hints


DEFAULT_MAX_HISTORY_TOKENS = 1_000_000


@dataclass
class ExperimentConfig:
    """Selects one implementation along each experimental axis."""

    prompting: Literal["minimal", "standard", "verbose", "text_initial_maze"] = "standard"
    observation: Literal["text_only", "image_text", "image_only"] = "image_text"
    observation_text_format: Literal["coords", "json", "ascii"] = "coords"
    include_current_observation_description: bool = True
    observation_text_includes_facing: bool = True
    context_window: Literal[
        "current",
        "last3",
        "last_n",
        "text_summary",
        "text_summary_and_last3",
        "text_summary_and_last_n",
        "full",
    ] = "last_n"
    context_n: int = 3
    max_history_tokens: int = DEFAULT_MAX_HISTORY_TOKENS
    querying: Literal["step_by_step", "subgoal", "full_trajectory"] = "step_by_step"
    chat_history: Literal["stateless", "rolling", "full"] = "stateless"
    chat_turns_max: int = 3
    max_parse_retries: int = 3
    in_context_learning: Literal["zero_shot", "one_shot"] = "one_shot"
    action_space: Literal["egocentric", "cardinal"] = "egocentric"
    feedback: Literal["minimal", "standard", "causal"] = "minimal"
    progress_stall_k: Optional[int] = None

    def __post_init__(self) -> None:
        hints = get_type_hints(type(self))
        for f in fields(self):
            hint = hints.get(f.name)
            if get_origin(hint) is not Literal:
                continue
            allowed = get_args(hint)
            value = getattr(self, f.name)
            if value not in allowed:
                raise ValueError(f"{f.name} must be one of {allowed}, got {value!r}")
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
        if (
            isinstance(self.context_n, bool)
            or not isinstance(self.context_n, int)
            or self.context_n <= 0
        ):
            raise ValueError(
                f"context_n must be a positive int, got {self.context_n!r}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        d = dict(d)
        w = d.get("context_window")
        if w == "last3":
            d["context_window"] = "last_n"
        elif w == "text_summary_and_last3":
            d["context_window"] = "text_summary_and_last_n"
        return cls(**d)
