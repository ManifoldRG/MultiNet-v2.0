from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass
class ExperimentConfig:
    """Selects one implementation along each experimental axis.

    prompting
        minimal   – goal + action list only (system prompt)
        standard  – adds ``MECHANISM_LIST`` to the system prompt
        verbose   – standard + ``MECHANISM_RULES`` + extra user fields (neighbours, hints).
        Maze **layout** text is in the system / user split from ``observation``, not from prompting.

    observation
        text_only        – initial NL maze in system; current situation text per user turn; last3 history
        image_text       – same as text_only + live PNG each turn; last3 = full feedback
        image_only       – live PNG only (no NL map); last3 = prior decision-frame PNGs + ``Action: …`` only (multimodal)

    context_window
        current  – only the current observation (no prior steps in the prompt)
        last3    – last 3 steps as structured lines prepended to the prompt (default)

    querying
        step_by_step    – one LLM call per env step (only the first action in FINAL_OUTPUT is used)
        subgoal         – SUB_GOAL + ACTIONS list; re-queries when queue empty, stuck, or mid-budget
        full_trajectory – same format as subgoal, but exactly one LLM call per episode (no re-query)

    chat_history
        stateless – each API call is only ``[system, current_user]`` (cheapest; no prior assistant text).
        rolling   – append user/assistant each query, but keep at most ``chat_turns_max`` prior
                    user+assistant **pairs** after ``system`` (default: good balance for vision + reasoning).
        full      – append without trimming (original behavior; high token use with images).

    chat_turns_max
        For ``chat_history=\"rolling\"``: max number of full ``(user, assistant)`` rounds kept in the API
        payload after ``system``. Ignored for ``stateless`` / ``full``.
    """

    prompting: Literal["minimal", "standard", "verbose"] = "standard"
    observation: Literal["text_only", "image_text", "image_only"] = "image_only"
    context_window: Literal["current", "last3"] = "current"
    querying: Literal["step_by_step", "subgoal", "full_trajectory"] = "step_by_step"
    chat_history: Literal["stateless", "rolling", "full"] = "rolling"
    chat_turns_max: int = 3

    def to_dict(self) -> dict:
        return asdict(self)
