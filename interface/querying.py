from __future__ import annotations

import re
from typing import List, Literal

from interface.parser import normalize_action, parse_final_output
from prompting_experiments.prompt_templates import querying as querying_templates

QueryingKind = Literal["step_by_step", "subgoal", "full_trajectory"]

_SUBGOAL_RE = re.compile(r"(?i)SUB_GOAL\s*:\s*(.+)")
_ACTIONS_RE = re.compile(r"(?i)ACTIONS\s*:\s*(.+)")


class QueryingMode:
    def __init__(self, kind: QueryingKind) -> None:
        self.kind = kind
        self.current_subgoal = ""
        self._trajectory_loaded = False

    def reset(self) -> None:
        self.current_subgoal = ""
        self._trajectory_loaded = False

    def should_query(self, queue, failures) -> bool:
        if self.kind == "step_by_step":
            return not queue
        if self.kind == "subgoal":
            return not queue or failures >= 3
        return not self._trajectory_loaded and not queue

    def parse_actions(self, model_text: str) -> List[str]:
        if self.kind == "step_by_step":
            out = parse_final_output(model_text)
            return [out[0]] if out else []

        m = _SUBGOAL_RE.search(model_text)
        self.current_subgoal = m.group(1).strip() if m else ""

        m2 = _ACTIONS_RE.search(model_text)
        if m2:
            actions = [a for a in (normalize_action(t) for t in m2.group(1).split(",")) if a]
        else:
            out = parse_final_output(model_text)
            actions = out if out else []

        if self.kind == "full_trajectory" and actions:
            self._trajectory_loaded = True
        return actions

    def user_prompt_suffix(self) -> str:
        if self.kind == "step_by_step":
            return ""
        if self.kind == "subgoal":
            return querying_templates.SUBGOAL_SUFFIX
        return querying_templates.FULL_TRAJECTORY_SUFFIX

    def user_prompt_question(self) -> str:
        if self.kind == "full_trajectory":
            return querying_templates.FULL_TRAJECTORY_QUESTION
        return ""

    def system_prompt_suffix(self) -> str:
        return ""

    def step_metadata(self) -> dict:
        if self.kind == "step_by_step":
            return {}
        meta = {"subgoal": self.current_subgoal}
        if self.kind == "full_trajectory":
            meta["full_trajectory"] = True
        return meta
