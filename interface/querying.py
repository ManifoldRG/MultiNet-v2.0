from __future__ import annotations

import re
from typing import List, Literal

from interface.parser import (
    VALID_ACTIONS,
    _SYNONYMS,
    normalize_action,
    parse_final_output,
)
from prompting_experiments.prompt_templates import querying as querying_templates

QueryingKind = Literal["step_by_step", "subgoal", "full_trajectory"]

_SUBGOAL_RE = re.compile(r"(?i)SUB_GOAL\s*:\s*(.+)")
# Line-anchored: a mid-reasoning "... actions: MOVE_FORWARD x 5" must never be
# read as the answer. Only a line that *starts* with ACTIONS: qualifies, and
# the last such line wins (models revise plans mid-reply).
_ACTIONS_LINE_RE = re.compile(r"(?im)^\s*ACTIONS\s*:\s*(.+)$")


class QueryingMode:
    def __init__(
        self,
        kind: QueryingKind,
        *,
        valid_actions: set = VALID_ACTIONS,
        synonyms: dict = _SYNONYMS,
    ) -> None:
        self.kind = kind
        self._valid_actions = valid_actions
        self._synonyms = synonyms
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
            out = parse_final_output(
                model_text,
                valid_actions=self._valid_actions,
                synonyms=self._synonyms,
            )
            return [out[0]] if out else []

        m = _SUBGOAL_RE.search(model_text)
        self.current_subgoal = m.group(1).strip() if m else ""

        # FINAL_OUTPUT is the designated answer line for every query mode, so
        # it is parsed first; ACTIONS: lines are a legacy fallback. Searching
        # ACTIONS anywhere first let reasoning prose preempt valid plans (28%
        # of full_trajectory queries were rejected that way).
        actions = parse_final_output(
            model_text,
            allow_regex_fallback=False,
            valid_actions=self._valid_actions,
            synonyms=self._synonyms,
        ) or []
        if not actions:
            for m2 in _ACTIONS_LINE_RE.finditer(model_text):
                parsed = [
                    a
                    for a in (
                        normalize_action(t, valid_actions=self._valid_actions)
                        for t in m2.group(1).split(",")
                    )
                    if a
                ]
                if parsed:
                    actions = parsed
        if not actions:
            actions = parse_final_output(
                model_text,
                valid_actions=self._valid_actions,
                synonyms=self._synonyms,
            ) or []

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

    def final_output_instruction(self) -> str:
        if self.kind == "full_trajectory":
            return querying_templates.FULL_TRAJECTORY_FINAL_OUTPUT_INSTRUCTION
        if self.kind == "subgoal":
            return querying_templates.SUBGOAL_FINAL_OUTPUT_INSTRUCTION
        return querying_templates.SINGLE_ACTION_FINAL_OUTPUT_INSTRUCTION

    def system_prompt_suffix(self) -> str:
        return ""

    def step_metadata(self) -> dict:
        if self.kind == "step_by_step":
            return {}
        meta = {"subgoal": self.current_subgoal}
        if self.kind == "full_trajectory":
            meta["full_trajectory"] = True
        return meta
