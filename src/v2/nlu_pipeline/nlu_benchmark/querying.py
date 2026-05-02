"""Querying modes for the NLU benchmark.

A single `QueryingMode` class covers all three behaviours; only `should_query()`
and a few small details differ:

  step_by_step    — one LLM call per env step: queue holds at most one action
                    (only the first action from FINAL_OUTPUT is used; then re-query).
  subgoal         — same output format as full trajectory, but re-query when the
                    queue runs out, after failures, or mid-episode.
  full_trajectory — one query per episode; same SUB_GOAL / ACTIONS format
                      (or FINAL_OUTPUT: … as fallback, like step_by_step).

The episode loop lives in ExperimentRunner.run() (runner.py), not here.
"""

from __future__ import annotations

import re
from typing import List, Literal

from nlu_benchmark.parser import normalize_action, parse_final_output

QueryingKind = Literal["step_by_step", "subgoal", "full_trajectory"]

_SUBGOAL_RE = re.compile(r"(?i)SUB_GOAL\s*:\s*(.+)")
_ACTIONS_RE = re.compile(r"(?i)ACTIONS\s*:\s*(.+)")


class QueryingMode:
    """When to call the model and how to parse its reply."""

    def __init__(self, kind: QueryingKind) -> None:
        self.kind               = kind
        self.current_subgoal    = ""
        self._trajectory_loaded = False

    def reset(self) -> None:
        self.current_subgoal    = ""
        self._trajectory_loaded = False

    def should_query(self, queue, failures) -> bool:
        if self.kind == "step_by_step":
            # With at most one queued action (see parse_actions), this is true after each step.
            return not queue
        if self.kind == "subgoal":
            return not queue or failures >= 3
        # full_trajectory
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

    def system_prompt_suffix(self) -> str:
        if self.kind == "step_by_step":
            return ""
        if self.kind == "subgoal":
            return (
                "For each turn output:\n"
                "  SUB_GOAL: <short description of your next waypoint>\n"
                "  ACTIONS: <comma-separated action list to reach it>"
            )
        return (
            "Output your complete trajectory once as:\n"
            "  SUB_GOAL: <short description of the full plan>\n"
            "  ACTIONS: <comma-separated action list from start to finish>\n"
            "The last action in ACTIONS should be DONE (when you expect to be at the goal).\n"
            "You will not be queried again — this is your only planning turn."
        )

    def step_metadata(self) -> dict:
        if self.kind == "step_by_step":
            return {}
        meta = {"subgoal": self.current_subgoal}
        if self.kind == "full_trajectory":
            meta["full_trajectory"] = True
        return meta
