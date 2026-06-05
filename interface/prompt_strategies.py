"""Prompt strategies for the NLU interface."""

from __future__ import annotations

from gridworld.backends.base import GridState
from gridworld.task_spec import TaskSpecification

from interface.coords import (
    agent_facing,
    agent_row_col,
    goal_row_col,
)
from prompting_experiments.prompt_templates import system as system_templates
from prompting_experiments.prompt_templates import user as user_templates

MECHANISM_LIST = system_templates.MECHANISM_LIST
MECHANISM_RULES = system_templates.MECHANISM_RULES


class MinimalPromptStrategy:
    def __init__(self, actions_hint: str) -> None:
        self._actions_hint = actions_hint

    def build_system_prompt(self, querying_suffix: str = "") -> str:
        del querying_suffix
        chunks = [
            system_templates.TASK_PREFIX,
            MECHANISM_LIST,
            system_templates.VALID_ACTIONS_TEMPLATE.format(actions_hint=self._actions_hint),
        ]
        return "\n".join(chunks)

    def build_user_prompt(
        self,
        obs_text: str,
        history_text: str,
        task_spec: TaskSpecification,
        state: GridState,
        last_feedback: str,
        *,
        include_status_footer: bool = False,
    ) -> str:
        obs_block = (
            user_templates.OBSERVATION_SECTION.format(obs_text=obs_text)
            if obs_text
            else ""
        )
        pos = agent_row_col(state)
        goal = goal_row_col(task_spec)
        status_block = _status_block(
            include_status_footer,
            position=pos,
            facing=agent_facing(state),
            goal=goal,
            last_feedback=last_feedback,
        )
        prompt = user_templates.STANDARD_USER_PROMPT.format(
            obs_block=obs_block,
            status_block=status_block,
        )
        return _with_history(prompt, history_text)


class StandardPromptStrategy(MinimalPromptStrategy):
    pass


class VerbosePromptStrategy(StandardPromptStrategy):
    include_mechanism_hints = False

    def build_system_prompt(self, querying_suffix: str = "") -> str:
        del querying_suffix
        std = StandardPromptStrategy.build_system_prompt(self).rstrip()
        return "\n\n".join([std, MECHANISM_RULES])

    def build_user_prompt(
        self,
        obs_text: str,
        history_text: str,
        task_spec: TaskSpecification,
        state: GridState,
        last_feedback: str,
        *,
        include_status_footer: bool = False,
    ) -> str:
        mechanism_block = (
            _mechanism_hints_text(task_spec) if self.include_mechanism_hints else ""
        )
        obs_block = (
            user_templates.OBSERVATION_SECTION.format(obs_text=obs_text)
            if obs_text
            else ""
        )
        pos = agent_row_col(state)
        goal = goal_row_col(task_spec)
        status_block = _status_block(
            include_status_footer,
            position=pos,
            facing=agent_facing(state),
            goal=goal,
            last_feedback=last_feedback,
        )

        prompt = user_templates.VERBOSE_USER_PROMPT.format(
            obs_block=obs_block,
            mechanism_block=mechanism_block,
            status_block=status_block,
        )
        return _with_history(prompt, history_text)


PromptStrategy = MinimalPromptStrategy


def _with_history(prompt: str, history_text: str) -> str:
    if not history_text:
        return prompt
    return f"{history_text}\n\n{prompt}"


def _status_block(
    include: bool,
    *,
    position: tuple[int, int],
    facing: str,
    goal: tuple[int, int],
    last_feedback: str,
) -> str:
    if not include:
        return ""
    return user_templates.STATUS_BLOCK.format(
        position=position,
        facing=facing,
        goal=goal,
        last_feedback=last_feedback,
    )


def _mechanism_hints_text(task_spec: TaskSpecification) -> str:
    lines = []
    if task_spec.mechanisms.keys or task_spec.mechanisms.doors:
        lines.append(user_templates.KEY_DOOR_HINT)
    if task_spec.mechanisms.switches or task_spec.mechanisms.gates:
        lines.append(user_templates.SWITCH_GATE_HINT)
    if not lines:
        return ""
    return user_templates.MECHANISM_HINTS_HEADER + "\n".join(lines) + "\n"
