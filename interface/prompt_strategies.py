"""Prompt strategies for the NLU interface."""

from __future__ import annotations

from gridworld.backends.base import GridState
from interface.coords import inventory_list
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
            system_templates.MIN_TASK_PREFIX,
            system_templates.VALID_ACTIONS_TEMPLATE.format(actions_hint=self._actions_hint),
        ]
        return "\n".join(chunks)

    def build_user_prompt(
        self,
        obs_text: str,
        history_text: str,
        state: GridState,
        *,
        observation: str = "image_only",
    ) -> str:
        return _build_user_prompt(
            observation=observation,
            obs_text=obs_text,
            history_text=history_text,
            state=state,
        )


class StandardPromptStrategy(MinimalPromptStrategy):
    def build_system_prompt(self, querying_suffix: str = "") -> str:
        del querying_suffix
        chunks = [
            system_templates.TASK_PREFIX,
            MECHANISM_LIST,
            system_templates.VALID_ACTIONS_TEMPLATE.format(actions_hint=self._actions_hint),
        ]
        return "\n".join(chunks)


class VerbosePromptStrategy(StandardPromptStrategy):
    def build_system_prompt(self, querying_suffix: str = "") -> str:
        del querying_suffix
        std = StandardPromptStrategy.build_system_prompt(self).rstrip()
        return "\n\n".join([std, MECHANISM_RULES])


class TextInitialMazePromptStrategy(StandardPromptStrategy):
    """Standard system prompt plus the initial maze section placeholder.

    This strategy returns the standard system prompt and appends the
    `INITIAL_MAZE_SECTION` template (containing the `{maze_text}` placeholder).
    The caller (for example `ExperimentRunner.build_prompt_message`) is
    responsible for formatting `{maze_text}` with the rendered maze text.
    """

    def build_system_prompt(self, querying_suffix: str = "") -> str:
        del querying_suffix
        std = StandardPromptStrategy.build_system_prompt(self).rstrip()
        return "\n\n".join([std, system_templates.INITIAL_MAZE_SECTION])


PromptStrategy = MinimalPromptStrategy


def _with_history(prompt: str, history_text: str) -> str:
    if not history_text:
        return prompt
    return f"{history_text}\n\n{prompt}"


def _text_section(text: str) -> str:
    if not text:
        return ""
    return f"{text.rstrip()}\n\n"


def _build_user_prompt(
    *,
    observation: str,
    obs_text: str,
    history_text: str,
    state: GridState,
) -> str:
    inventory = ", ".join(inventory_list(state)) or "empty"
    fields = {
        "current_image": user_templates.CURRENT_IMAGE_PLACEHOLDER,
        "inventory": inventory,
        "current_observation_text": _text_section(obs_text),
    }
    if observation == "text_only":
        prompt = user_templates.TEXT_ONLY_USER_PROMPT.format(**fields)
    elif observation == "image_text":
        prompt = user_templates.IMAGE_TEXT_USER_PROMPT.format(**fields)
    else:
        prompt = user_templates.STANDARD_IMAGE_ONLY_USER_PROMPT.format(**fields)
    return _with_history(prompt, history_text)
