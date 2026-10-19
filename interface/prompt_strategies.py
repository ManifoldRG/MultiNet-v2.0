"""Prompt strategies for the NLU interface."""

from __future__ import annotations

from dataclasses import dataclass

from gridworld.backends.base import GridState
from gridworld.render3d.cameras import EYE_PRESETS, PRESETS  # pure pose math; no mujoco
from gridworld.render3d.hud import turns_with_agent
from interface.coords import inventory_list
from prompting_experiments.prompt_templates import system as system_templates
from prompting_experiments.prompt_templates import user as user_templates

MECHANISM_LIST = system_templates.MECHANISM_LIST
MECHANISM_RULES = system_templates.MECHANISM_RULES


@dataclass(frozen=True)
class RenderContext:
    """How the model's frames are drawn, read off the backend (build_runner)
    rather than ExperimentConfig, so config hashes never move. ``camera`` is
    the 3D preset (None = the 2D grid frame); ``start_map``: the backend
    snapshots a top-down map of the reset state (run-config render.start_map)."""

    camera: str | None = None
    start_map: bool = False

    def __post_init__(self) -> None:
        if self.camera is not None and self.camera not in PRESETS:
            raise ValueError(f"unknown camera preset {self.camera!r}; choose from {PRESETS}")
        if self.start_map and self.camera is None:
            raise ValueError("start_map needs a 3D camera")
        if self.start_map and self.camera not in user_templates.CURRENT_VIEW_NAMES:
            raise ValueError(f"start_map is not offered with the {self.camera} camera (top_down is the map)")

    @classmethod
    def from_backend(cls, backend) -> RenderContext:
        camera = getattr(backend, "camera", None)  # only the 3D backend has one
        if not isinstance(camera, str):
            return cls()
        return cls(camera=camera, start_map=bool(getattr(backend, "start_map", False)))

    @property
    def first_person(self) -> bool:
        return self.camera in EYE_PRESETS

    @property
    def view_turns_with_agent(self) -> bool:
        return self.camera is not None and turns_with_agent(self.camera)


class MinimalPromptStrategy:
    def __init__(self, actions_hint: str, render: RenderContext | None = None) -> None:
        self._actions_hint = actions_hint
        self._render = render or RenderContext()

    @property
    def render(self) -> RenderContext:
        return self._render

    def start_map_text(self) -> str | None:
        """The text opening the start-map block; None without a start map."""
        return user_templates.START_MAP_MINIMAL if self._render.start_map else None

    def current_view_label(self) -> str | None:
        """Text right before the current image; minimal keeps today's layout."""
        return None

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
            view_label=self.current_view_label(),
        )


class StandardPromptStrategy(MinimalPromptStrategy):
    def build_system_prompt(self, querying_suffix: str = "") -> str:
        del querying_suffix
        chunks = [
            self._task_prefix(),
            MECHANISM_LIST,
            system_templates.VALID_ACTIONS_TEMPLATE.format(actions_hint=self._actions_hint),
        ]
        return "\n".join(chunks)

    def _task_prefix(self) -> str:
        """TASK_PREFIX describes the 2D triangle; 3D names what each camera
        family actually shows."""
        if self._render.camera is None:
            return system_templates.TASK_PREFIX
        if self._render.first_person:
            return system_templates.TASK_PREFIX_3D_FIRST_PERSON
        return system_templates.TASK_PREFIX_3D_THIRD_PERSON

    def start_map_text(self) -> str | None:
        return user_templates.START_MAP_STANDARD if self._render.start_map else None

    def current_view_label(self) -> str | None:
        if not self._render.start_map:
            return None
        return user_templates.CURRENT_VIEW_LABEL.format(
            view=user_templates.CURRENT_VIEW_NAMES[self._render.camera]
        )


class VerbosePromptStrategy(StandardPromptStrategy):
    def build_system_prompt(self, querying_suffix: str = "") -> str:
        del querying_suffix
        std = StandardPromptStrategy.build_system_prompt(self).rstrip()
        return "\n\n".join([std, MECHANISM_RULES])

    def start_map_text(self) -> str | None:
        return user_templates.START_MAP_VERBOSE if self._render.start_map else None

    def current_view_label(self) -> str | None:
        label = super().current_view_label()
        if label is None or not self._render.view_turns_with_agent:
            return label
        return f"{user_templates.COMPASS_NOTE}\n{label}"


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
    view_label: str | None = None,
) -> str:
    inventory = ", ".join(inventory_list(state)) or "empty"
    current_image = user_templates.CURRENT_IMAGE_PLACEHOLDER
    if view_label:
        current_image = f"{view_label}\n{current_image}"
    fields = {
        "current_image": current_image,
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
