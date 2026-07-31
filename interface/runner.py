"""ExperimentRunner — LLM episode loop using gridworld MiniGridBackend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List

import numpy as np

from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.task_spec import TaskSpecification

from interface import action_space as action_space_mod
from interface.agents.reply import Reply
from interface.config import ExperimentConfig
from interface.episode_log import state_snapshot
from interface.observation import (
    current_image_blocks,
    current_observation_text,
    history_content_blocks,
    history_text,
    leading_summary_blocks,
)
from interface.prompt_strategies import (
    MinimalPromptStrategy,
    PromptStrategy,
    StandardPromptStrategy,
    VerbosePromptStrategy,
)
from interface.prompt_strategies import TextInitialMazePromptStrategy
from interface.querying import QueryingMode
from interface.renderer import render_initial_maze_text
from prompting_experiments.prompt_templates import querying as querying_templates
from prompting_experiments.prompt_templates import system as system_templates
from prompting_experiments.prompt_templates import user as user_templates

logger = logging.getLogger(__name__)


_PROMPT_STRATEGIES = {
    "minimal": MinimalPromptStrategy,
    "standard": StandardPromptStrategy,
    "verbose": VerbosePromptStrategy,
    "text_initial_maze": TextInitialMazePromptStrategy,
}


def _user_message_has_image(message: dict) -> bool:
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(b, dict) and b.get("type") == "image_url" for b in content)


def _trim_rolling_chat(messages: List[dict], max_pairs: int) -> None:
    tail_len = len(messages) - 1
    cap = 2 * max_pairs
    if tail_len > cap:
        del messages[1 : 1 + (tail_len - cap)]


def _reset_agent_usage(agent: Callable[[List[dict]], str]) -> None:
    """Clear per-call telemetry so stale usage cannot leak into a later query."""
    reset_usage = getattr(agent, "reset_usage", None)
    if callable(reset_usage):
        reset_usage()
        return
    try:
        setattr(agent, "last_usage", None)
    except (AttributeError, TypeError):
        pass


def _replace_current_question(prompt_text: str, question: str) -> str:
    standard_question = user_templates.NEXT_ACTION_QUESTION
    before, match, after = prompt_text.rpartition(standard_question)
    if not match:
        return prompt_text
    return f"{before}{question}{after}"


def _append_after_current_question(prompt_text: str, instruction: str) -> str:
    questions = (
        querying_templates.FULL_TRAJECTORY_QUESTION,
        user_templates.NEXT_ACTION_QUESTION,
    )
    for question in questions:
        before, match, after = prompt_text.rpartition(question)
        if match:
            return f"{before}{match}\n\n{instruction}{after}"
    return f"{prompt_text}\n\n{instruction}"


def _expand_current_image_placeholder(prompt_text: str, images: list[dict]) -> list[dict]:
    placeholder = user_templates.CURRENT_IMAGE_PLACEHOLDER
    if placeholder not in prompt_text:
        return [{"type": "text", "text": prompt_text}]

    blocks: list[dict] = []
    parts = prompt_text.split(placeholder)
    for idx, part in enumerate(parts):
        if part:
            blocks.append({"type": "text", "text": part.lstrip("\n") if idx else part})
        if idx < len(parts) - 1:
            blocks.extend(images)
    return blocks


def build_runner(
    config: ExperimentConfig,
    backend: MiniGridBackend,
    task_spec: TaskSpecification,
) -> ExperimentRunner:
    space = config.action_space
    return ExperimentRunner(
        backend=backend,
        task_spec=task_spec,
        config=config,
        prompt=_PROMPT_STRATEGIES[config.prompting](action_space_mod.actions_hint(space)),
        querying=QueryingMode(
            config.querying,
            valid_actions=action_space_mod.valid_actions(space),
            synonyms=action_space_mod.synonyms(space),
        ),
    )


class ExperimentRunner:
    def __init__(
        self,
        backend: MiniGridBackend,
        task_spec: TaskSpecification,
        config: ExperimentConfig,
        prompt: PromptStrategy,
        querying: QueryingMode,
    ) -> None:
        self.backend = backend
        self.task_spec = task_spec
        self.config = config
        self.prompt = prompt
        self.querying = querying
        self.last_rgb: np.ndarray | None = None

    def build_prompt_message(
        self,
        state,
        last_feedback: str,
        transcript: List[dict],
    ) -> tuple[str, dict]:
        system_prompt = self.prompt.build_system_prompt()
        # If the system prompt includes the `{maze_text}` placeholder, format
        # it with the rendered maze. Otherwise, for text observations append
        # the `INITIAL_MAZE_SECTION` so the maze is present in system-level
        # context for text-only or image+text modes.
        if "{maze_text}" in system_prompt:
            system_prompt = system_prompt.format(maze_text=render_initial_maze_text(self.task_spec))
        elif self.config.observation in ("text_only", "image_text"):
            maze_text = render_initial_maze_text(self.task_spec)
            system_prompt = (
                system_prompt
                + "\n\n"
                + system_templates.INITIAL_MAZE_SECTION.format(maze_text=maze_text)
            )
        return system_prompt, self._build_message(
            state,
            last_feedback,
            transcript,
        )

    def run(
        self,
        agent: Callable[[List[dict]], str],
        *,
        verbose: bool = True,
        maze_path: str | Path | None = None,
    ) -> dict:
        # Lazy import: episode_step imports helpers from this module, so a
        # top-level import here would be circular.
        from interface.episode_step import EpisodeStepper

        stepper = EpisodeStepper(self, verbose=verbose, maze_path=maze_path)
        stepper.start()
        agent_error: str | None = None
        while (messages := stepper.next_query()) is not None:
            _reset_agent_usage(agent)
            try:
                if hasattr(agent, "generate"):
                    reply = agent.generate(messages)
                else:
                    # Legacy test doubles: ``__call__(messages) -> str`` plus the
                    # ``last_usage``/``last_thinking`` side-channels. Shim them into a
                    # Reply so the stepper sees one uniform result type.
                    text = agent(messages)
                    reply = Reply(
                        text=text,
                        usage=getattr(agent, "last_usage", None),
                        thinking=getattr(agent, "last_thinking", None),
                    )
                stepper.apply_reply(reply)
            except Exception as exc:  # noqa: BLE001 — never lose paid work
                # Episode artifacts are only written after this method returns
                # and callers (run_pipeline) have no per-episode isolation, so a
                # dead transport used to discard the whole episode. End it here
                # instead: the partial transcript survives and the end_reason
                # says plainly that this is an infra outcome, not a model one.
                # KeyboardInterrupt/SystemExit are BaseException and still
                # propagate — Ctrl-C must stay an interrupt.
                agent_error = f"agent_error:{type(exc).__name__}: {exc}"[:200]
                logger.error("episode aborted by agent error: %s", exc)
                # next_query already reserved this round's index; roll it back so
                # query_count equals the applied-query count.
                stepper.query_count -= 1
                break
        result = stepper.result()
        if agent_error is not None:
            result["end_reason"] = agent_error
            result["success"] = False
        return result

    def _one_shot_blocks(self, obs) -> list[dict]:
        """The one-shot ICL example blocks (example image + solution), or []."""
        if self.config.in_context_learning == "one_shot":
            from interface.one_shot import one_shot_content_blocks
            return one_shot_content_blocks(obs)
        return []

    def _build_message(
        self,
        state,
        last_feedback: str,
        transcript: List[dict],
        with_one_shot: bool = True,
        with_context_history: bool = True,
    ) -> dict:
        obs = self.config.observation
        # "current" disables the in-prompt history sections; multiturn chat
        # passes with_context_history=False because its turns already carry
        # the history and embedding it again duplicates every observation.
        ctx = self.config.context_window if with_context_history else "current"
        obs_text = current_observation_text(
            obs,
            self.task_spec,
            state,
            include_description=self.config.include_current_observation_description,
            include_facing=self.config.observation_text_includes_facing,
        )
        prompt_text = self.prompt.build_user_prompt(
            obs_text,
            history_text(obs, ctx, transcript, self.task_spec),
            state,
            observation=obs,
        )
        prompt_question = self.querying.user_prompt_question()
        if prompt_question:
            prompt_text = _replace_current_question(prompt_text, prompt_question)
        prompt_text = _append_after_current_question(
            prompt_text,
            self.querying.final_output_instruction(),
        )
        sections = [prompt_text]
        querying_suffix = self.querying.user_prompt_suffix()
        if querying_suffix:
            sections.append(querying_suffix)
        # image_text alone made Qwen ramble reasoning and never emit FINAL_OUTPUT.
        # Append the format reminder last so it is the final thing the model reads.
        if obs == "image_text":
            sections.append(user_templates.IMAGE_TEXT_ACTION_FORMAT_REMINDER)
        prompt_text = "\n\n".join(sections)
        summary_blocks = leading_summary_blocks(obs, ctx, transcript, self.task_spec)
        hist_blocks = history_content_blocks(obs, ctx, transcript)
        images = current_image_blocks(obs, self.last_rgb)
        prompt_blocks = _expand_current_image_placeholder(prompt_text, images)
        one_shot_blocks = self._one_shot_blocks(obs) if with_one_shot else []
        if one_shot_blocks or summary_blocks or hist_blocks or images:
            return {
                "role": "user",
                "content": one_shot_blocks + summary_blocks + hist_blocks + prompt_blocks,
            }
        return {"role": "user", "content": prompt_text}

    def _result(
        self,
        success: bool,
        state,
        transcript: List[dict],
        query_count: int,
        end_reason: str,
        initial_state: dict,
        maze_path: str | Path | None,
    ) -> dict:
        return {
            "success": success,
            "steps_used": state.step_count,
            "end_reason": end_reason,
            "query_count": query_count,
            "final_state": state_snapshot(state),
            "initial_state": initial_state,
            "transcript": transcript,
            "config": self.config.to_dict(),
            "task_spec": self.task_spec.to_dict(),
            "maze_path": str(maze_path) if maze_path is not None else None,
        }
