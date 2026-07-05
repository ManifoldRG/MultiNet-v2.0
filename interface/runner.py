"""ExperimentRunner — LLM episode loop using gridworld MiniGridBackend."""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Callable, List

import numpy as np

from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.task_spec import TaskSpecification

from interface import action_space as action_space_mod
from interface.actions_map import nlu_action_to_int
from interface.config import ExperimentConfig
from interface.coords import agent_facing, agent_row_col
from interface.episode_log import state_snapshot
from interface.feedback import format_step_feedback
from interface.observation import (
    current_image_blocks,
    current_observation_text,
    history_content_blocks,
    history_text,
    recent_history_steps,
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
from prompting_experiments.prompt_templates import feedback as feedback_templates
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
        self.last_rgb, state, reset_info = self.backend.reset(seed=self.task_spec.seed)
        self.querying.reset()

        # Build the initial system prompt (may include the initial maze for
        # text-based observations) and the initial user message block.
        system_prompt, _ = self.build_prompt_message(state, feedback_templates.INITIAL_FEEDBACK, [])
        system_message = {"role": "system", "content": system_prompt}
        chat_history = self.config.chat_history
        messages: List[dict] = [system_message] if chat_history in ("rolling", "full") else []

        action_queue: List[str] = []
        # Cardinal moves expand into several egocentric primitives; the buffer
        # holds (primitive, source_cardinal_token) pairs so each primitive runs
        # through the per-step machinery while keeping its provenance. Empty (and
        # source None) in egocentric mode.
        primitive_buffer: List[tuple[str, str | None]] = []
        actions_hint = action_space_mod.actions_hint(self.config.action_space)
        last_feedback = feedback_templates.INITIAL_FEEDBACK
        consecutive_failures = 0
        transcript: List[dict] = []
        max_steps = self.task_spec.max_steps
        query_count = 0
        parse_failures = 0
        step_index = 0
        current_query_index = 0
        action_queue_index = 0
        end_reason = "max_steps"
        initial_state = state_snapshot(state)

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "Episode start: task_id=%s seed=%s max_steps=%s querying=%s observation=%s context_window=%s chat_history=%s",
                self.task_spec.task_id,
                self.task_spec.seed,
                max_steps,
                self.config.querying,
                self.config.observation,
                self.config.context_window,
                chat_history,
            )

        transcript.append(
            {
                "kind": "reset",
                "state": initial_state,
                "backend_info": reset_info,
                "_reset_frame_rgb": self.last_rgb,
            }
        )

        while state.step_count < max_steps:
            # Never re-query while a cardinal move is still being drained into
            # primitives, or the buffered primitives would be abandoned.
            if not primitive_buffer and self.querying.should_query(
                action_queue, consecutive_failures
            ):
                consecutive_failures = 0
                query_count += 1
                current_query_index = query_count
                action_queue_index = 0
                # Multi-turn (rolling/full) history keeps the one-shot ICL example
                # on the CURRENT turn only and stores lean user turns in history, so
                # the example is not re-sent every turn (which bloated tokens ~4-5x
                # and swamped the observation, collapsing Claude to loops).
                user_message = self._build_message(
                    state,
                    last_feedback,
                    transcript,
                    with_one_shot=(chat_history == "stateless"),
                )
                has_image = _user_message_has_image(user_message)
                if chat_history == "stateless":
                    agent_messages: List[dict] = [system_message, user_message]
                else:
                    messages.append(user_message)
                    one_shot_blocks = self._one_shot_blocks(self.config.observation)
                    if one_shot_blocks:
                        cur = user_message.get("content")
                        if not isinstance(cur, list):
                            cur = [{"type": "text", "text": cur}]
                        current_turn = {"role": "user", "content": one_shot_blocks + cur}
                        agent_messages = messages[:-1] + [current_turn]
                    else:
                        agent_messages = messages
                if logger.isEnabledFor(logging.INFO):
                    logger.info(
                        "LLM query #%d: task_id=%s observation=%s messages_in_context=%d current_turn_has_image=%s",
                        query_count,
                        self.task_spec.task_id,
                        self.config.observation,
                        len(agent_messages),
                        has_image,
                    )
                _reset_agent_usage(agent)
                t_llm = time.perf_counter()
                model_text = agent(agent_messages)
                llm_s = time.perf_counter() - t_llm
                if chat_history != "stateless":
                    messages.append({"role": "assistant", "content": model_text})
                    if chat_history == "rolling":
                        _trim_rolling_chat(messages, max(1, self.config.chat_turns_max))
                action_queue = self.querying.parse_actions(model_text)
                if logger.isEnabledFor(logging.INFO):
                    logger.info(
                        "LLM query #%d finished: task_id=%s observation=%s elapsed=%.2fs reply_chars=%d actions_parsed=%d",
                        query_count,
                        self.task_spec.task_id,
                        self.config.observation,
                        llm_s,
                        len(model_text),
                        len(action_queue),
                    )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "LLM query #%d reply: task_id=%s observation=%s\n%s",
                        query_count,
                        self.task_spec.task_id,
                        self.config.observation,
                        model_text,
                    )
                query_record = {
                    "kind": "query",
                    "query_index": query_count,
                    "env_step_count": state.step_count,
                    "agent_messages": copy.deepcopy(agent_messages),
                    "assistant_reply": model_text,
                    "parsed_actions": list(action_queue),
                    "parse_ok": bool(action_queue),
                    "has_image": has_image,
                    "llm_latency_s": llm_s,
                    "chat_history_mode": chat_history,
                    "agent_message_count": len(agent_messages),
                    "actions_remaining_before_step": len(action_queue),
                }
                usage = getattr(agent, "last_usage", None)
                if isinstance(usage, dict):
                    query_record["usage"] = dict(usage)
                thinking = getattr(agent, "last_thinking", None)
                if thinking:
                    query_record["thinking"] = thinking
                transcript.append(query_record)
                # check if we got any valid actions; 
                # if not, we'll count it as a parse failure and give feedback, 
                # but still allow retries until max_parse_retries is reached
                if not action_queue:
                    parse_failures += 1
                    logger.warning(
                        "LLM query #%d: task_id=%s observation=%s no valid actions parsed; parse failure %d/%d",
                        query_count,
                        self.task_spec.task_id,
                        self.config.observation,
                        parse_failures,
                        self.config.max_parse_retries,
                    )
                    last_feedback = (
                        feedback_templates.PARSE_FAILURE_FEEDBACK.format(
                            actions_hint=actions_hint
                        )
                    )
                    if parse_failures >= self.config.max_parse_retries:
                        end_reason = "parse_failed"
                        break
                    continue
                parse_failures = 0

            # Refill the primitive buffer from the next queued action. In cardinal
            # mode a move expands into turns + MOVE_FORWARD for the current facing;
            # in egocentric mode it is the token itself with no source.
            if not primitive_buffer:
                if not action_queue:
                    end_reason = "exhausted"
                    break
                token = action_queue.pop(0)
                if self.config.action_space == "cardinal":
                    primitives = action_space_mod.cardinal_to_primitives(
                        token, agent_facing(state)
                    )
                    primitive_buffer = [(p, token) for p in primitives]
                else:
                    primitive_buffer = [(token, None)]

            action, cardinal_action = primitive_buffer.pop(0)
            step_index += 1
            position_before = agent_row_col(state)
            facing_before = agent_facing(state)
            state_before = state_snapshot(state)
            decision_frame_rgb = self.last_rgb
            actions_remaining_after = list(action_queue)

            prev_state = state
            try:
                action_int = nlu_action_to_int(action)
            except ValueError:
                step_detail, event_type = format_step_feedback(
                    action, prev_state, prev_state, 0.0, False, self.task_spec
                )
                last_feedback = step_detail
                consecutive_failures += 1
                action_queue.clear()
                primitive_buffer.clear()
                transcript.append(
                    {
                        "kind": "step",
                        "step_index": step_index,
                        "query_index": current_query_index,
                        "action_queue_index": action_queue_index,
                        "env_step_count": state.step_count,
                        "action": action,
                        "cardinal_action": cardinal_action,
                        "event_type": event_type,
                        "feedback": step_detail,
                        "prompt_feedback": last_feedback,
                        "facing_before": facing_before,
                        "facing_after": facing_before,
                        "position_before": list(position_before),
                        "position_after": list(position_before),
                        "state_before": state_before,
                        "state_after": state_snapshot(state),
                        "reward": 0.0,
                        "terminated": False,
                        "truncated": False,
                        "backend_info": None,
                        "actions_remaining_after": actions_remaining_after,
                        "consecutive_failures_after": consecutive_failures,
                        "_decision_frame_rgb": decision_frame_rgb,
                        "_post_step_rgb": decision_frame_rgb,
                        **self.querying.step_metadata(),
                    }
                )
                action_queue_index += 1
                continue

            self.last_rgb, reward, terminated, truncated, state, info = self.backend.step(
                action_int
            )
            step_detail, event_type = format_step_feedback(
                action, prev_state, state, reward, terminated, self.task_spec
            )
            last_feedback = step_detail

            if event_type in {"BLOCKED", "WRONG_DONE", "INVALID"}:
                consecutive_failures += 1
                action_queue.clear()
                primitive_buffer.clear()
            else:
                consecutive_failures = 0

            transcript.append(
                {
                    "kind": "step",
                    "step_index": step_index,
                    "query_index": current_query_index,
                    "action_queue_index": action_queue_index,
                    "env_step_count": state.step_count,
                    "action": action,
                    "cardinal_action": cardinal_action,
                    "event_type": event_type,
                    "feedback": step_detail,
                    "prompt_feedback": last_feedback,
                    "facing_before": facing_before,
                    "facing_after": agent_facing(state),
                    "position_before": list(position_before),
                    "position_after": list(agent_row_col(state)),
                    "state_before": state_before,
                    "state_after": state_snapshot(state),
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "backend_info": info,
                    "actions_remaining_after": actions_remaining_after,
                    "consecutive_failures_after": consecutive_failures,
                    "_decision_frame_rgb": decision_frame_rgb,
                    "_post_step_rgb": self.last_rgb,
                    **self.querying.step_metadata(),
                }
            )
            action_queue_index += 1

            if event_type == "DONE":
                end_reason = "success"
                if verbose:
                    print(f"  Success at step {state.step_count}")
                return self._result(
                    True, state, transcript, query_count, end_reason, initial_state, maze_path
                )

            if verbose:
                print(f"  Step {state.step_count}/{max_steps}: {action} -> {event_type}")

            if truncated:
                end_reason = "truncated"
                break

        success = end_reason == "success"
        return self._result(
            success, state, transcript, query_count, end_reason, initial_state, maze_path
        )

    def _one_shot_blocks(self, obs) -> list[dict]:
        """The one-shot ICL example blocks (example image + solution), or []."""
        if self.config.in_context_learning == "one_shot":
            from interface.one_shot import one_shot_content_blocks
            return one_shot_content_blocks(obs)
        return []

    def _build_message(
        self, state, last_feedback: str, transcript: List[dict], with_one_shot: bool = True
    ) -> dict:
        obs = self.config.observation
        ctx = self.config.context_window
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
        hist_blocks = history_content_blocks(obs, ctx, transcript)
        images = current_image_blocks(obs, self.last_rgb)
        prompt_blocks = _expand_current_image_placeholder(prompt_text, images)
        one_shot_blocks = self._one_shot_blocks(obs) if with_one_shot else []
        if one_shot_blocks or hist_blocks or images:
            return {"role": "user", "content": one_shot_blocks + hist_blocks + prompt_blocks}
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
