"""EpisodeStepper — the shared per-episode state machine.

This is a **verbatim** extraction of the episode loop that used to live inline
in ``ExperimentRunner.run`` (``interface/runner.py``). Both the serial runner
and the batch-API lockstep runner drive the SAME per-episode logic through this
seam so they cannot drift.

The loop is split at the agent call:

    stepper.start()                       # backend reset + transcript seed
    while (messages := stepper.next_query()) is not None:
        reply = ...call the model with `messages`...
        stepper.apply_reply(reply)
    result = stepper.result()

``next_query`` drains locally (primitive_buffer / action_queue / env steps /
termination checks) until either a query is needed (returns agent_messages) or
the episode is over (returns None). ``apply_reply`` parses the reply, writes the
query record, appends chat history, and does parse-failure bookkeeping.

Every counter, transcript-record field, the history normalization block, the
``logged_agent_messages`` pop, stall-K logic, parse-retry ``continue``
semantics, and cardinal ``primitive_buffer`` expansion are preserved exactly as
they were in ``ExperimentRunner.run``. The message-building helpers
(``_build_message`` / ``_one_shot_blocks`` / ``build_prompt_message``) and the
result builder (``_result``) stay on ``ExperimentRunner`` and are proxied here
via the ``runner`` reference; ``last_rgb`` is read/written on the runner so
those helpers observe the current frame — behaviour identical to before.
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import List, Optional

from interface import action_space as action_space_mod
from interface.actions_map import nlu_action_to_int
from interface.coords import agent_facing, agent_row_col
from interface.episode_log import state_snapshot
from interface.feedback import format_step_feedback
from interface.progress_watchdog import ProgressStallWatchdog
from interface.runner import _user_message_has_image, _trim_rolling_chat
from prompting_experiments.prompt_templates import feedback as feedback_templates

logger = logging.getLogger(__name__)

# Reply.stop_reason values that mean the transport/provider failed to deliver a
# response, NOT that the model produced an unparseable move. These must not
# consume the per-episode parse-retry budget (they'd terminate a valid episode
# on provider flakiness). Set by the batch/sync agents (moonshot_batch, the
# sync-transport path) and by provider overload responses. Token-cap truncation
# is intentionally NOT here — exhausting the output budget is a real model
# outcome the run enforces.
_INFRA_STOP_REASONS = frozenset(
    {"sync_error", "batch_expired", "batch_errored", "engine_overloaded"}
)


class EpisodeStepper:
    """Drives one episode; the loop body moved here verbatim from the runner."""

    def __init__(
        self,
        runner,
        *,
        verbose: bool = False,
        maze_path: str | Path | None = None,
    ) -> None:
        self._runner = runner
        self.verbose = verbose
        self.maze_path = maze_path
        # Proxy the immutable collaborators for readable, verbatim loop code.
        self.backend = runner.backend
        self.task_spec = runner.task_spec
        self.config = runner.config
        self.prompt = runner.prompt
        self.querying = runner.querying
        self._finished = False
        self._stall_watchdog: ProgressStallWatchdog | None = None

    def start(self) -> None:
        # Persist the exact seed the backend was reset with so a checkpoint can
        # replay the episode deterministically from the same starting maze.
        self.seed = self.task_spec.seed
        self._runner.last_rgb, state, reset_info = self.backend.reset(seed=self.seed)
        self.querying.reset()

        # Build the initial system prompt (may include the initial maze for
        # text-based observations) and the initial user message block.
        system_prompt, _ = self._runner.build_prompt_message(
            state, feedback_templates.INITIAL_FEEDBACK, []
        )
        self.system_message = {"role": "system", "content": system_prompt}
        self.chat_history = self.config.chat_history
        self.messages: List[dict] = (
            [self.system_message] if self.chat_history in ("rolling", "full") else []
        )

        self.action_queue: List[str] = []
        # Cardinal moves expand into several egocentric primitives; the buffer
        # holds (primitive, source_cardinal_token) pairs so each primitive runs
        # through the per-step machinery while keeping its provenance. Empty (and
        # source None) in egocentric mode.
        self.primitive_buffer: List[tuple[str, str | None]] = []
        self.actions_hint = action_space_mod.actions_hint(self.config.action_space)
        self.last_feedback = feedback_templates.INITIAL_FEEDBACK
        self.consecutive_failures = 0
        self.transcript: List[dict] = []
        self.state = state
        self.max_steps = self.task_spec.max_steps
        self.query_count = 0
        self.parse_failures = 0
        self.step_index = 0
        self.current_query_index = 0
        self.action_queue_index = 0
        self.end_reason = "max_steps"
        self.initial_state = state_snapshot(state)

        k = self.config.progress_stall_k
        if k is not None and getattr(self.task_spec.goal, "goal_type", None) == "survive_steps":
            raise ValueError(
                "progress_stall_k is incompatible with survive_steps goals: "
                "repeated states are the intended behavior."
            )
        self.stall_k = k
        self._stall_watchdog = ProgressStallWatchdog(k, state) if k else None

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "Episode start: task_id=%s seed=%s max_steps=%s querying=%s observation=%s context_window=%s chat_history=%s",
                self.task_spec.task_id,
                self.task_spec.seed,
                self.max_steps,
                self.config.querying,
                self.config.observation,
                self.config.context_window,
                self.chat_history,
            )

        self.transcript.append(
            {
                "kind": "reset",
                "state": self.initial_state,
                "backend_info": reset_info,
                "_reset_frame_rgb": self._runner.last_rgb,
            }
        )
        self._finished = False

    def next_query(self) -> Optional[List[dict]]:
        # ``apply_reply`` can end the episode (e.g. ``parse_failed`` at the retry
        # cap); the driver still loops and calls ``next_query`` once more, which
        # must report the episode is over rather than re-enter the loop.
        if self._finished:
            return None
        state = self.state
        while state.step_count < self.max_steps:
            # Never re-query while a cardinal move is still being drained into
            # primitives, or the buffered primitives would be abandoned.
            if not self.primitive_buffer and self.querying.should_query(
                self.action_queue, self.consecutive_failures
            ):
                self.consecutive_failures = 0
                self.query_count += 1
                self.current_query_index = self.query_count
                self.action_queue_index = 0
                # Multi-turn (rolling/full) history keeps the one-shot ICL example
                # on the CURRENT turn only and stores lean user turns in history, so
                # the example is not re-sent every turn (which bloated tokens ~4-5x
                # and swamped the observation, collapsing Claude to loops). For the
                # same reason the turns must not embed the context_window history
                # (last3/text_summary) — the chat itself is the history there.
                user_message = self._runner._build_message(
                    state,
                    self.last_feedback,
                    self.transcript,
                    with_one_shot=(self.chat_history == "stateless"),
                    with_context_history=(self.chat_history == "stateless"),
                )
                has_image = _user_message_has_image(user_message)
                if self.chat_history == "stateless":
                    agent_messages: List[dict] = [self.system_message, user_message]
                else:
                    self.messages.append(user_message)
                    one_shot_blocks = self._runner._one_shot_blocks(self.config.observation)
                    if one_shot_blocks:
                        cur = user_message.get("content")
                        if not isinstance(cur, list):
                            cur = [{"type": "text", "text": cur}]
                        current_turn = {"role": "user", "content": one_shot_blocks + cur}
                        agent_messages = self.messages[:-1] + [current_turn]
                    else:
                        agent_messages = self.messages
                if logger.isEnabledFor(logging.INFO):
                    logger.info(
                        "LLM query #%d: task_id=%s observation=%s messages_in_context=%d current_turn_has_image=%s",
                        self.query_count,
                        self.task_spec.task_id,
                        self.config.observation,
                        len(agent_messages),
                        has_image,
                    )
                self._pending_agent_messages = agent_messages
                self._pending_has_image = has_image
                self._t_llm = time.perf_counter()
                return agent_messages

            # Refill the primitive buffer from the next queued action. In cardinal
            # mode a move expands into turns + MOVE_FORWARD for the current facing;
            # in egocentric mode it is the token itself with no source.
            if not self.primitive_buffer:
                if not self.action_queue:
                    self.end_reason = "exhausted"
                    self._finished = True
                    return None
                token = self.action_queue.pop(0)
                if self.config.action_space == "cardinal":
                    primitives = action_space_mod.cardinal_to_primitives(
                        token, agent_facing(state)
                    )
                    self.primitive_buffer = [(p, token) for p in primitives]
                else:
                    self.primitive_buffer = [(token, None)]

            action, cardinal_action = self.primitive_buffer.pop(0)
            self.step_index += 1
            position_before = agent_row_col(state)
            facing_before = agent_facing(state)
            state_before = state_snapshot(state)
            decision_frame_rgb = self._runner.last_rgb
            actions_remaining_after = list(self.action_queue)

            prev_state = state
            try:
                action_int = nlu_action_to_int(action)
            except ValueError:
                step_detail, event_type = format_step_feedback(
                    action, prev_state, prev_state, 0.0, False, self.task_spec
                )
                self.last_feedback = step_detail
                self.consecutive_failures += 1
                self.action_queue.clear()
                self.primitive_buffer.clear()
                self.transcript.append(
                    {
                        "kind": "step",
                        "step_index": self.step_index,
                        "query_index": self.current_query_index,
                        "action_queue_index": self.action_queue_index,
                        "env_step_count": state.step_count,
                        "action": action,
                        "cardinal_action": cardinal_action,
                        "event_type": event_type,
                        "feedback": step_detail,
                        "prompt_feedback": self.last_feedback,
                        "facing_before": facing_before,
                        "facing_after": facing_before,
                        "position_before_row_col": list(position_before),
                        "position_after_row_col": list(position_before),
                        "state_before": state_before,
                        "state_after": state_snapshot(state),
                        "reward": 0.0,
                        "terminated": False,
                        "truncated": False,
                        "backend_info": None,
                        "actions_remaining_after": actions_remaining_after,
                        "consecutive_failures_after": self.consecutive_failures,
                        "_decision_frame_rgb": decision_frame_rgb,
                        "_post_step_rgb": decision_frame_rgb,
                        **self.querying.step_metadata(),
                    }
                )
                self.action_queue_index += 1
                continue

            self._runner.last_rgb, reward, terminated, truncated, state, info = self.backend.step(
                action_int
            )
            self.state = state
            step_detail, event_type = format_step_feedback(
                action, prev_state, state, reward, terminated, self.task_spec
            )
            self.last_feedback = step_detail

            if event_type in {"BLOCKED", "WRONG_DONE", "INVALID"}:
                self.consecutive_failures += 1
                self.action_queue.clear()
                self.primitive_buffer.clear()
            else:
                self.consecutive_failures = 0

            self.transcript.append(
                {
                    "kind": "step",
                    "step_index": self.step_index,
                    "query_index": self.current_query_index,
                    "action_queue_index": self.action_queue_index,
                    "env_step_count": state.step_count,
                    "action": action,
                    "cardinal_action": cardinal_action,
                    "event_type": event_type,
                    "feedback": step_detail,
                    "prompt_feedback": self.last_feedback,
                    "facing_before": facing_before,
                    "facing_after": agent_facing(state),
                    "position_before_row_col": list(position_before),
                    "position_after_row_col": list(agent_row_col(state)),
                    "state_before": state_before,
                    "state_after": state_snapshot(state),
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "backend_info": info,
                    "actions_remaining_after": actions_remaining_after,
                    "consecutive_failures_after": self.consecutive_failures,
                    "_decision_frame_rgb": decision_frame_rgb,
                    "_post_step_rgb": self._runner.last_rgb,
                    **self.querying.step_metadata(),
                }
            )
            self.action_queue_index += 1

            if self._stall_watchdog and not terminated and not truncated:
                if self._stall_watchdog.observe(state):
                    self.end_reason = "stalled"
                    self._finished = True
                    return None

            reached_goal = event_type == "DONE" or (
                terminated and getattr(state, "goal_reached", False)
            )
            if reached_goal:
                self.end_reason = "success"
                if self.verbose:
                    print(f"  Success at step {state.step_count}")
                self._finished = True
                return None

            if self.verbose:
                print(f"  Step {state.step_count}/{self.max_steps}: {action} -> {event_type}")

            if terminated:
                # Backend ended the episode without reaching the goal (e.g. a hazard).
                self.end_reason = "terminated_failure"
                self._finished = True
                return None

            if truncated:
                self.end_reason = "truncated"
                self._finished = True
                return None

        self._finished = True
        return None

    def apply_reply(self, reply) -> None:
        model_text = reply.text
        llm_s = time.perf_counter() - self._t_llm
        agent_messages = self._pending_agent_messages
        has_image = self._pending_has_image
        state = self.state
        self.action_queue = self.querying.parse_actions(model_text)
        if self.chat_history != "stateless":
            # Keep successful action history in the same canonical form
            # required for the next reply. In particular, do not let a
            # legacy parser fallback such as ``ACTION: TURN_LEFT`` teach
            # that delimiter back to the model on later rolling turns.
            if self.action_queue:
                subgoal = (
                    f"SUB_GOAL: {self.querying.current_subgoal}\n"
                    if self.querying.kind == "subgoal"
                    and self.querying.current_subgoal
                    else ""
                )
                history_reply = (
                    f"{subgoal}FINAL_OUTPUT: {', '.join(self.action_queue)}"
                )
            else:
                # The verbatim response remains in the query artifact,
                # but retaining a rejected ``ACTION: ...`` turn in the
                # model's chat context would reinforce the exact legacy
                # delimiter this normalization is intended to remove.
                history_reply = "The previous response did not contain a valid action."
            self.messages.append({"role": "assistant", "content": history_reply})
            if self.chat_history == "rolling":
                _trim_rolling_chat(self.messages, max(1, self.config.chat_turns_max))
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "LLM query #%d finished: task_id=%s observation=%s elapsed=%.2fs reply_chars=%d actions_parsed=%d",
                self.query_count,
                self.task_spec.task_id,
                self.config.observation,
                llm_s,
                len(model_text),
                len(self.action_queue),
            )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "LLM query #%d reply: task_id=%s observation=%s\n%s",
                self.query_count,
                self.task_spec.task_id,
                self.config.observation,
                model_text,
            )
        # ``messages`` is extended with the current assistant reply
        # above in multi-turn modes. Persist the actual request, not a
        # live list that now also contains the response it elicited.
        logged_agent_messages = copy.deepcopy(agent_messages)
        if self.chat_history != "stateless" and logged_agent_messages:
            if logged_agent_messages[-1].get("role") == "assistant":
                logged_agent_messages.pop()
        query_record = {
            "kind": "query",
            "query_index": self.query_count,
            "env_step_count": state.step_count,
            "agent_messages": logged_agent_messages,
            "assistant_reply": model_text,
            "parsed_actions": list(self.action_queue),
            "parse_ok": bool(self.action_queue),
            "has_image": has_image,
            "llm_latency_s": llm_s,
            "chat_history_mode": self.chat_history,
            "agent_message_count": len(logged_agent_messages),
            "actions_remaining_before_step": len(self.action_queue),
        }
        usage = reply.usage
        if isinstance(usage, dict):
            query_record["usage"] = dict(usage)
        thinking = reply.thinking
        if thinking:
            query_record["thinking"] = thinking
        # Additive-only: record model stop metadata when the Reply carries it.
        # Legacy ``__call__`` agents (shimmed into Reply) leave these at their
        # defaults, so the pre-existing record shape is unchanged for them.
        if reply.stop_reason is not None:
            query_record["stop_reason"] = reply.stop_reason
        if reply.token_truncated:
            query_record["token_truncated"] = reply.token_truncated
        self.transcript.append(query_record)
        # check if we got any valid actions;
        # if not, we'll count it as a parse failure and give feedback,
        # but still allow retries until max_parse_retries is reached
        if not self.action_queue:
            # Infrastructure failures (transport/provider unavailability) are NOT
            # the model failing to produce a valid move — the model never got to
            # answer. Don't spend its parse-retry budget on them; retry on the
            # next round instead of terminating the episode. Token-cap truncation
            # (finish_reason=length / token_truncated) is deliberately EXCLUDED
            # here: overrunning the 64k budget IS a real, enforced model outcome.
            if reply.stop_reason in _INFRA_STOP_REASONS and not reply.token_truncated:
                logger.warning(
                    "LLM query #%d: task_id=%s infrastructure failure "
                    "(stop_reason=%s); NOT counting toward parse-retry budget %d/%d",
                    self.query_count,
                    self.task_spec.task_id,
                    reply.stop_reason,
                    self.parse_failures,
                    self.config.max_parse_retries,
                )
                self.last_feedback = feedback_templates.PARSE_FAILURE_FEEDBACK.format(
                    actions_hint=self.actions_hint
                )
                return
            self.parse_failures += 1
            logger.warning(
                "LLM query #%d: task_id=%s observation=%s no valid actions parsed; parse failure %d/%d",
                self.query_count,
                self.task_spec.task_id,
                self.config.observation,
                self.parse_failures,
                self.config.max_parse_retries,
            )
            self.last_feedback = (
                feedback_templates.PARSE_FAILURE_FEEDBACK.format(
                    actions_hint=self.actions_hint
                )
            )
            if self.parse_failures >= self.config.max_parse_retries:
                self.end_reason = "parse_failed"
                self._finished = True
                return
            return
        self.parse_failures = 0

    @property
    def finished(self) -> bool:
        return self._finished

    def result(self) -> dict:
        success = self.end_reason == "success"
        return self._runner._result(
            success,
            self.state,
            self.transcript,
            self.query_count,
            self.end_reason,
            self.initial_state,
            self.maze_path,
        )
