"""ExperimentRunner — the single episode loop for all experiment configurations.

Usage
-----
    from nlu_benchmark.config import ExperimentConfig
    from nlu_benchmark.runner import build_runner

    cfg    = ExperimentConfig(prompting="verbose", querying="full_trajectory")
    runner = build_runner(cfg, env, maze_json_path="path/to/maze.json")
    result = runner.run(agent)                      # verbose=True: print progress
    result = runner.run(agent, verbose=False)     # quiet for batch runs

Or enable library logging in your script::

    import logging
    logging.basicConfig(level=logging.INFO)
    result = runner.run(agent, verbose=False)

Or from a JSON file directly:

    runner = ExperimentRunner.from_json("path/to/maze.json", config=cfg)
    result = runner.run(agent)
"""

from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


def _user_message_has_image(message: dict) -> bool:
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(b, dict) and b.get("type") == "image_url" for b in content)

from nlu_benchmark.config import ExperimentConfig
from nlu_benchmark.feedback import action_feedback_for_prompt, format_step_feedback
from nlu_benchmark.observation import ObservationBuilder
from nlu_benchmark.prompt_strategies import (
    PromptStrategy,
    MinimalPromptStrategy,
    StandardPromptStrategy,
    VerbosePromptStrategy,
)
from nlu_benchmark.parser import ACTIONS_HINT
from nlu_benchmark.querying import QueryingMode
from nlu_benchmark.renderer import render_initial_maze_text


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_runner(
    config: ExperimentConfig,
    env,
    maze_json_path: Optional[str] = None,
) -> ExperimentRunner:
    """Assemble an ExperimentRunner from a config.

    This is the one place that maps config values to concrete implementations.
    """
    obs = ObservationBuilder(config.observation, config.context_window)

    prompt: PromptStrategy = {
        "minimal":   MinimalPromptStrategy,
        "standard":  StandardPromptStrategy,
        "verbose":   VerbosePromptStrategy,
    }[config.prompting](ACTIONS_HINT)

    querying = QueryingMode(config.querying)

    return ExperimentRunner(
        env=env,
        config=config,
        obs_builder=obs,
        prompt_strategy=prompt,
        querying_mode=querying,
        maze_json_path=maze_json_path,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Runs a maze episode.  Owns the full episode loop."""

    def __init__(
        self,
        env,
        config: ExperimentConfig,
        obs_builder: ObservationBuilder,
        prompt_strategy: PromptStrategy,
        querying_mode,
        maze_json_path: Optional[str] = None,
    ) -> None:
        self.env            = env
        self.config         = config
        self.obs            = obs_builder
        self.prompt         = prompt_strategy
        self.querying       = querying_mode
        self.maze_json_path = maze_json_path

    @classmethod
    def from_json(
        cls,
        path: str,
        config: Optional[ExperimentConfig] = None,
    ) -> ExperimentRunner:
        from nlu_benchmark.loader import load_maze
        return build_runner(config or ExperimentConfig(), load_maze(path), path)

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------

    def run(self, agent: Callable[[List[dict]], str], *, verbose: bool = True) -> dict:
        """Run one full episode.

        Parameters
        ----------
        verbose:
            If True, print per-step progress to stdout.  Use False for batch evaluation.

        Returns
        -------
        dict:
            success       – bool
            steps_used    – int
            final_state   – GridState
            transcript    – list[dict] with one record per executed action
            config        – dict, serialised ExperimentConfig for this run
        """
        state = self.env.reset()
        self.obs.reset()
        self.querying.reset()

        system_prompt = self.prompt.build_system_prompt(self.querying.system_prompt_suffix())
        if self.config.observation in ("text_only", "image_text"):
            system_prompt = (
                f"{system_prompt}\n\nInitial maze (fixed for this episode):\n"
                f"{render_initial_maze_text(state)}"
            )
        messages: List[dict] = [{"role": "system", "content": system_prompt}]

        action_queue: List[str] = []
        last_feedback           = "Episode start."
        consecutive_failures    = 0
        transcript: List[dict]  = []
        max_steps               = self.env.initial.max_steps
        query_count             = 0

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "Episode start: max_steps=%s querying=%s observation=%s context_window=%s",
                max_steps,
                self.config.querying,
                self.config.observation,
                self.config.context_window,
            )

        while state.step_count < max_steps:

            # --- Query model if needed ---
            if self.querying.should_query(action_queue, consecutive_failures):
                consecutive_failures = 0
                query_count += 1
                user_message = self._build_message(state, last_feedback)
                has_image = _user_message_has_image(user_message)
                messages.append(user_message)
                if logger.isEnabledFor(logging.INFO):
                    logger.info(
                        "LLM query #%d: messages_in_context=%d current_turn_has_image=%s",
                        query_count,
                        len(messages),
                        has_image,
                    )
                t_llm = time.perf_counter()
                model_text = agent(messages)
                llm_s = time.perf_counter() - t_llm
                messages.append({"role": "assistant", "content": model_text})
                action_queue = self.querying.parse_actions(model_text)
                if logger.isEnabledFor(logging.INFO):
                    logger.info(
                        "LLM query #%d finished in %.2fs: reply_chars=%d actions_parsed=%d",
                        query_count,
                        llm_s,
                        len(model_text),
                        len(action_queue),
                    )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("LLM query #%d reply preview: %s", query_count, model_text[:4000])

                if not action_queue:
                    logger.warning(
                        "LLM query #%d: no valid actions parsed; retrying with parser feedback",
                        query_count,
                    )
                    last_feedback = (
                        f"Could not parse FINAL_OUTPUT (one or more valid actions). "
                        f"Use only: {ACTIONS_HINT}."
                    )
                    continue

            if not action_queue:
                # e.g. full trajectory finished executing (no re-query)
                break

            # --- Execute next queued action ---
            action = action_queue.pop(0)
            position_before = state.agent_pos

            state, event = self.env.step(action)
            step_detail = format_step_feedback(action, event.type, event.message, position_before)
            last_feedback = action_feedback_for_prompt(self.config.observation, step_detail)
            event_type = event.type

            if event_type in {"BLOCKED", "WRONG_DONE", "INVALID"}:
                consecutive_failures += 1
                action_queue.clear()    # abandon the rest of the planned sequence
            else:
                consecutive_failures = 0

            transcript.append({
                "step":            state.step_count,
                "position_before": position_before,
                "position_after":  state.agent_pos,
                "action":          action,
                "event_type":      event_type,
                "feedback":        step_detail,
                **self.querying.step_metadata(),
            })

            self.obs.record(state.agent_pos, state.facing, action, last_feedback)

            if event_type == "DONE":
                if logger.isEnabledFor(logging.INFO):
                    logger.info("Episode success at env step %s (LLM queries=%d)", state.step_count, query_count)
                if verbose:
                    print(f"  Success at step {state.step_count}")
                return self._result(True, state, transcript)

            if verbose:
                print(f"  Step {state.step_count}/{max_steps}: {action} -> {event_type}")

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "Episode ended without DONE: env_steps=%s success=false LLM_queries=%d",
                state.step_count,
                query_count,
            )
        return self._result(False, state, transcript)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_message(self, state, last_feedback: str) -> dict:
        obs_text     = self.obs.build_text(state)
        history_text = self.obs.history_text()
        prompt_text  = self.prompt.build_user_prompt(obs_text, history_text, state, last_feedback)
        images = self.obs.build_image_blocks(state, self.maze_json_path)
        if images:
            return {"role": "user", "content": images + [{"type": "text", "text": prompt_text}]}
        return {"role": "user", "content": prompt_text}

    def _result(self, success: bool, state, transcript: List[dict]) -> dict:
        return {
            "success":     success,
            "steps_used":  state.step_count,
            "final_state": state,
            "transcript":  transcript,
            "config":      self.config.to_dict(),
        }
