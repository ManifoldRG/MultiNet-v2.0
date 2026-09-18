"""Last-step feedback lives on the current user turn, with 3 verbosity levels."""

from __future__ import annotations

import dataclasses

from interface.config import ExperimentConfig
from interface.feedback import format_step_feedback
from interface.loader import load_task
from tests.test_prompt_observation_text import (
    _initial_user_prompt_text,
    _user_prompt_text_with_transcript,
)


def test_last_feedback_is_on_the_current_turn_for_text_modes():
    text = _initial_user_prompt_text(ExperimentConfig(observation="text_only"))
    image_text = _initial_user_prompt_text(ExperimentConfig(observation="image_text"))
    image_only = _initial_user_prompt_text(ExperimentConfig(observation="image_only"))

    assert "Last feedback: Episode start." in text
    assert "Last feedback: Episode start." in image_text
    assert "Last feedback:" not in image_only


def test_last3_history_does_not_repeat_feedback():
    transcript = [{
        "kind": "step",
        "event_type": "MOVED",
        "position_after_row_col": (1, 2),
        "facing_after": "EAST",
        "action": "MOVE_FORWARD",
        "prompt_feedback": "MOVED — MOVE_FORWARD: Moved to (1, 2).",
    }]
    prompt = _user_prompt_text_with_transcript(
        ExperimentConfig(observation="text_only", context_window="last3"),
        transcript,
    )
    history, _, current = prompt.partition("You are at")
    assert "Feedback:" not in history
    assert "Last feedback: Episode start." in current or "Last feedback: Episode start." in prompt


def test_feedback_levels_on_a_closed_gate():
    backend, spec = load_task("gridworld/tasks/tier3/key_switch_001.json")
    _, state, _ = backend.reset(seed=spec.seed)
    blocked = dataclasses.replace(state, agent_position=(5, 4), agent_direction=0)

    def fb(level):
        return format_step_feedback(
            "MOVE_FORWARD", blocked, blocked, 0.0, False, spec, level=level
        )[0]

    assert fb("minimal") == "BLOCKED"
    assert "Activate switch" not in fb("standard")
    assert "Activate switch" in fb("causal")
    assert fb("standard") != fb("causal")


def test_feedback_levels_on_a_locked_door():
    backend, spec = load_task("gridworld/tasks/tier3/key_switch_001.json")
    _, state, _ = backend.reset(seed=spec.seed)
    blocked = dataclasses.replace(state, agent_position=(2, 3), agent_direction=0)
    holding = dataclasses.replace(blocked, agent_carrying="blue")

    def fb(state, level):
        return format_step_feedback(
            "MOVE_FORWARD", state, state, 0.0, False, spec, level=level
        )[0]

    assert fb(blocked, "minimal") == "BLOCKED"
    assert "blue key" not in fb(blocked, "standard")
    assert "It requires the blue key" in fb(blocked, "causal")
    assert "TOGGLE" not in fb(holding, "standard")
    assert "TOGGLE" in fb(holding, "causal")
