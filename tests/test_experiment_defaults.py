"""Locks the fair default baseline that every conditional experiment ablates down from.

The baseline must be a config the models can actually perform in (image+text,
3-step single-message history, cardinal actions, one worked example), so that
changing one dependent variable measures signal rather than a floored baseline.
"""
from __future__ import annotations

from interface.config import ExperimentConfig
from prompting_experiments import iter_condition_configs


def _by_variant(name: str) -> dict:
    return {v: cfg for v, cfg in iter_condition_configs(name)}


def test_fair_default_baseline() -> None:
    c = ExperimentConfig()
    assert c.observation == "image_text"
    assert c.include_current_observation_description is True  # text half of image+text is on
    assert c.observation_text_includes_facing is True
    assert c.context_window == "last3"
    assert c.chat_history == "stateless"  # single-message history is the default mechanism
    assert c.action_space == "egocentric"  # egocentric is the standard interface; cardinal is the arm
    assert c.in_context_learning == "one_shot"
    assert c.querying == "step_by_step"
    assert c.chat_turns_max == 3


def test_ablation_arms_flip_from_fair_default() -> None:
    obs = _by_variant("Observation format")
    assert obs["image_only"].observation == "image_only"
    assert obs["image_only"].include_current_observation_description is False  # pure image, no text
    assert obs["text_only"].observation == "text_only"
    ctx = _by_variant("Context window")
    assert ctx["current"].context_window == "current" and ctx["current"].chat_history == "stateless"
    assert ctx["text_summary"].context_window == "text_summary" and ctx["text_summary"].chat_history == "stateless"
    act = _by_variant("Action space")
    assert act["cardinal"].action_space == "cardinal"  # cardinal is the arm; egocentric is the default
    icl = _by_variant("In-context learning")
    assert icl["zero_shot"].in_context_learning == "zero_shot"


def test_history_mechanism_set() -> None:
    hm = _by_variant("History mechanism")
    assert hm["single_message"].chat_history == "stateless"
    assert hm["single_message"].context_window == "last3"
    assert hm["multiturn"].chat_history == "rolling"
    assert hm["multiturn"].context_window == "current"
    assert hm["multiturn"].chat_turns_max == 3
