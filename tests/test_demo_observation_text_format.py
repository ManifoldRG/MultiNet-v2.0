"""The demo exposes observation_text_format as a live settings axis."""

from __future__ import annotations

import dataclasses
import json

from demo.r1_config import R1_CONFIG
from demo.session import SETTINGS_AXES, MiniGridPlaySession

MAZE = "gridworld/tasks/tier3/key_switch_001.json"


def _session(**overrides):
    return MiniGridPlaySession(
        task_path=MAZE,
        config=dataclasses.replace(R1_CONFIG, observation="text_only", **overrides),
    )


def _section(session, title):
    return next(text for name, text in session._build_model_view_sections() if name == title)


def test_observation_text_format_settings_and_model_view():
    assert ("6", "observation_text_format", ("coords", "json", "ascii")) in SETTINGS_AXES
    assert ("7", "feedback", ("minimal", "standard", "causal")) in SETTINGS_AXES
    session = _session()
    seen = [session.config.observation_text_format]
    for _ in range(3):
        session._cycle_setting("6")
        seen.append(session.config.observation_text_format)
    assert seen == ["coords", "json", "ascii", "coords"]

    session = _session(observation_text_format="json")
    assert json.loads(_section(session, "Current observation"))["agent"] == {
        "row": 1, "col": 1, "facing": "EAST",
    }

    session = _session(observation_text_format="coords")
    session._cycle_setting("6")
    session._cycle_setting("6")
    current = _section(session, "Current observation")
    assert "Legend:" in _section(session, "Initial maze (system prompt)")
    assert "Legend:" in current
    assert current.split("\n")[2].split() == [">", ".", "#", ".", ".", "#", ".", "."]
