"""The interaction-rules variant appends exactly one line to the minimal system prompt."""

from interface.config import ExperimentConfig
from interface.prompt_strategies import (
    MinimalInteractionRulesStandPromptStrategy,
    MinimalPromptStrategy,
)
from interface.runner import _PROMPT_STRATEGIES

HINT = "TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, PICKUP, DROP, TOGGLE, DONE"
MINIMAL = "Task: Solve the maze by reaching the goal.\nValid actions: " + HINT + "."


def test_minimal_prompt_unchanged():
    assert MinimalPromptStrategy(HINT).build_system_prompt() == MINIMAL


def test_interaction_rules_stand_appends_one_line_after_actions():
    assert MinimalInteractionRulesStandPromptStrategy(HINT).build_system_prompt() == (
        MINIMAL
        + "\nThere are two interaction actions, PICKUP and TOGGLE. Keys must be picked up. "
        "Doors and switches must be toggled. A key door will not open if you do not have a "
        "matching color key in your inventory. A switch door will open when you TOGGLE its switch. "
        "You must stand on a key/switch to act on it."
    )


def test_variant_is_selectable_from_config():
    name = "minimal_interaction_rules_stand"
    assert ExperimentConfig(prompting=name).prompting == name
    assert _PROMPT_STRATEGIES[name] is MinimalInteractionRulesStandPromptStrategy
