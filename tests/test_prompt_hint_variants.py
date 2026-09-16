"""Each hint probe appends exactly one line to the minimal system prompt."""

from interface.config import ExperimentConfig
from interface.prompt_strategies import (
    MinimalInteractionRulesPromptStrategy,
    MinimalInteractionRulesStandPromptStrategy,
    MinimalPromptStrategy,
    MinimalStateChangeHintPromptStrategy,
    MinimalSwitchHintPromptStrategy,
)
from interface.runner import _PROMPT_STRATEGIES

HINT = "TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, PICKUP, DROP, TOGGLE, DONE"
MINIMAL = "Task: Solve the maze by reaching the goal.\nValid actions: " + HINT + "."


def test_minimal_prompt_unchanged():
    assert MinimalPromptStrategy(HINT).build_system_prompt() == MINIMAL


def test_switch_hint_appends_one_line_after_actions():
    assert MinimalSwitchHintPromptStrategy(HINT).build_system_prompt() == (
        MINIMAL + "\nSwitches must be toggled."
    )


def test_state_change_hint_appends_one_line_after_actions():
    assert MinimalStateChangeHintPromptStrategy(HINT).build_system_prompt() == (
        MINIMAL
        + "\nSuccessful actions will always change the maze state, either in the rendering or inventory."
    )


def test_interaction_rules_appends_one_line_after_actions():
    assert MinimalInteractionRulesPromptStrategy(HINT).build_system_prompt() == (
        MINIMAL
        + "\nThere are two interaction actions, PICKUP and TOGGLE. Keys must be picked up. "
        "Doors and switches must be toggled. A key door will not open if you do not have a "
        "matching color key in your inventory. A switch door will open when you TOGGLE its switch."
    )


def test_interaction_rules_stand_appends_the_stand_sentence():
    assert MinimalInteractionRulesStandPromptStrategy(HINT).build_system_prompt() == (
        MINIMAL
        + "\nThere are two interaction actions, PICKUP and TOGGLE. Keys must be picked up. "
        "Doors and switches must be toggled. A key door will not open if you do not have a "
        "matching color key in your inventory. A switch door will open when you TOGGLE its switch. "
        "You must stand on a key/switch to act on it."
    )


def test_variants_are_selectable_from_config():
    for name, cls in (
        ("minimal_switch_hint", MinimalSwitchHintPromptStrategy),
        ("minimal_state_change_hint", MinimalStateChangeHintPromptStrategy),
        ("minimal_interaction_rules", MinimalInteractionRulesPromptStrategy),
        ("minimal_interaction_rules_stand", MinimalInteractionRulesStandPromptStrategy),
    ):
        assert ExperimentConfig(prompting=name).prompting == name
        assert _PROMPT_STRATEGIES[name] is cls
