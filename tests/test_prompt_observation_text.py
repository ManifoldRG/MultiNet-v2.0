from __future__ import annotations

import numpy as np

from interface.config import ExperimentConfig
from interface.loader import default_maze_path, load_task
from interface.observation import current_observation_text, history_content_blocks
from interface.parser import ACTIONS_HINT
from interface.prompt_strategies import (
    MinimalPromptStrategy,
    StandardPromptStrategy,
    VerbosePromptStrategy,
)
from interface.runner import build_runner
from prompting_experiments import CONDITION_SETS
from prompting_experiments.condition_set_2_observation_format import CONDITION_SET
from prompting_experiments.prompt_templates import feedback as feedback_templates


def _initial_spec_and_state():
    backend, spec = load_task(default_maze_path())
    _rgb, state, _info = backend.reset(seed=spec.seed)
    return spec, state


def _initial_user_prompt_text(cfg: ExperimentConfig) -> str:
    backend, spec = load_task(default_maze_path())
    runner = build_runner(cfg, backend, spec)
    runner.last_rgb, state, _info = backend.reset(seed=spec.seed)
    message = runner._build_message(state, feedback_templates.INITIAL_FEEDBACK, [])
    content = message["content"]
    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return content


def _user_prompt_text_with_transcript(
    cfg: ExperimentConfig, transcript: list[dict]
) -> str:
    backend, spec = load_task(default_maze_path())
    runner = build_runner(cfg, backend, spec)
    runner.last_rgb, state, _info = backend.reset(seed=spec.seed)
    message = runner._build_message(state, feedback_templates.INITIAL_FEEDBACK, transcript)
    content = message["content"]
    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return content


def _initial_user_prompt_text_for_maze(cfg: ExperimentConfig, maze_name: str) -> str:
    backend, spec = load_task(default_maze_path(maze_name))
    runner = build_runner(cfg, backend, spec)
    runner.last_rgb, state, _info = backend.reset(seed=spec.seed)
    message = runner._build_message(state, feedback_templates.INITIAL_FEEDBACK, [])
    content = message["content"]
    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return content


def test_current_observation_omits_description_by_default():
    spec, state = _initial_spec_and_state()

    text = current_observation_text("image_text", spec, state)

    assert text == ""


def test_current_observation_can_render_without_facing():
    spec, state = _initial_spec_and_state()

    text = current_observation_text(
        "image_text",
        spec,
        state,
        include_description=True,
    )

    assert "Current situation (this step):" in text
    assert "The goal is at" not in text
    assert "You are at (1, 1)." in text
    assert "You are at (1, 1) facing EAST." not in text


def test_observation_format_text_variants_keep_facing():
    spec, state = _initial_spec_and_state()
    base = ExperimentConfig(observation_text_includes_facing=False)

    for variant_name in ("text_only", "image_text"):
        cfg = CONDITION_SET.variants[variant_name].build_config(base)
        text = current_observation_text(
            cfg.observation,
            spec,
            state,
            include_description=cfg.include_current_observation_description,
            include_facing=cfg.observation_text_includes_facing,
        )

        assert "Current situation (this step):" in text
        assert "The goal is at" not in text
        assert "You are at (1, 1) facing EAST." in text


def test_observation_format_image_only_has_no_current_observation_text():
    spec, state = _initial_spec_and_state()
    cfg = CONDITION_SET.variants["standard"].build_config(ExperimentConfig())

    text = current_observation_text(
        cfg.observation,
        spec,
        state,
        include_description=cfg.include_current_observation_description,
        include_facing=cfg.observation_text_includes_facing,
    )

    assert text == ""
    assert "Current situation (this step):" not in text
    assert "You are at" not in text


def test_image_only_prompt_puts_inventory_text_after_current_image():
    backend, spec = load_task(default_maze_path())
    runner = build_runner(ExperimentConfig(observation="image_only"), backend, spec)
    runner.last_rgb, state, _info = backend.reset(seed=spec.seed)

    message = runner._build_message(state, feedback_templates.INITIAL_FEEDBACK, [])
    content = message["content"]

    assert isinstance(content, list)
    assert content[0]["type"] == "image_url"
    assert content[1]["type"] == "text"
    assert "Current situation (this step):" not in content[1]["text"]
    assert content[1]["text"].startswith("Your inventory: empty.\nWhat is your next action?")


def test_image_only_last3_history_puts_inventory_before_action_under_images():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    transcript = [
        {
            "kind": "step",
            "event_type": "VALID",
            "action": "MOVE_FORWARD",
            "state_before": {"inventory": []},
            "_decision_frame_rgb": frame,
        },
        {
            "kind": "step",
            "event_type": "VALID",
            "action": "PICKUP",
            "state_before": {"inventory": ["red"]},
            "_decision_frame_rgb": frame,
        },
    ]

    blocks = history_content_blocks("image_only", "last3", transcript)

    assert blocks[0]["type"] == "text"
    assert blocks[1]["type"] == "image_url"
    assert blocks[2] == {
        "type": "text",
        "text": "Your inventory: empty.\nAction: MOVE_FORWARD\n\n",
    }
    assert blocks[3]["type"] == "image_url"
    assert blocks[4] == {
        "type": "text",
        "text": "Your inventory: red.\nAction: PICKUP\n\n",
    }


def test_non_observation_format_conditions_omit_current_description_from_prompt():
    for condition_name, condition in CONDITION_SETS.items():
        if condition is CONDITION_SET:
            continue

        for variant in condition.variants.values():
            if not variant.implemented:
                continue

            backend, spec = load_task(default_maze_path())
            cfg = variant.build_config(ExperimentConfig())
            runner = build_runner(cfg, backend, spec)
            runner.last_rgb, state, _info = backend.reset(seed=spec.seed)

            message = runner._build_message(state, feedback_templates.INITIAL_FEEDBACK, [])
            content = message["content"]
            if isinstance(content, list):
                prompt_text = "\n".join(
                    block["text"]
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            else:
                prompt_text = content

            assert "Observation:\nCurrent situation (this step):" not in prompt_text, (
                condition_name,
                variant.name,
            )


def test_non_observation_format_conditions_omit_initial_maze_from_prompt():
    for condition_name, condition in CONDITION_SETS.items():
        if condition is CONDITION_SET:
            continue

        for variant in condition.variants.values():
            if not variant.implemented:
                continue

            prompt_text = _initial_user_prompt_text(variant.build_config(ExperimentConfig()))

            assert "Initial maze (fixed for this episode):" not in prompt_text, (
                condition_name,
                variant.name,
            )


def test_observation_format_initial_maze_only_for_text_variants():
    text_variants = {"text_only", "image_text"}
    for variant_name, variant in CONDITION_SET.variants.items():
        cfg = variant.build_config(ExperimentConfig())
        prompt_text = _initial_user_prompt_text(cfg)
        has_initial_maze = "Initial maze (fixed for this episode):" in prompt_text

        assert has_initial_maze is (variant_name in text_variants), variant_name


def test_initial_prompts_omit_current_status_footer_without_history_context():
    for variant in CONDITION_SET.variants.values():
        cfg = variant.build_config(ExperimentConfig())
        prompt_text = _initial_user_prompt_text(cfg)

        assert "Position: (1, 1)  |  Facing: EAST  |  Goal: (6, 6)" not in prompt_text
        assert "Last result: Episode start." not in prompt_text


def test_text_last3_prompt_omits_unused_recent_history_text():
    transcript = [
        {
            "kind": "step",
            "event_type": "VALID",
            "position_after": (1, 2),
            "facing_after": "EAST",
            "action": "MOVE_FORWARD",
            "prompt_feedback": "MOVED",
        }
    ]
    cfg = ExperimentConfig(observation="text_only", context_window="last3")

    prompt_text = _user_prompt_text_with_transcript(
        cfg,
        transcript,
    )

    assert "Recent history (last 3 steps, oldest first):" in prompt_text
    assert "Position: (1, 1)  |  Facing: EAST  |  Goal: (6, 6)" not in prompt_text
    assert "Last result: Episode start." not in prompt_text


def test_observation_format_image_only_matches_standard_prompt_text():
    standard_text = _initial_user_prompt_text(ExperimentConfig())
    image_only_text = _initial_user_prompt_text(
        CONDITION_SET.variants["standard"].build_config(ExperimentConfig())
    )

    assert image_only_text == standard_text


def test_minimal_prompt_uses_minimal_system_and_inventory_only_user_status():
    system_prompt = MinimalPromptStrategy(ACTIONS_HINT).build_system_prompt()
    prompt_text = _initial_user_prompt_text(ExperimentConfig(prompting="minimal"))

    assert system_prompt.startswith("Task: Solve the maze by reaching the goal.")
    assert "The environment may contain:" not in system_prompt
    assert prompt_text.startswith("Your inventory: empty.\nWhat is your next action?")
    assert "Observation:" not in prompt_text
    assert "Position:" not in prompt_text
    assert "Facing:" not in prompt_text
    assert "Goal:" not in prompt_text
    assert "Last result:" not in prompt_text


def test_prompting_variants_share_image_only_user_prompt():
    standard_text = _initial_user_prompt_text(ExperimentConfig(prompting="standard"))
    minimal_text = _initial_user_prompt_text(ExperimentConfig(prompting="minimal"))
    verbose_text = _initial_user_prompt_text(ExperimentConfig(prompting="verbose"))

    assert standard_text == minimal_text == verbose_text
    assert standard_text.startswith("Your inventory: empty.\nWhat is your next action?")
    assert "Position:" not in standard_text
    assert "Last result:" not in standard_text
    assert "Hints:" not in standard_text


def test_standard_variants_use_default_config_without_overrides():
    for condition in CONDITION_SETS.values():
        variant = condition.variants.get("standard")
        if variant is None or not variant.implemented:
            continue

        cfg = variant.build_config()

        assert cfg == ExperimentConfig()
        assert variant.config_overrides is None


def test_implemented_non_verbose_conditions_share_standard_system_prompt():
    standard_prompt = StandardPromptStrategy(ACTIONS_HINT).build_system_prompt()
    verbose_prompt = None
    for condition_name, condition in CONDITION_SETS.items():
        for variant in condition.variants.values():
            if not variant.implemented:
                continue

            backend, spec = load_task(default_maze_path())
            cfg = variant.build_config(ExperimentConfig())
            runner = build_runner(cfg, backend, spec)
            system_prompt = runner.prompt.build_system_prompt()

            if variant.name == "verbose":
                verbose_prompt = system_prompt
            elif variant.name == "minimal":
                assert system_prompt == MinimalPromptStrategy(ACTIONS_HINT).build_system_prompt()
            else:
                assert system_prompt == standard_prompt, (condition_name, variant.name)

    assert verbose_prompt is not None
    assert verbose_prompt != standard_prompt


def test_verbose_prompt_omits_mechanism_hints_by_default():
    prompt_text = _initial_user_prompt_text_for_maze(
        ExperimentConfig(prompting="verbose"),
        "V04_single_key.json",
    )

    assert "Hints:" not in prompt_text
    assert "Face an adjacent key and PICKUP" not in prompt_text
    assert "Inventory:" not in prompt_text
    assert "From your perspective:" not in prompt_text
