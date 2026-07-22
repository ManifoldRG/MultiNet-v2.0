from __future__ import annotations

import numpy as np

from gridworld.backends.base import GridState
from interface.config import ExperimentConfig
from interface.coords import inventory_list
from interface.loader import default_maze_path, load_task
from interface.observation import current_observation_text, history_content_blocks
from interface.parser import ACTIONS_HINT
from interface.prompt_strategies import (
    MinimalPromptStrategy,
    StandardPromptStrategy,
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

    # No fixed "Current situation" header required; ensure position renders
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

        # No fixed header required; ensure facing is preserved when requested
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
    # image_only ablation arm: pure image, no description, no ICL example.
    cfg = ExperimentConfig(observation="image_only",
                           include_current_observation_description=False,
                           observation_text_includes_facing=False,
                           in_context_learning="zero_shot")
    runner = build_runner(cfg, backend, spec)
    runner.last_rgb, state, _info = backend.reset(seed=spec.seed)

    message = runner._build_message(state, feedback_templates.INITIAL_FEEDBACK, [])
    content = message["content"]

    assert isinstance(content, list)
    assert content[0]["type"] == "image_url"
    assert content[1]["type"] == "text"
    assert "Current situation (this step):" not in content[1]["text"]
    assert content[1]["text"].startswith("Your inventory: empty.\nWhat is your next action?")


def test_inventory_list_omits_consumed_or_removed_keys():
    state = GridState(
        agent_position=(1, 1),
        agent_direction=0,
        agent_carrying=None,
        collected_keys={"kR"},
    )

    assert inventory_list(state) == []


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
        "text": "Your inventory: empty.\nFINAL_OUTPUT: MOVE_FORWARD\n",
    }
    assert blocks[3]["type"] == "image_url"
    assert blocks[4] == {
        "type": "text",
        "text": "Your inventory: red.\nFINAL_OUTPUT: PICKUP\n",
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
        # initial maze is provided at system level; user prompt should not contain it
        has_initial_maze = "Initial maze (fixed for this episode):" in prompt_text

        assert has_initial_maze is False, variant_name


def test_image_text_prompt_carries_action_format_reminder():
    """image_text is the only observation mode where Qwen rambled spatial
    reasoning and never emitted FINAL_OUTPUT (parse failures). It gets an extra
    'don't narrate, end with FINAL_OUTPUT' reminder; image_only and text_only —
    which had zero parse failures — must NOT get it."""
    from prompting_experiments.prompt_templates import user as user_templates

    reminder = user_templates.IMAGE_TEXT_ACTION_FORMAT_REMINDER
    assert "without narrating" in reminder
    assert "FINAL_OUTPUT" in reminder

    image_text = _initial_user_prompt_text(ExperimentConfig(observation="image_text"))
    assert reminder in image_text
    # The reminder is the final thing the model reads (highest attention).
    assert image_text.rstrip().endswith(reminder)

    for obs in ("image_only", "text_only"):
        assert reminder not in _initial_user_prompt_text(ExperimentConfig(observation=obs)), obs


def test_initial_prompts_omit_current_status_footer_without_history_context():
    for variant in CONDITION_SET.variants.values():
        cfg = variant.build_config(ExperimentConfig())
        prompt_text = _initial_user_prompt_text(cfg)

        assert "Position: (1, 1)  |  Facing: EAST  |  Goal: (6, 6)" not in prompt_text
        assert "Last result: Episode start." not in prompt_text


def test_text_last3_prompt_includes_recent_history_text():
    transcript = [
        {
            "kind": "step",
            "event_type": "VALID",
            "position_after_row_col": (1, 2),
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
    assert "Position after: (1, 2), facing EAST" in prompt_text
    assert "FINAL_OUTPUT: MOVE_FORWARD" in prompt_text
    assert "Feedback: MOVED" in prompt_text
    assert "What is your next action?" in prompt_text
    assert "Position: (1, 1)  |  Facing: EAST  |  Goal: (6, 6)" not in prompt_text
    assert "Last result: Episode start." not in prompt_text


def test_cardinal_last3_history_shows_cardinal_action_not_primitive():
    # In cardinal runs FINAL_OUTPUT must be MOVE_NORTH/…/DONE; the history line
    # must therefore show the model's own cardinal emission, not the executed
    # primitive it expanded into.
    transcript = [
        {
            "kind": "step",
            "event_type": "TURNED",
            "position_after_row_col": (1, 2),
            "facing_after": "WEST",
            "action": "TURN_RIGHT",
            "cardinal_action": "MOVE_WEST",
            "prompt_feedback": "TURNED",
        }
    ]
    cfg = ExperimentConfig(
        observation="text_only", context_window="last3", action_space="cardinal"
    )

    prompt_text = _user_prompt_text_with_transcript(cfg, transcript)

    assert "FINAL_OUTPUT: MOVE_WEST" in prompt_text
    assert "FINAL_OUTPUT: TURN_RIGHT" not in prompt_text


def test_text_summary_and_last3_prompt_includes_summary_and_recent_history_text():
    transcript = [
        {
            "kind": "step",
            "event_type": "MOVED",
            "position_after_row_col": (1, 2),
            "facing_after": "EAST",
            "action": "MOVE_FORWARD",
            "prompt_feedback": "MOVED",
        }
    ]
    cfg = ExperimentConfig(observation="text_only", context_window="text_summary_and_last3")

    prompt_text = _user_prompt_text_with_transcript(cfg, transcript)

    assert "Activity summary:" in prompt_text
    assert "first you passed (1, 2)" in prompt_text
    assert "Recent history (last 3 steps, oldest first):" in prompt_text
    assert "Position after: (1, 2), facing EAST" in prompt_text
    assert "FINAL_OUTPUT: MOVE_FORWARD" in prompt_text
    assert "Feedback: MOVED" in prompt_text
    assert "What is your next action?" in prompt_text


def test_image_only_text_summary_and_last3_includes_summary_text_and_last3_images():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    transcript = [
        {
            "kind": "step",
            "event_type": "MOVED",
            "position_after_row_col": (1, 2),
            "action": "MOVE_FORWARD",
            "state_before": {"inventory": []},
            "_decision_frame_rgb": frame,
        },
    ]
    cfg = ExperimentConfig(observation="image_only", context_window="text_summary_and_last3")

    prompt_text = _user_prompt_text_with_transcript(cfg, transcript)

    assert "Activity summary:" in prompt_text
    assert "first you passed (1, 2)" in prompt_text
    # image_only never gets the last3 *text* history block, only the images.
    assert "Recent history (last 3 steps, oldest first):" not in prompt_text
    assert "Recent steps (oldest first):" in prompt_text
    assert "FINAL_OUTPUT: MOVE_FORWARD" in prompt_text
    # Summary must precede the last3 steps in the message content.
    assert prompt_text.index("Activity summary:") < prompt_text.index(
        "Recent steps (oldest first):"
    )


def test_image_text_summary_and_last3_orders_summary_before_last3():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    transcript = [
        {
            "kind": "step",
            "event_type": "MOVED",
            "position_after_row_col": (1, 2),
            "facing_after": "EAST",
            "action": "MOVE_FORWARD",
            "prompt_feedback": "MOVED",
            "state_before": {"inventory": []},
            "_decision_frame_rgb": frame,
        },
    ]
    cfg = ExperimentConfig(observation="image_text", context_window="text_summary_and_last3")

    prompt_text = _user_prompt_text_with_transcript(cfg, transcript)

    assert "Activity summary:" in prompt_text
    assert "first you passed (1, 2)" in prompt_text
    assert "Recent steps (oldest first):" in prompt_text
    assert "Recent history (last 3 steps, oldest first):" in prompt_text
    # Both the image label and text recap represent the prior model action
    # using exactly the same delimiter required for the next action.
    assert prompt_text.count("FINAL_OUTPUT: MOVE_FORWARD") == 2
    # Summary must precede the last3 steps (both the image labels and the
    # text recap) in the message content.
    summary_idx = prompt_text.index("Activity summary:")
    assert summary_idx < prompt_text.index("Recent steps (oldest first):")
    assert summary_idx < prompt_text.index("Recent history (last 3 steps, oldest first):")


def test_observation_format_image_only_differs_from_image_text_default():
    # After the fair-default rebase the baseline is image_text (with description),
    # so the image_only ablation arm no longer matches the default prompt text.
    standard_text = _initial_user_prompt_text(ExperimentConfig())
    image_only_text = _initial_user_prompt_text(
        CONDITION_SET.variants["standard"].build_config(ExperimentConfig())
    )

    assert image_only_text != standard_text
    # image_text carries the current-observation description ("You are at ..."),
    # image_only does not.
    assert "You are at" not in image_only_text
    assert "You are at" in standard_text


def test_minimal_prompt_uses_minimal_system_and_inventory_only_user_status():
    system_prompt = MinimalPromptStrategy(ACTIONS_HINT).build_system_prompt()
    # Isolate the minimal *prompting* axis from the observation: pin image_only so
    # the user status is inventory-only (the default is now image_text + description).
    prompt_text = _initial_user_prompt_text(ExperimentConfig(
        prompting="minimal", observation="image_only",
        include_current_observation_description=False,
        observation_text_includes_facing=False,
        in_context_learning="zero_shot"))

    assert system_prompt.startswith("Task: Solve the maze by reaching the goal.")
    assert "The environment may contain:" not in system_prompt
    assert prompt_text.startswith("Your inventory: empty.\nWhat is your next action?")
    assert "Observation:" not in prompt_text
    assert "Position:" not in prompt_text
    assert "Facing:" not in prompt_text
    assert "Goal:" not in prompt_text
    assert "Last result:" not in prompt_text


def test_prompting_variants_share_image_only_user_prompt():
    # Prompting only changes the SYSTEM prompt; the user prompt is identical across
    # prompting variants. Pin image_only to keep the user status inventory-only.
    def _cfg(p):
        return ExperimentConfig(prompting=p, observation="image_only",
                                include_current_observation_description=False,
                                observation_text_includes_facing=False,
                                in_context_learning="zero_shot")
    standard_text = _initial_user_prompt_text(_cfg("standard"))
    minimal_text = _initial_user_prompt_text(_cfg("minimal"))
    verbose_text = _initial_user_prompt_text(_cfg("verbose"))

    assert standard_text == minimal_text == verbose_text
    assert standard_text.startswith("Your inventory: empty.\nWhat is your next action?")
    assert "Position:" not in standard_text
    assert "Last result:" not in standard_text
    assert "Hints:" not in standard_text


def test_each_set_has_exactly_one_baseline_equal_to_default():
    # After the fair-default rebase the baseline arm of most sets carries an
    # explicit override that RESOLVES to the default (e.g. image_text, last3,
    # one_shot). The invariant is: every set has exactly one variant that builds
    # the default ExperimentConfig, and Prompt/standard is the canonical no-override
    # one (run once as the shared baseline for all axes).
    default = ExperimentConfig()
    prompt_std = CONDITION_SETS["Prompt"].variants["standard"]
    assert prompt_std.config_overrides is None
    assert prompt_std.build_config() == default

    for name, condition in CONDITION_SETS.items():
        baselines = [
            v for v in condition.variants.values()
            if v.implemented and v.build_config() == default
        ]
        assert len(baselines) == 1, (name, [v.name for v in baselines])


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
            elif variant.name == "cardinal":
                # Cardinal shares the standard template but advertises the
                # cardinal action vocabulary instead of the egocentric one.
                from interface import action_space

                expected = StandardPromptStrategy(
                    action_space.actions_hint("cardinal")
                ).build_system_prompt()
                assert system_prompt == expected, (condition_name, variant.name)
                assert system_prompt != standard_prompt
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
