from __future__ import annotations

from dataclasses import replace

from interface.config import ExperimentConfig
from interface.loader import default_maze_path, load_task
from interface.observation import current_observation_text
from interface.parser import ACTIONS_HINT
from interface.prompt_strategies import (
    StandardPromptStrategy,
    VerbosePromptStrategy,
    _mechanism_hints_text,
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
        assert "You are at (1, 1) facing EAST." in text


def test_observation_format_image_only_still_omits_text_description():
    spec, state = _initial_spec_and_state()
    cfg = CONDITION_SET.variants["image_only"].build_config(
        replace(ExperimentConfig(), observation_text_includes_facing=False)
    )

    text = current_observation_text(
        cfg.observation,
        spec,
        state,
        include_description=cfg.include_current_observation_description,
        include_facing=cfg.observation_text_includes_facing,
    )

    assert text == ""


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


def test_observation_format_image_only_matches_standard_prompt_text():
    standard_text = _initial_user_prompt_text(ExperimentConfig())
    image_only_text = _initial_user_prompt_text(
        CONDITION_SET.variants["image_only"].build_config(ExperimentConfig())
    )

    assert image_only_text == standard_text


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


def test_mechanism_hint_insertion_helper_still_generates_hints():
    _backend, spec = load_task(default_maze_path("V04_single_key.json"))
    hints = _mechanism_hints_text(spec)

    assert "Hints:" in hints
    assert "Face an adjacent key and PICKUP" in hints


def test_verbose_prompt_can_insert_mechanism_hints_when_enabled():
    class HintingVerbosePromptStrategy(VerbosePromptStrategy):
        include_mechanism_hints = True

    backend, spec = load_task(default_maze_path("V04_single_key.json"))
    runner = build_runner(ExperimentConfig(prompting="verbose"), backend, spec)
    runner.prompt = HintingVerbosePromptStrategy(ACTIONS_HINT)
    runner.last_rgb, state, _info = backend.reset(seed=spec.seed)

    message = runner._build_message(state, feedback_templates.INITIAL_FEEDBACK, [])
    content = message["content"]
    prompt_text = "\n".join(
        block["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )

    assert "Hints:" in prompt_text
    assert "Face an adjacent key and PICKUP" in prompt_text
