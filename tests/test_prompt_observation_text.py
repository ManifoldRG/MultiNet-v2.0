from __future__ import annotations

from dataclasses import replace

from interface.config import ExperimentConfig
from interface.loader import default_maze_path, load_task
from interface.observation import current_observation_text
from interface.runner import build_runner
from prompting_experiments import CONDITION_SETS
from prompting_experiments.condition_set_2_observation_format import CONDITION_SET
from prompting_experiments.prompt_templates import feedback as feedback_templates


def _initial_spec_and_state():
    backend, spec = load_task(default_maze_path())
    _rgb, state, _info = backend.reset(seed=spec.seed)
    return spec, state


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
