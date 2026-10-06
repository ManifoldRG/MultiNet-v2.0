"""3D-aware prompt scaffolding: the start-map text and current-view label per
prompting level, and the 3D task prefix per camera family.

The render context comes from the backend (camera + whether it has a start
map), never from ExperimentConfig, so config hashes do not move."""

from __future__ import annotations

import pytest

from gridworld.render3d.cameras import PRESETS
from interface.action_space import actions_hint
from interface.prompt_strategies import (
    MinimalPromptStrategy,
    RenderContext,
    StandardPromptStrategy,
    TextInitialMazePromptStrategy,
    VerbosePromptStrategy,
)
from prompting_experiments.prompt_templates import system as system_templates
from prompting_experiments.prompt_templates import user as user_templates

HINT = actions_hint("egocentric")
LEVELS = {
    "minimal": MinimalPromptStrategy,
    "standard": StandardPromptStrategy,
    "verbose": VerbosePromptStrategy,
    "text_initial_maze": TextInitialMazePromptStrategy,
}
MAP_CAMERAS = ("first_person", "first_person_narrow", "chase", "fixed_angled")
FIRST_PERSON = ("first_person", "first_person_narrow")
THIRD_PERSON = ("top_down", "chase", "fixed_angled")

MINIMAL_MAP = (
    "Map: a top-down view of the whole maze at the start, north up. "
    "It is a snapshot and does not change as you move."
)
STANDARD_MAP = (
    "Map: a top-down view of the whole maze at the start, north up. "
    "The red wedge is where you started, pointing the way you faced. "
    "It is a snapshot and does not change as you move."
)
VIEW_LABELS = {
    "first_person": "Your view now (first person):",
    "first_person_narrow": "Your view now (first person):",
    "chase": "Your view now (from behind and above you):",
    "fixed_angled": "Your view now (from a fixed angle):",
}
THIRD_PERSON_PREFIX = (
    "Task: You are the red wedge-shaped agent trying to navigate this maze. "
    "The wedge points the way you face. Move to the green goal tile."
)
FIRST_PERSON_PREFIX = (
    "Task: You are navigating this maze and see it through your own eyes; "
    "the view looks the way you face. Move to the green goal tile."
)


def _strategy(level, camera=None, start_map=False):
    return LEVELS[level](HINT, render=RenderContext(camera=camera, start_map=start_map))


# --- start-map text ------------------------------------------------------------

@pytest.mark.parametrize("camera", MAP_CAMERAS)
def test_minimal_map_text_and_no_view_label(camera):
    strategy = _strategy("minimal", camera, start_map=True)
    assert strategy.start_map_text() == MINIMAL_MAP
    assert strategy.current_view_label() is None  # minimal keeps today's layout


@pytest.mark.parametrize("level", ["standard", "text_initial_maze"])
@pytest.mark.parametrize("camera", MAP_CAMERAS)
def test_standard_map_text_and_view_label(level, camera):
    strategy = _strategy(level, camera, start_map=True)
    assert strategy.start_map_text() == STANDARD_MAP
    assert strategy.current_view_label() == VIEW_LABELS[camera]


LEGEND_MENTIONS = {
    "walls": ("wall",),
    "floor": ("floor",),
    "you": ("red wedge is you",),
    "goal": ("goal",),
    "keys": ("key",),
    "doors": ("closed door", "open door"),
    "switches": ("switch", "off", "on"),
    "gates": ("closed gate", "open gate"),
    "portals": ("portal", "pairs", "sign"),
    "death tiles": ("skull", "death tile"),
    "ice tiles": ("ice tile",),
    "turntables": ("turntable", "arrow", "direction it carries you"),
    "compass": ("compass", "north"),
}


@pytest.mark.parametrize("camera", MAP_CAMERAS)
def test_verbose_map_text_is_standard_plus_a_full_legend(camera):
    text = _strategy("verbose", camera, start_map=True).start_map_text()
    assert text.startswith(STANDARD_MAP + "\n")
    legend = text[len(STANDARD_MAP) + 1:]
    assert legend.startswith("The map may contain:\n")
    for kind, words in LEGEND_MENTIONS.items():
        for word in words:
            assert word in legend, (kind, word)
    # one short sentence per thing shown
    items = legend.splitlines()[1:]
    assert len(items) == 12 and all(item.startswith("  - ") and item.endswith(".") for item in items)


@pytest.mark.parametrize("camera", MAP_CAMERAS)
def test_verbose_view_label_explains_the_compass_on_views_that_turn(camera):
    label = _strategy("verbose", camera, start_map=True).current_view_label()
    assert label.endswith("\n" + VIEW_LABELS[camera]) or label == VIEW_LABELS[camera]
    if camera == "fixed_angled":  # north-up view: no turning compass to explain
        assert label == VIEW_LABELS[camera]
    else:
        compass = label.splitlines()[0]
        assert compass == user_templates.COMPASS_NOTE
        assert "compass" in compass and "north" in compass and "face" in compass


@pytest.mark.parametrize("level", list(LEVELS))
@pytest.mark.parametrize("camera", [None, *PRESETS])
def test_without_a_start_map_there_is_no_map_text_or_label(level, camera):
    strategy = _strategy(level, camera, start_map=False)
    assert strategy.start_map_text() is None
    assert strategy.current_view_label() is None
    assert LEVELS[level](HINT).start_map_text() is None  # default context = 2D


def test_appearance_words_live_in_the_one_template_block():
    assert user_templates.START_MAP_MINIMAL == MINIMAL_MAP
    assert user_templates.START_MAP_STANDARD == STANDARD_MAP
    assert user_templates.START_MAP_VERBOSE.startswith(STANDARD_MAP)


# --- the label sits right before the current image ----------------------------

def test_view_label_precedes_the_current_image_placeholder():
    from gridworld.backends.base import GridState

    state = GridState(agent_position=(1, 1), agent_direction=0)
    placeholder = user_templates.CURRENT_IMAGE_PLACEHOLDER
    labelled = _strategy("standard", "chase", start_map=True).build_user_prompt(
        "", "", state, observation="image_only"
    )
    assert labelled == f"{VIEW_LABELS['chase']}\n{placeholder}\nYour inventory: empty.\nWhat is your next action?"
    for level in ("minimal",):
        plain = _strategy(level, "chase", start_map=True).build_user_prompt("", "", state, observation="image_only")
        assert plain.startswith(placeholder)
    no_map = _strategy("standard", "chase").build_user_prompt("", "", state, observation="image_only")
    assert no_map.startswith(placeholder)


# --- 3D task prefixes ---------------------------------------------------------

@pytest.mark.parametrize("level", ["standard", "verbose", "text_initial_maze"])
@pytest.mark.parametrize("start_map", [False, True])
@pytest.mark.parametrize("camera", PRESETS)
def test_3d_system_prompt_names_the_agent_for_its_camera(level, camera, start_map):
    if start_map and camera == "top_down":
        pytest.skip("top_down + start_map is rejected")
    prompt = _strategy(level, camera, start_map).build_system_prompt()
    prefix = FIRST_PERSON_PREFIX if camera in FIRST_PERSON else THIRD_PERSON_PREFIX
    two_d = LEVELS[level](HINT).build_system_prompt()
    assert two_d.startswith(system_templates.TASK_PREFIX)
    # only the task line changes; the mechanisms, rules and actions stay
    assert prompt == prefix + two_d[len(system_templates.TASK_PREFIX):]


@pytest.mark.parametrize("camera", [None, *PRESETS])
def test_minimal_system_prompt_is_camera_neutral(camera):
    prompt = _strategy("minimal", camera).build_system_prompt()
    assert prompt == MinimalPromptStrategy(HINT).build_system_prompt()
    assert prompt.startswith(system_templates.MIN_TASK_PREFIX)


def test_camera_families_cover_every_preset():
    assert set(FIRST_PERSON) | set(THIRD_PERSON) == set(PRESETS)
    assert system_templates.TASK_PREFIX_3D_FIRST_PERSON == FIRST_PERSON_PREFIX
    assert system_templates.TASK_PREFIX_3D_THIRD_PERSON == THIRD_PERSON_PREFIX


# --- the render context -------------------------------------------------------

def test_render_context_rejects_impossible_combinations():
    with pytest.raises(ValueError, match="camera"):
        RenderContext(camera="isometric")
    with pytest.raises(ValueError, match="start_map"):
        RenderContext(camera=None, start_map=True)
    with pytest.raises(ValueError, match="top_down"):
        RenderContext(camera="top_down", start_map=True)


def test_render_context_comes_from_the_backend():
    pytest.importorskip("mujoco")
    pytest.importorskip("minigrid")
    from gridworld.backends import get_backend

    assert RenderContext.from_backend(get_backend("minigrid", render_mode="rgb_array")) == RenderContext()
    plain = get_backend("mujoco3d", camera="chase", resolution=64)
    mapped = get_backend("mujoco3d", camera="first_person", resolution=64, start_map=True)
    assert RenderContext.from_backend(plain) == RenderContext(camera="chase")
    assert RenderContext.from_backend(mapped) == RenderContext(camera="first_person", start_map=True)


@pytest.mark.parametrize("camera, start_map", [("first_person", True), ("chase", False), ("top_down", False)])
def test_build_runner_hands_the_backends_render_context_to_the_prompt(camera, start_map):
    pytest.importorskip("mujoco")
    pytest.importorskip("minigrid")
    from gridworld.backends import get_backend
    from interface.config import ExperimentConfig
    from interface.runner import build_runner
    from start_map_test_utils import r1_spec

    spec = r1_spec()
    backend = get_backend("mujoco3d", camera=camera, resolution=64, start_map=start_map)
    backend.configure(spec)
    try:
        runner = build_runner(ExperimentConfig(prompting="standard"), backend, spec)
        assert runner.prompt.render == RenderContext(camera=camera, start_map=start_map)
        prefix = FIRST_PERSON_PREFIX if camera in FIRST_PERSON else THIRD_PERSON_PREFIX
        assert runner.prompt.build_system_prompt().startswith(prefix)
    finally:
        backend.close()
    flat = get_backend("minigrid", render_mode="rgb_array")
    flat.configure(spec)
    runner = build_runner(ExperimentConfig(prompting="standard"), flat, spec)
    assert runner.prompt.render == RenderContext()
    assert runner.prompt.build_system_prompt().startswith(system_templates.TASK_PREFIX)
