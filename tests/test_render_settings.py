"""Run-config render block: which backend/camera/resolution an episode renders with."""

from __future__ import annotations

import pytest

from gridworld.render3d import RENDER3D_VERSION
from gridworld.render_settings import RenderSettings
from gridworld.task_spec import TaskSpecification
from render3d_test_utils import CORRIDOR  # 7x3 maze

SPEC = TaskSpecification.from_dict(CORRIDOR)
V = f"_r{RENDER3D_VERSION}"  # every 3D label ends with the render version


def test_default_is_the_2d_backend_and_changes_nothing():
    settings = RenderSettings.from_run_config({})
    assert settings == RenderSettings()
    assert settings.backend == "minigrid"
    assert settings.label == "minigrid"  # artifact paths and run hashes stay as they were
    assert settings.backend_kwargs(SPEC) == {}


def test_3d_block_sets_camera_and_sizes_the_frame_by_the_longer_side():
    settings = RenderSettings.from_run_config({"render": {"backend": "mujoco3d", "camera": "first_person"}})
    assert settings.label == "mujoco3d_first_person_grid" + V
    # "grid" = MiniGrid's 32 px per cell of the longer side, in a square frame:
    # the 2D pixel budget only for a square maze (this 7x3 one is 224x96 in 2D)
    assert settings.backend_kwargs(SPEC) == {"camera": "first_person", "resolution": 7 * 32}


def test_fixed_resolution_is_recorded_in_the_label():
    settings = RenderSettings.from_run_config(
        {"render": {"backend": "mujoco3d", "camera": "first_person", "resolution": 512}}
    )
    assert settings.label == "mujoco3d_first_person_512" + V
    assert settings.backend_kwargs(SPEC) == {"camera": "first_person", "resolution": 512}


def test_a_render_version_bump_moves_3d_artifacts_to_a_new_label(monkeypatch):
    """The version is in the artifact path: a rerun after a visual change writes
    beside the old episodes instead of over them. 2D has no render version."""
    import gridworld.render3d as render3d

    settings = RenderSettings.from_run_config({"render": {"backend": "mujoco3d", "camera": "top_down"}})
    before = settings.label
    monkeypatch.setattr(render3d, "RENDER3D_VERSION", "99")
    assert settings.label == "mujoco3d_top_down_grid_r99" != before
    assert RenderSettings().label == "minigrid"


@pytest.mark.parametrize(
    "block, message",
    [
        ({"backend": "minigrid", "camera": "first_person"}, "camera"),
        ({"backend": "mujoco3d"}, "camera"),
        ({"backend": "mujoco3d", "camera": "isometric"}, "camera"),
        # retired presets: a run-config naming one fails fast, before any call
        ({"backend": "mujoco3d", "camera": "chase"}, "camera"),
        ({"backend": "mujoco3d", "camera": "fixed_angled"}, "camera"),
        ({"backend": "mujoco3d", "camera": "first_person_narrow"}, "camera"),
        ({"backend": "holodeck", "camera": "first_person"}, "backend"),
        ({"backend": "mujoco3d", "camera": "first_person", "resolution": 0}, "resolution"),
        ({"backend": "mujoco3d", "camera": "first_person", "zoom": 2}, "zoom"),
    ],
)
def test_bad_render_blocks_fail_before_any_paid_call(block, message):
    with pytest.raises(ValueError, match=message):
        RenderSettings.from_run_config({"render": block})


# --- start map (run-config render.start_map) ---------------------------------

FIRST_PERSON_MAP = {"backend": "mujoco3d", "camera": "first_person", "start_map": True}


def test_start_map_gets_its_own_label_kwarg_and_provenance():
    settings = RenderSettings.from_run_config({"render": FIRST_PERSON_MAP})
    assert settings.start_map is True
    # its own artifact directory and run hash: never collides with a no-map run
    assert settings.label == "mujoco3d_first_person_grid_map" + V
    assert settings.backend_kwargs(SPEC) == {"camera": "first_person", "resolution": 7 * 32, "start_map": True}
    # the flag is part of the 3D cache key and the recorded render provenance
    assert settings.provenance(SPEC)["start_map"] is True


@pytest.mark.parametrize("block", [
    {"backend": "mujoco3d", "camera": "first_person"},
    {"backend": "mujoco3d", "camera": "first_person", "start_map": False},
])
def test_no_start_map_changes_nothing(block):
    settings = RenderSettings.from_run_config({"render": block})
    assert settings.start_map is False
    assert settings.label == "mujoco3d_first_person_grid" + V
    assert settings.backend_kwargs(SPEC) == {"camera": "first_person", "resolution": 7 * 32}
    assert settings.provenance(SPEC)["start_map"] is False


@pytest.mark.parametrize(
    "block, message",
    [
        ({"backend": "minigrid", "start_map": True}, "start_map"),
        ({"start_map": True}, "start_map"),
        ({"backend": "mujoco3d", "camera": "top_down", "start_map": True}, "top_down"),
        ({"backend": "mujoco3d", "camera": "first_person", "start_map": "yes"}, "bool"),
        ({"backend": "mujoco3d", "camera": "first_person", "start_map": 1}, "bool"),
        ({"backend": "mujoco3d", "camera": "first_person", "start_map": None}, "bool"),
    ],
)
def test_bad_start_map_fails_before_any_paid_call(block, message):
    with pytest.raises(ValueError, match=message):
        RenderSettings.from_run_config({"render": block})


def test_start_map_needs_an_observation_that_carries_images():
    settings = RenderSettings.from_run_config({"render": FIRST_PERSON_MAP})
    settings.check_observation("image_only")
    settings.check_observation("image_text")
    with pytest.raises(ValueError, match="text_only"):
        settings.check_observation("text_only")
    # 2D allows every observation
    RenderSettings().check_observation("text_only")


def test_3d_refuses_text_only_observation():
    """text_only never shows a frame, so a 3D run would repeat the 2D run under
    another label and pay for GL work the model never sees."""
    settings = RenderSettings.from_run_config({"render": {"backend": "mujoco3d", "camera": "top_down"}})
    settings.check_observation("image_only")
    settings.check_observation("image_text")
    with pytest.raises(ValueError, match="text_only"):
        settings.check_observation("text_only")


@pytest.mark.parametrize("block", [[], "", 0, False, None, "mujoco3d", ["mujoco3d"]])
def test_a_present_render_block_must_be_an_object(block):
    # A falsy non-object must not silently mean "2D": only an absent block does.
    with pytest.raises(ValueError, match="'render' must be an object"):
        RenderSettings.from_run_config({"render": block})


def test_an_empty_render_object_is_the_2d_default():
    assert RenderSettings.from_run_config({"render": {}}) == RenderSettings()


@pytest.mark.parametrize(
    "block",
    [
        {"backend": "minigrid", "resolution": 512},
        {"resolution": 512},
        {"backend": "minigrid", "resolution": "grid"},
        {"camera": "first_person"},
    ],
)
def test_3d_only_keys_are_rejected_for_the_minigrid_backend(block):
    # MiniGrid ignores resolution/camera: accepting them would record a setting
    # the run never used.
    with pytest.raises(ValueError, match="3D setting"):
        RenderSettings.from_run_config({"render": block})


# --- 2D observability override (run-config render.observability) ------------

from gridworld.render_settings import FOG_RENDER_VERSION  # noqa: E402
from interface.config import ExperimentConfig  # noqa: E402


@pytest.mark.parametrize("block, mode, view", [
    ({"observability": "fog_of_war"}, "fog_of_war", 7),  # MiniGrid's default view
    ({"backend": "minigrid", "observability": "fog_of_war", "view_size": 5}, "fog_of_war", 5),
    ({"observability": "view_cone", "view_size": 9}, "view_cone", 9),
])
def test_an_observability_block_imposes_its_rules_under_its_own_label(block, mode, view):
    settings = RenderSettings.from_run_config({"render": block})
    assert (settings.backend, settings.observability, settings.view_size) == ("minigrid", mode, view)
    # its own artifact directory and run hash, versioned like a 3D render
    assert settings.label == f"minigrid_{mode}_v{view}_r{FOG_RENDER_VERSION}"
    assert settings.backend_kwargs(SPEC) == {}
    assert settings.unit_block() == {"backend": "minigrid", "observability": mode, "view_size": view}
    assert settings.provenance(SPEC) == {**settings.unit_block(), "fog_render_version": FOG_RENDER_VERSION}
    assert RenderSettings.from_run_config({"render": settings.unit_block()}) == settings
    fogged = settings.apply_to_spec(SPEC)
    assert (fogged.rules.observability, fogged.rules.view_size) == (mode, view)
    assert fogged.maze == SPEC.maze and fogged.mechanisms == SPEC.mechanisms
    assert SPEC.rules.observability == "full"  # the task file's spec is untouched


def test_without_the_key_2d_settings_change_nothing():
    settings = RenderSettings.from_run_config({"render": {"backend": "minigrid"}})
    assert settings == RenderSettings()
    assert settings.provenance(SPEC) is None and settings.unit_block() is None
    assert settings.apply_to_spec(SPEC) is SPEC


def test_3d_unit_block_and_provenance_carry_no_observability_fields():
    """The 3D records predate the 2D override; their keys (and hashes) stay put."""
    settings = RenderSettings.from_run_config({"render": {"backend": "mujoco3d", "camera": "top_down"}})
    assert settings.unit_block() == {
        "backend": "mujoco3d", "camera": "top_down", "resolution": "grid", "start_map": False}


@pytest.mark.parametrize("block, message", [
    ({"observability": "full"}, "observability must be one of"),
    ({"observability": "fog"}, "observability must be one of"),
    ({"observability": None}, "observability must be one of"),
    ({"view_size": 5}, "needs render observability"),
    ({"observability": "fog_of_war", "view_size": 4}, "odd int >= 3"),
    ({"observability": "fog_of_war", "view_size": 1}, "odd int >= 3"),
    ({"observability": "fog_of_war", "view_size": True}, "odd int >= 3"),
    ({"observability": "fog_of_war", "view_size": "7"}, "odd int >= 3"),
    ({"backend": "mujoco3d", "camera": "top_down", "observability": "fog_of_war"}, "2D setting"),
])
def test_bad_observability_blocks_fail_before_any_paid_call(block, message):
    with pytest.raises(ValueError, match=message):
        RenderSettings.from_run_config({"render": block})


def test_fog_needs_image_only_and_a_prompt_without_the_maze_text():
    settings = RenderSettings.from_run_config({"render": {"observability": "fog_of_war"}})
    settings.check_experiment(ExperimentConfig(observation="image_only", prompting="minimal"))
    for observation in ("text_only", "image_text"):
        with pytest.raises(ValueError, match="image_only"):
            settings.check_experiment(ExperimentConfig(observation=observation))
    with pytest.raises(ValueError, match="text_initial_maze"):
        settings.check_experiment(ExperimentConfig(observation="image_only", prompting="text_initial_maze"))
    # the plain 2D default accepts every cell, as before
    RenderSettings().check_experiment(ExperimentConfig(observation="text_only", prompting="text_initial_maze"))
