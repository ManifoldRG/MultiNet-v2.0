"""Run-config render block: which backend/camera/resolution an episode renders with."""

from __future__ import annotations

import pytest

from gridworld.render_settings import RenderSettings
from gridworld.task_spec import TaskSpecification
from render3d_test_utils import CORRIDOR  # 7x3 maze

SPEC = TaskSpecification.from_dict(CORRIDOR)


def test_default_is_the_2d_backend_and_changes_nothing():
    settings = RenderSettings.from_run_config({})
    assert settings == RenderSettings()
    assert settings.backend == "minigrid"
    assert settings.label == "minigrid"  # artifact paths and run hashes stay as they were
    assert settings.backend_kwargs(SPEC) == {}


def test_3d_block_sets_camera_and_sizes_the_frame_by_the_longer_side():
    settings = RenderSettings.from_run_config({"render": {"backend": "mujoco3d", "camera": "first_person"}})
    assert settings.label == "mujoco3d_first_person_grid"
    # "grid" = MiniGrid's 32 px per cell of the longer side, in a square frame:
    # the 2D pixel budget only for a square maze (this 7x3 one is 224x96 in 2D)
    assert settings.backend_kwargs(SPEC) == {"camera": "first_person", "resolution": 7 * 32}


def test_fixed_resolution_is_recorded_in_the_label():
    settings = RenderSettings.from_run_config(
        {"render": {"backend": "mujoco3d", "camera": "first_person", "resolution": 512}}
    )
    assert settings.label == "mujoco3d_first_person_512"
    assert settings.backend_kwargs(SPEC) == {"camera": "first_person", "resolution": 512}


@pytest.mark.parametrize(
    "block, message",
    [
        ({"backend": "minigrid", "camera": "first_person"}, "camera"),
        ({"backend": "mujoco3d"}, "camera"),
        ({"backend": "mujoco3d", "camera": "isometric"}, "camera"),
        # retired presets: a run-config naming one fails fast, before any call
        ({"backend": "mujoco3d", "camera": "chase"}, "camera"),
        ({"backend": "mujoco3d", "camera": "fixed_angled"}, "camera"),
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
    assert settings.label == "mujoco3d_first_person_grid_map"
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
    assert settings.label == "mujoco3d_first_person_grid"
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
    # without a start map every observation stays allowed
    RenderSettings().check_observation("text_only")


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
