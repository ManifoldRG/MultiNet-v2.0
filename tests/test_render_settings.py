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


def test_3d_block_sets_camera_and_matches_the_2d_pixel_budget():
    settings = RenderSettings.from_run_config({"render": {"backend": "mujoco3d", "camera": "chase"}})
    assert settings.label == "mujoco3d_chase_grid"
    # "grid" = MiniGrid's 32 px per cell, so a frame costs the same image tokens as 2D
    assert settings.backend_kwargs(SPEC) == {"camera": "chase", "resolution": 7 * 32}


def test_fixed_resolution_is_recorded_in_the_label():
    settings = RenderSettings.from_run_config(
        {"render": {"backend": "mujoco3d", "camera": "first_person", "resolution": 512}}
    )
    assert settings.label == "mujoco3d_first_person_512"
    assert settings.backend_kwargs(SPEC) == {"camera": "first_person", "resolution": 512}


@pytest.mark.parametrize(
    "block, message",
    [
        ({"backend": "minigrid", "camera": "chase"}, "camera"),
        ({"backend": "mujoco3d"}, "camera"),
        ({"backend": "mujoco3d", "camera": "isometric"}, "camera"),
        ({"backend": "holodeck", "camera": "chase"}, "backend"),
        ({"backend": "mujoco3d", "camera": "chase", "resolution": 0}, "resolution"),
        ({"backend": "mujoco3d", "camera": "chase", "zoom": 2}, "zoom"),
    ],
)
def test_bad_render_blocks_fail_before_any_paid_call(block, message):
    with pytest.raises(ValueError, match=message):
        RenderSettings.from_run_config({"render": block})


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
        {"camera": "chase"},
    ],
)
def test_3d_only_keys_are_rejected_for_the_minigrid_backend(block):
    # MiniGrid ignores resolution/camera: accepting them would record a setting
    # the run never used.
    with pytest.raises(ValueError, match="3D setting"):
        RenderSettings.from_run_config({"render": block})
