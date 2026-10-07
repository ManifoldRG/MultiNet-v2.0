"""The 3D backend's start map: a top-down snapshot of the reset state, taken
once per episode at the run's frame resolution, None everywhere else."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("minigrid")

from gridworld.actions import MiniGridActions as A  # noqa: E402
from gridworld.backends import get_backend  # noqa: E402
from gridworld.render3d.renderer import SceneRenderer  # noqa: E402
from gridworld.task_spec import TaskSpecification  # noqa: E402
from render3d_test_utils import CORRIDOR  # noqa: E402
from start_map_test_utils import r1_spec  # noqa: E402

RES = 96


def _fresh_top_down(backend, spec, state):
    renderer = SceneRenderer(spec, camera="top_down", resolution=RES)
    try:
        return renderer.render(
            state,
            backend.door_states(),
            rotators=backend.rotator_directions(),
            freeze=backend.freeze_remaining(),
        )
    finally:
        renderer.close()


@pytest.mark.parametrize("camera", ["first_person", "first_person_narrow"])
def test_start_map_is_a_fresh_top_down_render_of_the_reset_state(camera):
    spec = r1_spec()
    backend = get_backend("mujoco3d", camera=camera, resolution=RES, start_map=True)
    backend.configure(spec)
    try:
        assert backend.start_map_frame() is None  # nothing until an episode starts
        view, state, _ = backend.reset(seed=0)
        start_map = backend.start_map_frame()
        assert start_map.shape == (RES, RES, 3)
        assert np.array_equal(start_map, _fresh_top_down(backend, spec, state))
        assert not np.array_equal(start_map, view)  # the map is not the agent's view
    finally:
        backend.close()


def test_start_map_is_a_snapshot_that_never_updates():
    spec = TaskSpecification.from_dict(CORRIDOR)
    backend = get_backend("mujoco3d", camera="first_person", resolution=RES, start_map=True)
    backend.configure(spec)
    try:
        _, state0, _ = backend.reset(seed=0)
        start_map = backend.start_map_frame().copy()
        for action in (A.MOVE_FORWARD, A.PICKUP, A.TOGGLE, A.MOVE_FORWARD):
            backend.step(int(action))
        assert np.array_equal(backend.start_map_frame(), start_map)
        # a new episode takes a new snapshot
        backend.reset(seed=0)
        assert np.array_equal(backend.start_map_frame(), _fresh_top_down(backend, spec, state0))
    finally:
        backend.close()


def test_start_map_does_not_change_the_agents_view():
    spec = r1_spec()
    plain = get_backend("mujoco3d", camera="first_person", resolution=RES)
    mapped = get_backend("mujoco3d", camera="first_person", resolution=RES, start_map=True)
    for backend in (plain, mapped):
        backend.configure(spec)
    try:
        assert np.array_equal(plain.reset(seed=0)[0], mapped.reset(seed=0)[0])
        for action in (A.MOVE_FORWARD, A.TURN_RIGHT, A.MOVE_FORWARD):
            assert np.array_equal(plain.step(int(action))[0], mapped.step(int(action))[0])
    finally:
        plain.close()
        mapped.close()


def test_start_map_follows_a_reconfigured_spec():
    corridor, r1 = TaskSpecification.from_dict(CORRIDOR), r1_spec()
    backend = get_backend("mujoco3d", camera="first_person", resolution=RES, start_map=True)
    try:
        backend.configure(corridor)
        backend.reset(seed=0)
        backend.configure(r1)
        assert backend.start_map_frame() is None  # the old maze's map is not offered
        _, state, _ = backend.reset(seed=0)
        assert np.array_equal(backend.start_map_frame(), _fresh_top_down(backend, r1, state))
    finally:
        backend.close()


def test_close_releases_the_map_renderer():
    backend = get_backend("mujoco3d", camera="first_person", resolution=RES, start_map=True)
    backend.configure(r1_spec())
    backend.reset(seed=0)
    assert backend._map_renderer is not None
    backend.close()
    assert backend._map_renderer is None


def test_no_start_map_without_the_flag_or_on_2d():
    spec = r1_spec()
    plain = get_backend("mujoco3d", camera="first_person", resolution=RES)
    flat = get_backend("minigrid", render_mode="rgb_array")
    for backend in (plain, flat):
        backend.configure(spec)
        backend.reset(seed=0)
        assert backend.start_map_frame() is None
        backend.close()
    assert plain.start_map is False


def test_start_map_with_the_top_down_camera_is_rejected():
    with pytest.raises(ValueError, match="top_down"):
        get_backend("mujoco3d", camera="top_down", resolution=RES, start_map=True)
