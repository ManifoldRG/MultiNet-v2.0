"""The 3D wrapper must pass state through untouched: same GridState, reward,
termination and info as the bare state backend, every step."""

from __future__ import annotations

import copy
import random

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("minigrid")

from gridworld.actions import MiniGridActions as A  # noqa: E402
from gridworld.backends import get_backend  # noqa: E402
from gridworld.backends.minigrid_backend import MiniGridBackend  # noqa: E402
from gridworld.baselines import plan_bfs_path  # noqa: E402
from gridworld.render3d.scene import UnsupportedSpecError  # noqa: E402
from gridworld.task_spec import TaskSpecification  # noqa: E402
from maze_test_utils import MAZE_JSON_DIR  # noqa: E402
from render3d_test_utils import CORRIDOR, corpus_sample  # noqa: E402

RES = 96
_SET_FIELDS = ("open_doors", "collected_keys", "active_switches", "open_gates", "visible_cells", "explored_cells")


def _canon(state) -> dict:
    d = state.to_dict()
    for name in _SET_FIELDS:
        d[name] = sorted(map(str, d[name]))
    return d


def _assert_lockstep(spec, actions):
    ref = MiniGridBackend(render_mode="rgb_array")
    wrapped = get_backend("mujoco3d", resolution=RES)
    ref.configure(spec)
    wrapped.configure(spec)
    try:
        _, s_ref, i_ref = ref.reset(seed=spec.seed)
        frame, s_w, i_w = wrapped.reset(seed=spec.seed)
        assert frame.shape == (RES, RES, 3) and frame.dtype == np.uint8
        assert _canon(s_w) == _canon(s_ref) and i_w == i_ref
        for action in actions:
            _, r1, t1, u1, s1, i1 = ref.step(action)
            frame, r2, t2, u2, s2, i2 = wrapped.step(action)
            assert (r2, t2, u2, i2) == (r1, t1, u1, i1)
            assert _canon(s2) == _canon(s1)
            assert wrapped.door_states() == ref.door_states()
            assert frame.shape == (RES, RES, 3)
            if t1 or u1:
                break
    finally:
        wrapped.close()
        ref.close()


@pytest.mark.parametrize("path", corpus_sample(), ids=lambda p: p.stem)
def test_bfs_plan_replays_identically_through_wrapper(path):
    spec = TaskSpecification.from_json(str(path))
    plan = plan_bfs_path(spec)
    assert plan.success
    _assert_lockstep(spec, plan.actions)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("path", corpus_sample(), ids=lambda p: p.stem)
def test_random_actions_replay_identically_through_wrapper(path, seed):
    spec = TaskSpecification.from_json(str(path))
    rng = random.Random(f"{path.stem}-{seed}")
    _assert_lockstep(spec, [rng.randrange(7) for _ in range(60)])


@pytest.mark.slow
@pytest.mark.parametrize(
    "path", sorted(MAZE_JSON_DIR.rglob("*.json")), ids=lambda p: f"{p.parent.name}/{p.stem}"
)
def test_full_corpus_bfs_replays_identically_through_wrapper(path):
    spec = TaskSpecification.from_json(str(path))
    _assert_lockstep(spec, plan_bfs_path(spec).actions)


def test_corridor_reaches_goal_and_reclosed_door_is_drawn_closed():
    spec = TaskSpecification.from_dict(CORRIDOR)
    backend = get_backend("mujoco3d", resolution=RES)
    backend.configure(spec)
    backend.reset(seed=0)
    try:
        for action in (A.MOVE_FORWARD, A.PICKUP):
            backend.step(int(action))
        opened = backend.step(int(A.TOGGLE))[0].copy()
        reclosed, _r, _t, _u, state, _i = backend.step(int(A.TOGGLE))
        assert backend.door_states() == {"d1": False} and "d1" in state.open_doors
        assert not np.array_equal(opened, reclosed)
        backend.step(int(A.TOGGLE))  # reopen
        for _ in range(3):
            *_, state, _info = backend.step(int(A.MOVE_FORWARD))
        assert state.goal_reached
    finally:
        backend.close()


def test_render_is_cached_until_state_or_camera_changes(monkeypatch):
    backend = get_backend("mujoco3d", resolution=RES)
    backend.configure(TaskSpecification.from_dict(CORRIDOR))
    backend.reset(seed=0)
    try:
        calls = []
        real_render = backend._renderer.render
        monkeypatch.setattr(
            backend._renderer, "render", lambda s, d: calls.append(1) or real_render(s, d)
        )
        first, second = backend.render(), backend.render()
        assert first is second and calls == []  # reset already rendered this state
        backend.set_camera("chase")
        assert backend.render().shape == (RES, RES, 3) and calls == [1]
        backend.step(int(A.TURN_LEFT))
        assert calls == [1, 1]
    finally:
        backend.close()


def test_configure_rejects_unsupported_spec_before_touching_state_backend():
    d = copy.deepcopy(CORRIDOR)
    d["mechanisms"] = {"blocks": [{"id": "b1", "position": [4, 1]}]}
    backend = get_backend("mujoco3d", resolution=RES)
    with pytest.raises(UnsupportedSpecError, match="blocks"):
        backend.configure(TaskSpecification.from_dict(d))
    assert not backend.is_configured and not backend.state_backend.is_configured


def test_registry_wraps_minigrid_by_default():
    backend = get_backend("mujoco3d", camera="chase", resolution=RES)
    assert isinstance(backend.state_backend, MiniGridBackend)
    assert backend.camera == "chase"
    assert backend.camera_names == ("top_down", "chase", "fixed_angled", "first_person")
    assert backend.frame_is_grid_aligned is False
    assert backend.observation_shape == (RES, RES, 3)


def test_unknown_camera_rejected():
    with pytest.raises(ValueError, match="unknown camera preset"):
        get_backend("mujoco3d", camera="isometric")
