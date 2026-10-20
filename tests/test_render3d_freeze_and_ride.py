"""Frozen-state frost (Sean 2026-09-23: fills the screen on step-on, then
lightens and cracks over the five no-op actions until thaw) and the
ride-direction glyph while the agent stands on a rotating tile."""

from __future__ import annotations

import numpy as np
import pytest

from gridworld.backends.base import GridState
from gridworld.render3d import hud
from gridworld.task_spec import TaskSpecification
from render3d_test_utils import TILES

RES = 256
CORRIDOR_FROZEN = {
    "task_id": "frz", "seed": 0, "difficulty_tier": 1,
    "maze": {"dimensions": [8, 3], "walls": [], "start": [1, 1], "goal": [6, 1]},
    "mechanisms": {"frozen_tiles": [[3, 1]], "freeze_steps": 5},
    "goal": {"type": "reach_position", "target": [6, 1]}, "max_steps": 30,
}


# --- backend hook ---------------------------------------------------------------


def test_state_backend_default_is_not_frozen():
    from gridworld.backends.base import AbstractGridBackend

    class Stub(AbstractGridBackend):
        def configure(self, task_spec): ...
        def reset(self, seed=None): ...
        def step(self, action): ...
        def render(self): ...
        def get_mission_text(self): return ""
        def get_state(self): return GridState(agent_position=(0, 0), agent_direction=0)

    assert Stub().freeze_remaining() == 0


def test_minigrid_backend_reports_the_freeze_countdown():
    pytest.importorskip("minigrid")
    from gridworld.actions import MiniGridActions as A
    from gridworld.backends.minigrid_backend import MiniGridBackend

    backend = MiniGridBackend(render_mode="rgb_array")
    assert backend.freeze_remaining() == 0
    backend.configure(TaskSpecification.from_dict(CORRIDOR_FROZEN))
    backend.reset(seed=0)
    backend.step(int(A.MOVE_FORWARD))
    assert backend.freeze_remaining() == 0
    backend.step(int(A.MOVE_FORWARD))  # onto (3,1)
    assert backend.freeze_remaining() == 5
    seen = []
    for _ in range(5):
        backend.step(int(A.TURN_LEFT))
        seen.append(backend.freeze_remaining())
    assert seen == [4, 3, 2, 1, 0]


# --- frost overlay (pure function) ----------------------------------------------


def _blank():
    return np.full((RES, RES, 3), 60, dtype=np.uint8)


def test_frost_is_absent_when_not_frozen():
    frame = _blank()
    assert np.array_equal(hud.draw_frost(frame, 0, 5), frame)


def test_frost_fills_the_frame_on_step_on_and_thins_toward_thaw():
    base = _blank().astype(int)
    coverage = []
    for remaining in (5, 4, 3, 2, 1):
        frosted = hud.draw_frost(_blank(), remaining, 5).astype(int)
        changed = np.any(frosted != base, axis=-1)
        coverage.append(changed.mean())
    assert coverage[0] > 0.95  # step-on: the whole screen
    assert all(a > b for a, b in zip(coverage, coverage[1:]))  # thins every action
    assert coverage[-1] > 0.05  # still visibly frosted on the last no-op


def test_frost_cracks_grow_as_it_thaws():
    # cracks are drawn in a distinct colour so we can count them growing
    full = hud.draw_frost(_blank(), 5, 5)
    late = hud.draw_frost(_blank(), 1, 5)
    assert hud.crack_px(late) > hud.crack_px(full)


def test_frost_lightens_progressively():
    tints = [hud.draw_frost(_blank(), r, 5).astype(int).mean() for r in (5, 3, 1)]
    assert tints[0] > tints[1] > tints[2] > 60  # paler tint fades toward the 60-grey base


# --- through the 3D backend ---------------------------------------------------------


def test_mujoco3d_frames_frost_and_thaw_with_the_countdown():
    pytest.importorskip("mujoco")
    pytest.importorskip("minigrid")
    from gridworld.actions import MiniGridActions as A
    from gridworld.backends import get_backend

    backend = get_backend("mujoco3d", camera="first_person", resolution=RES)
    try:
        backend.configure(TaskSpecification.from_dict(CORRIDOR_FROZEN))
        backend.reset(seed=0)
        backend.step(int(A.MOVE_FORWARD))
        on_tile, *_ = backend.step(int(A.MOVE_FORWARD))
        assert backend.freeze_remaining() == 5
        frames = [on_tile]
        for _ in range(5):
            f, *_ = backend.step(int(A.TURN_LEFT))
            frames.append(f)
        # the scene is static (turns are no-ops) so every difference is the frost thinning
        diffs = [np.any(a != b, axis=-1).mean() for a, b in zip(frames, frames[1:])]
        assert all(d > 0.01 for d in diffs)
        thawed = frames[-1]
        assert backend.freeze_remaining() == 0
        again, *_ = backend.step(int(A.TURN_LEFT))  # a real turn now: view changes
        assert not np.array_equal(thawed, again)
    finally:
        backend.close()


# --- 2D: the agent token ices over ------------------------------------------------


def test_2d_agent_tile_ices_over_and_thaws():
    pytest.importorskip("minigrid")
    from gridworld.actions import MiniGridActions as A
    from gridworld.backends.minigrid_backend import MiniGridBackend

    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(TaskSpecification.from_dict(CORRIDOR_FROZEN))
    backend.reset(seed=0)
    backend.step(int(A.MOVE_FORWARD))
    backend.step(int(A.MOVE_FORWARD))
    env = backend.env
    ts = env.tile_size
    cell = (slice(1 * ts, 2 * ts), slice(3 * ts, 4 * ts))
    tiles = [env.get_frame(highlight=False)[cell].astype(int)]
    for _ in range(5):
        backend.step(int(A.TURN_LEFT))
        tiles.append(env.get_frame(highlight=False)[cell].astype(int))
    # each no-op thins the ice and cracks it; the thawed tile has no cracks left
    diffs = [np.any(a != b, axis=-1).mean() for a, b in zip(tiles, tiles[1:])]
    assert all(d > 0.02 for d in diffs)
    crack = np.array([58, 82, 110])
    cracks = [int(np.all(t == crack, axis=-1).sum()) for t in tiles]
    assert cracks[0] == 0 and cracks[-2] > 0 and cracks[-1] == 0
    # the frozen tile itself is pale blue; the thawed agent tile still shows the red agent
    assert int(((tiles[-1][..., 0] > 200) & (tiles[-1][..., 1] < 60)).sum()) > 20


# --- ride-direction glyph while on a rotator ---------------------------------------


def _render(camera, state, rotators):
    from gridworld.render3d.renderer import SceneRenderer

    r = SceneRenderer(TaskSpecification.from_dict(TILES), camera=camera, resolution=RES)
    try:
        return r.render(state, {}, rotators=rotators)
    finally:
        r.close()


def _glyph_region(frame):
    top, left, bottom, right = hud.ride_box(frame.shape[1])
    return frame[top:bottom, left:right]


@pytest.mark.parametrize("camera", ["first_person", "chase"])
def test_ride_glyph_only_while_standing_on_a_rotator(camera):
    pytest.importorskip("mujoco")
    beside = _render(camera, GridState(agent_position=(3, 2), agent_direction=0), (0, 3))
    on_tile = _render(camera, GridState(agent_position=(4, 2), agent_direction=0), (0, 3))
    assert hud.ride_glyph_px(_glyph_region(beside)) == 0
    assert hud.ride_glyph_px(_glyph_region(on_tile)) > 20


def test_ride_glyph_points_relative_to_the_view():
    pytest.importorskip("mujoco")
    on = GridState(agent_position=(4, 2), agent_direction=0)  # facing E
    ahead = _render("first_person", on, (0, 3))   # arrow E = straight ahead
    right = _render("first_person", on, (1, 3))   # arrow S = to the right
    behind = _render("first_person", on, (2, 3))  # arrow W = behind
    assert hud.ride_glyph_px(_glyph_region(ahead)) > 20
    assert not np.array_equal(_glyph_region(ahead), _glyph_region(right))
    assert not np.array_equal(_glyph_region(right), _glyph_region(behind))
    # relative direction, not absolute: facing S with arrow S is also "ahead"
    facing_s = _render("first_person", GridState(agent_position=(4, 2), agent_direction=1), (1, 3))
    assert np.array_equal(_glyph_region(facing_s), _glyph_region(ahead))


def test_no_ride_glyph_in_north_up_views():
    pytest.importorskip("mujoco")
    on_tile = _render("top_down", GridState(agent_position=(4, 2), agent_direction=0), (0, 3))
    assert hud.ride_glyph_px(_glyph_region(on_tile)) == 0


def test_lifted_arrow_is_translucent_so_the_wedge_shows_through():
    pytest.importorskip("mujoco")
    import mujoco

    from gridworld.render3d.renderer import SceneRenderer

    r = SceneRenderer(TaskSpecification.from_dict(TILES), camera="top_down", resolution=64)
    try:
        arrow = mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_GEOM, "rotator:0:arrow0")
        r.render(GridState(agent_position=(4, 2), agent_direction=0), {}, rotators=(0, 3))
        assert 0.3 <= float(r.model.geom_rgba[arrow][3]) <= 0.7
        r.render(GridState(agent_position=(3, 2), agent_direction=0), {}, rotators=(0, 3))
        assert float(r.model.geom_rgba[arrow][3]) == 1.0
    finally:
        r.close()
