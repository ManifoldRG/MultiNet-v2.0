"""gridworld.render2d: the MiniGrid look on square and hex cells.

Square frames must be MiniGrid's pixel for pixel (MultiGrid-square frames are
also compared with MiniGrid's step by step in test_multigrid_rulebook.py). Hex
frames must use the same palette and icons, with the agent and the turntable
arrows pointing along the six hex directions.
"""

import math
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from gridworld.actions import MiniGridActions as A
from gridworld.backends import MiniGridBackend
from gridworld.backends.multigrid_backend import MultiGridBackend
from gridworld.render2d import AGENT, ARROW, HEX_SIZE, TilingFrameRenderer, hex_frame_shape
from gridworld.task_spec import TaskSpecification
from gridworld.world_model import TaskPlanningContext, apply
from rulebook_test_utils import every_mechanic


def _backend(spec):
    backend = MultiGridBackend()
    backend.configure(spec)
    backend.reset(seed=spec.seed)
    return backend


def _cell_pixels(renderer: TilingFrameRenderer, xy) -> tuple[np.ndarray, np.ndarray]:
    """(rows, cols) of the full-scale pixels inside cell ``xy``."""
    k = xy[1] * renderer.width + xy[0]
    return np.nonzero(renderer._full.cell_index == k)


def _centre(renderer: TilingFrameRenderer, xy) -> tuple[float, float]:
    rows, cols = _cell_pixels(renderer, xy)
    return cols.mean(), rows.mean()


def _pointing(frame, renderer, xy, colour) -> tuple[int, float]:
    """Which hex direction a triangle of ``colour`` in cell ``xy`` points, and
    by what margin: how far its pixels reach from their centroid along each
    of the six directions, the best reach over the runner-up. A nearly
    equilateral triangle (MiniGrid's turntable arrow) reaches almost as far
    toward a base corner as toward its tip, so 60-degree turns of it are
    ambiguous; the margin catches that."""
    rows, cols = _cell_pixels(renderer, xy)
    hit = (frame[rows, cols] == colour).all(axis=-1)
    ys, xs = rows[hit] - rows[hit].mean(), cols[hit] - cols[hit].mean()
    reach = [
        (xs * math.cos(k * math.pi / 3) + ys * math.sin(k * math.pi / 3)).max() for k in range(6)
    ]
    best = int(np.argmax(reach))
    return best, reach[best] / sorted(reach)[-2]


def test_square_frames_are_minigrids():
    spec = every_mechanic("square")
    minigrid = MiniGridBackend()
    minigrid.configure(spec)
    frame, _state, _ = minigrid.reset(seed=spec.seed)
    renderer = TilingFrameRenderer(spec)
    ctx = TaskPlanningContext(spec)
    assert renderer.frame_shape == frame.shape
    assert np.array_equal(renderer.render(ctx.initial_state()), frame)


def test_hex_frame_size_keeps_minigrids_pixels_per_cell():
    assert hex_frame_shape(18, 18) == (550, 641)
    # A cell covers about MiniGrid's 32 x 32 px.
    assert 3 * math.sqrt(3) / 2 * HEX_SIZE ** 2 == pytest.approx(32 * 32, rel=0.02)
    spec = every_mechanic("hex")
    frame, *_ = _backend(spec).reset(seed=spec.seed)
    assert frame.shape == (*hex_frame_shape(9, 9), 3) == (280, 330, 3)


def test_hex_frames_use_minigrids_palette():
    """Antialiased hex edges add blends, but every colour that covers a real
    area of the hex frame is one MiniGrid draws for the same maze, and the
    other way round."""
    minigrid = MiniGridBackend()
    minigrid.configure(every_mechanic("square"))
    square, *_ = minigrid.reset(seed=3)
    hexed, *_ = _backend(every_mechanic("hex")).reset(seed=3)

    def dominant(frame):
        counts = Counter(map(tuple, frame.reshape(-1, 3).tolist()))
        return {c for c, n in counts.items() if n >= 40 and len(set(c)) > 1}  # skip grey blends

    assert dominant(hexed) <= set(map(tuple, square.reshape(-1, 3).tolist()))
    assert dominant(square) <= set(map(tuple, hexed.reshape(-1, 3).tolist()))


def test_hex_agent_triangle_points_along_its_facing():
    spec = every_mechanic("hex")
    backend = _backend(spec)
    renderer = backend._frame_renderer
    frame = backend.render()
    for direction in range(6):
        pointing, margin = _pointing(frame, renderer, (1, 1), AGENT)
        assert pointing == direction and margin > 1.3, (direction, pointing, margin)
        frame, *_ = backend.step(A.TURN_RIGHT)


def test_hex_turntable_arrow_points_along_its_direction():
    spec = every_mechanic("hex")
    renderer = TilingFrameRenderer(spec)
    ctx = TaskPlanningContext(spec)
    state = ctx.initial_state()
    for _ in range(6):
        frame = renderer.render(state)
        direction = state.rotator_dirs[0]
        pointing, margin = _pointing(frame, renderer, (3, 5), ARROW)
        assert pointing == direction and margin > 1.3, (direction, pointing, margin)
        state = apply(ctx, state, A.TURN_LEFT)


def test_hex_frame_follows_the_state():
    spec = every_mechanic("hex")
    backend = _backend(spec)
    renderer = backend._frame_renderer

    def cell(frame, xy):
        rows, cols = _cell_pixels(renderer, xy)
        return frame[rows, cols].astype(int)

    ctx = backend._planning_ctx
    start = ctx.initial_state()
    red = np.array([255, 0, 0])

    # The red key lies at (3, 1) until it is picked up.
    with_key = renderer.render(start)
    without_key = renderer.render(replace(start, key_positions=frozenset(), carrying_key="kR"))
    assert (cell(with_key, (3, 1)) == red).all(axis=-1).sum() > 50
    assert not (cell(without_key, (3, 1)) == red).all(axis=-1).any()

    # A locked door is a filled red hex; an open one is floor with a bar.
    opened = renderer.render(replace(start, open_doors=frozenset({"DR"})))
    assert cell(opened, (7, 2)).mean() < cell(with_key, (7, 2)).mean() / 2

    # The switch's core turns white when it is on.
    on = renderer.render(replace(start, active_switches=frozenset({"s1"})))
    cx, cy = _centre(renderer, (1, 3))
    assert tuple(with_key[round(cy), round(cx)]) == (30, 30, 30)
    assert tuple(on[round(cy), round(cx)]) == (255, 255, 255)


def test_hex_frost_overlay_thins_as_the_ice_thaws():
    spec = every_mechanic("hex")
    renderer = TilingFrameRenderer(spec)
    ctx = TaskPlanningContext(spec)
    on_ice = replace(ctx.initial_state(), agent_pos=(4, 3))
    rows, cols = _cell_pixels(renderer, (4, 3))
    ice = np.array([214, 232, 246])

    def frost(remaining):
        frame = renderer.render(replace(on_ice, freeze_remaining=remaining))
        return np.abs(frame[rows, cols].astype(int) - ice).mean()

    assert frost(3) < frost(1) < frost(0)


def test_hex_partial_observability_dims_cells_out_of_view():
    spec = TaskSpecification.from_dict({
        "task_id": "hex_view", "seed": 0, "difficulty_tier": 1,
        "maze": {"dimensions": [9, 9], "walls": [], "start": [2, 2], "goal": [6, 6], "tiling": "hex"},
        "rules": {"observability": "view_cone", "view_size": 5},
        "goal": {"type": "reach_position", "target": [6, 6]},
        "max_steps": 20,
    })
    backend = MultiGridBackend()
    backend.configure(spec)
    frame, state, _ = backend.reset(seed=0)
    renderer = backend._frame_renderer
    assert state.visible_cells and (6, 6) not in state.visible_cells
    rows, cols = _cell_pixels(renderer, (6, 6))
    assert (frame[rows, cols] == 30).all()  # never seen
    rows, cols = _cell_pixels(renderer, (2, 2))
    assert (frame[rows, cols] == AGENT).all(axis=-1).any()
