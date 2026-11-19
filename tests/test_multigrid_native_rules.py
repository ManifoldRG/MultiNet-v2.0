"""MultiGridBackend under multigrid's native rules (``execute_action``).

Salvaged from PR #82: the native rules have no death, frozen or rotating
tiles, so the backend refuses a spec that has them instead of dropping them;
portals keep their spec colour; and the legacy square renderer's agent arrow
points where MOVE_FORWARD goes.
"""

import numpy as np
import pytest

from gridworld.actions import MiniGridActions
from gridworld.backends.multigrid_backend import MultiGridBackend
from gridworld.task_spec import TaskSpecification
from multigrid.rendering import COLORS


def _spec(**mechanisms) -> TaskSpecification:
    return TaskSpecification.from_dict({
        "task_id": "multigrid_native",
        "seed": 1,
        "difficulty_tier": 1,
        "maze": {"dimensions": [7, 7], "walls": [], "start": [3, 3], "goal": [5, 5]},
        "mechanisms": mechanisms,
        "goal": {"type": "reach_position", "target": [5, 5]},
        "max_steps": 40,
    })


@pytest.mark.parametrize(
    "mechanisms, refused",
    [
        ({"death_portals": [[2, 2]]}, "death_portals"),
        ({"frozen_tiles": [[2, 2]], "freeze_steps": 3}, "frozen_tiles"),
        ({"rotating_tiles": [[2, 2]], "rotating_initial_directions": [1]}, "rotating_tiles"),
    ],
)
@pytest.mark.parametrize("tiling", ["square", "hex", "triangle"])
def test_tiles_the_native_rules_lack_are_refused_not_dropped(tiling, mechanisms, refused):
    backend = MultiGridBackend(tiling=tiling)
    with pytest.raises(ValueError, match=refused):
        backend.configure(_spec(**mechanisms))
    assert not backend.is_configured and backend.env is None


def test_every_refused_tile_is_named():
    spec = _spec(
        death_portals=[[1, 1]], frozen_tiles=[[2, 2]],
        rotating_tiles=[[4, 4]], rotating_initial_directions=[0],
    )
    with pytest.raises(ValueError, match="death_portals, frozen_tiles, rotating_tiles"):
        MultiGridBackend().configure(spec)


def test_native_mechanisms_still_configure():
    backend = MultiGridBackend(tiling="square")
    backend.configure(_spec(
        keys=[{"id": "k1", "position": [2, 1], "color": "red"}],
        doors=[{"id": "d1", "position": [4, 1], "requires_key": "red"}],
        switches=[{"id": "s1", "position": [1, 2], "controls": ["g1"]}],
        gates=[{"id": "g1", "position": [1, 4]}],
        blocks=[{"id": "b1", "position": [4, 4], "color": "grey"}],
        hazards=[{"id": "h1", "position": [2, 5], "hazard_type": "lava"}],
        teleporters=[{"id": "tp1", "position_a": [5, 1], "position_b": [1, 5]}],
        freeze_steps=7,  # a parameter, not a tile: nothing to refuse
    ))
    assert backend.is_configured


def test_portals_keep_their_spec_colour():
    backend = MultiGridBackend(tiling="square", render_mode="rgb_array")
    backend.configure(_spec(teleporters=[
        {"id": "tp1", "position_a": [1, 1], "position_b": [5, 1], "color": "cyan"},
        {"id": "tp2", "position_a": [1, 5], "position_b": [5, 3]},  # spec default
    ]))
    frame, _state, _info = backend.reset(seed=1)
    objects = backend.env.state.objects
    assert objects["tp1_a"].color == objects["tp1_b"].color == "cyan"
    assert objects["tp2_a"].color == objects["tp2_b"].color == "purple"
    pixels = {tuple(p) for p in frame.reshape(-1, 3)}
    assert COLORS["cyan"] in pixels and COLORS["purple"] in pixels


def _agent_centroid(frame: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(np.all(frame == np.array(COLORS["agent"], dtype=frame.dtype), axis=-1))
    assert len(xs), "no agent pixels in the frame"
    return np.array([xs.mean(), ys.mean()])


def test_square_agent_arrow_points_where_forward_moves():
    backend = MultiGridBackend(tiling="square", render_mode="rgb_array")
    backend.configure(_spec())
    n_dirs = len(backend.env.tiling.directions)
    centroids, moves = [], []
    for facing in range(n_dirs):
        backend.reset(seed=1)
        backend.env.state.agent.facing = facing
        centroids.append(_agent_centroid(backend.render()))
        start = np.array(backend.get_state().agent_position)
        *_, state, _info = backend.step(int(MiniGridActions.MOVE_FORWARD))
        moves.append(np.array(state.agent_position) - start)
    # The four arrows share a cell, so their mean is its centre; each arrow
    # leans towards its tip, along the step forward takes ((x, y), y down).
    centre = np.mean(centroids, axis=0)
    for facing, (centroid, move) in enumerate(zip(centroids, moves)):
        assert np.abs(move).sum() == 1, (facing, move)
        lean = centroid - centre
        assert np.dot(lean, move) > 0.8 * np.linalg.norm(lean), (
            f"facing {backend.env.tiling.directions[facing]}: arrow leans {lean}, forward moves {move}"
        )
