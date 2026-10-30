# test_regression.py

"""
Regression tests for previously-fixed bugs in MultiGrid.

E.7.1: Hex odd-row neighbor symmetry
E.7.2: Triangle facing validity after movement
E.7.3: Triangle cross-hex neighbours share an edge
"""

import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multigrid.tilings import HexTiling, TriangleTiling
from multigrid.env import MultiGridEnv, Action


class TestRegression:
    """Regression tests for previously-identified edge-case bugs."""

    def test_hex_neighbor_at_odd_row(self):
        """Hex cells at odd rows have correct bidirectional neighbor links.

        Validates odd-r offset coordinate neighbor computation: every
        neighbor link must have a reverse link back to the original cell.
        """
        tiling = HexTiling()
        tiling.generate_graph(8, 8)

        # Pick all cells at odd rows
        odd_row_cells = [
            cid for cid, cell in tiling.cells.items() if cell.row % 2 == 1
        ]
        assert len(odd_row_cells) > 0, "Should have cells at odd rows"

        for cell_id in odd_row_cells:
            cell = tiling.cells[cell_id]
            for direction, neighbor_id in cell.neighbors.items():
                # Neighbor must exist in the tiling
                assert neighbor_id in tiling.cells, \
                    f"Neighbor {neighbor_id} of {cell_id} not in tiling"

                # Neighbor must have a reverse link back
                neighbor_cell = tiling.cells[neighbor_id]
                reverse_found = cell_id in neighbor_cell.neighbors.values()
                assert reverse_found, (
                    f"Cell {cell_id} links to {neighbor_id} via {direction}, "
                    f"but {neighbor_id} has no reverse link back"
                )

    def test_triangle_facing_after_move(self):
        """Agent facing remains valid after movement on triangle grid.

        Triangle tiling has 3 directions. Moving forward must not corrupt
        the facing index outside the valid range.
        """
        task = {
            "task_id": "test_tri_facing",
            "seed": 42,
            "scene": {
                "bounds": {"width": 1.0, "height": 1.0},
                "objects": [],
                "agent": {"position": {"x": 0.5, "y": 0.5}, "facing": 0},
            },
            "goal": {"predicate": "reach_position", "position": {"x": 0.9, "y": 0.9}},
            "limits": {"max_steps": 50},
            "tiling": {"type": "triangle", "grid_size": {"width": 5, "height": 5}},
        }

        env = MultiGridEnv(task, tiling="triangle")
        env.reset()

        num_directions = len(env.tiling.directions)

        # Execute a series of movements and turns
        actions = [
            Action.FORWARD,
            Action.TURN_RIGHT,
            Action.FORWARD,
            Action.TURN_LEFT,
            Action.FORWARD,
            Action.TURN_RIGHT,
            Action.TURN_RIGHT,
            Action.FORWARD,
        ]

        for i, action in enumerate(actions):
            env.step(action)
            facing = env.state.agent.facing
            assert 0 <= facing < num_directions, (
                f"After action {action.name} (step {i+1}), facing={facing} "
                f"is outside valid range [0, {num_directions})"
            )

    def test_triangle_cross_hex_neighbour_shares_the_outer_edge(self):
        """A triangle's edge2 neighbour is the triangle across its outer edge.

        Triangle i spans the hex centre and vertices i and i+1 (vertex 0 at
        the top, clockwise), as the renderer draws it; its edge2 neighbour must
        have the same two vertices. The map used to follow the hex direction
        list's order instead and linked triangles that do not touch.
        """
        tiling = TriangleTiling()
        tiling.generate_graph(6, 6)

        def outer_edge(cell):
            hx, hy = cell.tiling_coords["hex_center"]
            size = cell.tiling_coords["hex_size"]
            i = cell.tiling_coords["tri_idx"]
            angles = (math.pi / 2 - k * math.pi / 3 for k in (i, i + 1))
            return [(hx + size * math.cos(a), hy - size * math.sin(a)) for a in angles]

        links = 0
        for cell in tiling.cells.values():
            neighbour_id = cell.neighbors.get("edge2")
            if neighbour_id is None:
                continue
            links += 1
            ours, theirs = outer_edge(cell), outer_edge(tiling.cells[neighbour_id])
            for p in ours:
                assert min(math.dist(p, q) for q in theirs) < 1e-9, (cell.id, neighbour_id)
        assert links > 0
