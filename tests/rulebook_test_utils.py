"""Shared fixtures for the R1-rulebook tests on square and hex tilings."""

from gridworld.task_spec import TaskSpecification


def every_mechanic(tiling: str, **overrides) -> TaskSpecification:
    """A 9x9 serpentine whose only route (on either tiling: whole wall rows
    split it) takes the red key, opens the red door, crosses the ice, toggles
    the switch, passes the gate, rides the turntable and takes the portal over
    a death cell. A second death cell sits in a pocket next to the start.

        row 1  S . k . . . .       S start, k key
        row 2  # x # # # # D       x death pocket, D red door
        row 3  s . . * . . .       s switch, * ice
        row 4  G # # # # # #       G gate
        row 5  . . @ . . . .       @ turntable
        row 6  # # # # # # .
        row 7  g p . x . P .       g goal, p/P portal ends
    """
    d = {
        "task_id": f"every_mechanic_{tiling}",
        "seed": 3,
        "difficulty_tier": 1,
        "maze": {
            "dimensions": [9, 9],
            "walls": [[1, 2], [3, 2], [4, 2], [5, 2], [6, 2]]
            + [[x, 4] for x in range(2, 8)]
            + [[x, 6] for x in range(1, 7)],
            "start": [1, 1],
            "goal": [1, 7],
            "tiling": tiling,
        },
        "mechanisms": {
            "keys": [{"id": "kR", "position": [3, 1], "color": "red"}],
            "doors": [{"id": "DR", "position": [7, 2], "requires_key": "red", "initial_state": "locked"}],
            "switches": [{"id": "s1", "position": [1, 3], "controls": ["g1"], "color": "yellow"}],
            "gates": [{"id": "g1", "position": [1, 4], "color": "black"}],
            "teleporters": [{
                "id": "portal_cyan", "position_a": [6, 7], "position_b": [2, 7],
                "color": "cyan", "bidirectional": True,
            }],
            "death_portals": [[2, 2], [4, 7]],
            "frozen_tiles": [[4, 3]],
            "freeze_steps": 3,
            "rotating_tiles": [[3, 5]],
            "rotating_initial_directions": [2],
        },
        "goal": {"type": "reach_position", "target": [1, 7]},
        "max_steps": 400,
    }
    d.update(overrides)
    return TaskSpecification.from_dict(d)
