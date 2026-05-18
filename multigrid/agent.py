# multigrid/agent.py

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
from .objects.base import WorldObj
from .base import Tiling


class Action(IntEnum):
    """
    Discrete action space for MultiGrid.

    MultiGridBackend exposes MiniGrid's 7-action external interface and maps it
    onto this native enum internally.
    """
    # Movement
    FORWARD = 0       # Move in facing direction
    BACKWARD = 1      # Move opposite to facing direction

    # Rotation
    TURN_LEFT = 2     # Rotate facing counter-clockwise
    TURN_RIGHT = 3    # Rotate facing clockwise

    # Object interaction
    PICKUP = 4        # Pick up object in facing cell
    DROP = 5          # Drop held object in facing cell
    TOGGLE = 6        # Interact: unlock door (with key), activate switch
    PUSH = 7          # Push object in facing direction

    # No-op
    WAIT = 8


@dataclass
class AgentState:
    """Complete agent state."""
    cell_id: str                       # Current cell
    facing: int                        # Direction index (0 to num_directions-1)
    holding: Optional[WorldObj] = None # Picked up object

    def get_facing_direction(self, tiling: Tiling) -> str:
        """Get direction label agent is facing."""
        return tiling.directions[self.facing]
