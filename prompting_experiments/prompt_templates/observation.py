"""Observation and history prompt templates."""

RECENT_HISTORY_HEADER = "Recent history (last 3 steps, oldest first):"
RECENT_HISTORY_STEP = "  ({row}, {col}) facing {facing} -> {action} -> {feedback}"

IMAGE_HISTORY_ACTION = "Action: {action}\n\n"
IMAGE_ONLY_HISTORY_INTRO = (
    "Recent steps (oldest first). Each image is the maze view from which the "
    "following action was chosen; infer pose and environment state from the image.\n\n"
)
IMAGE_TEXT_HISTORY_INTRO = "Recent step views (oldest first):\n\n"

WORLD_SIZE_LINE = "The world is a {rows} by {cols} grid."
COORDINATE_EXPLANATION = (
    "Coordinates: JSON lists use ``[x, y]`` (east, south) from the **top-left** corner ``(1, 1)``;"
    " tuples in this text use ``(row, column)`` matching env state (row southward, column east)."
    " So ``x`` = column index, ``y`` = row index."
)
START_LINE = "The start is at {start}."
GOAL_LINE = "The goal is at {goal}."
WALLS_LINE = "The following cells are walls: {walls}."

KEY_LINE = "There is a {color} key at ({row},{col})."
DOOR_LINE = (
    "There is a {status} {requires_key} door at ({row},{col})."
    " It requires the {requires_key} key to open."
)
SWITCH_LINE = (
    "There is a {switch_type} switch at ({row},{col}) (currently {state})."
    " It controls: {controls}."
)
GATE_LINE = (
    "There is a gate ({gate_id}) at ({row},{col})."
    " It is currently {state} (initially {initial_state})."
)

CURRENT_SITUATION_HEADER = "Current situation (this step):"
CURRENT_GOAL_LINE = "The goal is at {goal}."
CURRENT_AGENT_LINE = "You are at {position} facing {facing}."
CURRENT_STEPS_LINE = "Environment steps used so far: {step_count} (max {max_steps} before timeout)."
CURRENT_INVENTORY_LINE = "Your inventory: {inventory}."
CURRENT_MAP_CONTENTS_HEADER = "Map contents as of this step (keys on the ground, doors, switches, gates):"
NO_MECHANISMS_LINE = "(No keys on the ground, doors, switches, or gates in the current state description.)"

CELL_OUT_OF_BOUNDS = "out of bounds"
CELL_WALL = "wall"
CELL_GOAL = "GOAL ({row},{col})"
CELL_KEY = "{key_color} key ({row},{col})"
CELL_DOOR = "{status} {requires_key} door ({row},{col})"
CELL_GATE = "{state} gate ({row},{col})"
CELL_SWITCH = "switch ({state}) ({row},{col})"
CELL_OPEN = "open ({row},{col})"
