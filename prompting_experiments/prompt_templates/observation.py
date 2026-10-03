"""Observation and history prompt templates."""

RECENT_HISTORY_HEADER = "Recent history (last {n} steps, oldest first):"
RECENT_HISTORY_STEP = (
    "Position before: ({row}, {col}), facing {facing}\n"
    "FINAL_OUTPUT: {action}\n"
    "Feedback: {feedback}"
)
RECENT_HISTORY_SNAPSHOT_STEP = (
    "{snapshot}\n"
    "FINAL_OUTPUT: {action}\n"
    "Feedback: {feedback}"
)

TEXT_SUMMARY_START = "You started at ({row}, {col}) facing {facing}."
TEXT_SUMMARY_BLOCK_HEADER = "Activity summary:"
TEXT_SUMMARY_FIRST_EVENT = "first you {event}"
TEXT_SUMMARY_THEN_EVENT = "then you {event}"
TEXT_SUMMARY_FINAL_EVENT = "finally you {event}"
TEXT_SUMMARY_PICKUP_KEY = "picked up the {key_color} key"
TEXT_SUMMARY_DROP_KEY = "dropped the {key_color} key"
TEXT_SUMMARY_DROP_KEY_UNKNOWN = "dropped the key"
TEXT_SUMMARY_OPEN_DOOR = "opened the {door_color} door"
TEXT_SUMMARY_OPEN_GATE = "opened the {gate_color} gate"
TEXT_SUMMARY_CLOSE_GATE = "closed the {gate_color} gate"
TEXT_SUMMARY_DEATH_PORTAL = "stepped on a death portal at ({row},{col}) and lost your progress"
TEXT_SUMMARY_FROZEN = "stepped on a frozen tile at ({row},{col})"
TEXT_SUMMARY_CARRIED = "stepped on a rotating tile at ({row},{col}) and rode it in {direction}"
TEXT_SUMMARY_TELEPORTED = "teleported to ({row},{col})"
TEXT_SUMMARY_NAV_TO = "navigated to ({row}, {col})"
TEXT_SUMMARY_PASSED = "passed ({row}, {col})"
TEXT_SUMMARY_EMPTY = "you haven't done anything yet"

WORLD_SIZE_LINE = "The world is a {rows} by {cols} grid."
COORDINATE_EXPLANATION = (
    "Coordinates are ``(row, column)`` from the **top-left** corner ``(1, 1)``:"
    " the row number increases going **south** (down); the column number increases"
    " going **east** (right). To reach a cell with a larger row number, go south;"
    " with a larger column number, go east."
)
START_LINE = "The start is at {start}."
GOAL_LINE = "The goal is at {goal}."
WALLS_LINE = "The following cells are walls: {walls}."

KEY_LINE = "There is a {color} key at ({row},{col})."
DOOR_LINE = "There is a {status} {requires_key} door at ({row},{col})."
SWITCH_LINE = (
    "There is a {switch_type} switch ({switch_id}) at ({row},{col}) (currently {state})."
)
GATE_LINE = (
    "There is a gate ({gate_id}) at ({row},{col})."
    " It is currently {state} (initially {initial_state})."
)
DEATH_PORTAL_LINE = "There is a death portal at ({row},{col})."
FROZEN_TILE_LINE = "There is a frozen tile at ({row},{col})."
FROZEN_STATUS_LINE = "You are frozen, wait for freeze to end."
ROTATING_TILE_LINE = "There is a rotating tile at ({row},{col}) pointing {direction}."
TELEPORTER_LINE = "There is a {color} teleporter between ({row_a},{col_a}) and ({row_b},{col_b})."
TELEPORTER_ONE_WAY_LINE = "There is a {color} teleporter from ({row_a},{col_a}) to ({row_b},{col_b})."

CURRENT_AGENT_LINE = "You are at {position} facing {facing}."
CURRENT_AGENT_POSITION_LINE = "You are at {position}."
CURRENT_INVENTORY_LINE = "Your inventory: {inventory}."
MOVES_REMAINING_LINE = "Moves remaining: {n}."
STALL_REMAINING_LINE = (
    "Moves remaining until stall: {n}. "
    "If you make no progress and there is no state change in the environment "
    "for that many steps, the episode will be terminated due to a stall."
)
CURRENT_KEYS_USED_UP_LINE = "Keys used up: {keys}."
CURRENT_MAP_CONTENTS_HEADER = "Map contents as of this step (keys on the ground, doors, switches, gates):"
NO_MECHANISMS_LINE = "(No keys on the ground, doors, switches, or gates in the current state description.)"

CELL_OUT_OF_BOUNDS = "out of bounds"
CELL_WALL = "wall"
CELL_GOAL = "GOAL ({row},{col})"
CELL_KEY = "{key_color} key ({row},{col})"
CELL_DOOR = "{status} {requires_key} door ({row},{col})"
CELL_GATE = "{state} gate ({row},{col})"
CELL_SWITCH = "switch ({state}) ({row},{col})"
CELL_DEATH_PORTAL = "death portal ({row},{col})"
CELL_FROZEN_TILE = "frozen tile ({row},{col})"
CELL_ROTATING_TILE = "rotating tile pointing {direction} ({row},{col})"
CELL_TELEPORTER = "{color} teleporter ({row},{col})"
CELL_OPEN = "open ({row},{col})"
