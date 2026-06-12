"""User prompt templates."""

OBSERVATION_SECTION = "Observation:\n{obs_text}\n\n"

MINIMAL_USER_PROMPT = (
    "{obs_block}"
    "Position: {position}  |  Facing: {facing}  |  Goal: {goal}\n"
    "Last result: {last_feedback}\n"
    "What is your next action?\n"
    "Reply exactly as one line: FINAL_OUTPUT: <one valid action>"
)

VERBOSE_USER_PROMPT = (
    "{obs_block}"
    "Position: {position}  |  Facing: {facing}  |  Goal: {goal}  |  "
    "Manhattan: {manhattan}\n"
    "Inventory: {inventory}\n"
    "{neighbour_block}"
    "{mechanism_block}"
    "Last result: {last_feedback}\n"
    "What is your next action?\n"
    "Reply exactly as one line: FINAL_OUTPUT: <one valid action>"
)

NEIGHBOUR_BLOCK_HEADER = "From your perspective:\n"
NEIGHBOUR_LINE = "  {relative_direction}: {description}"

MECHANISM_HINTS_HEADER = "Hints:\n"
KEY_DOOR_HINT = (
    "  - Face an adjacent key and PICKUP (do not walk onto the key). "
    "Face a locked door with the matching key and TOGGLE to open it, then MOVE_FORWARD through."
)
SWITCH_GATE_HINT = (
    "  - MOVE_FORWARD onto a switch, then TOGGLE (hold switches activate on step). "
    "Gates cannot be toggled — activate their linked switch(es)."
)
