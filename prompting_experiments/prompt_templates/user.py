"""User prompt templates."""

OBSERVATION_SECTION = "Observation:\n{obs_text}\n\n"

MINIMAL_USER_PROMPT = (
    "{history_block}"
    "{obs_block}"
    "Position: {position}  |  Facing: {facing}  |  Goal: {goal}  |  "
    "Step {step_num}/{max_steps}\n"
    "Last result: {last_feedback}\n"
    "What is your next action?"
)

VERBOSE_USER_PROMPT = (
    "{history_block}"
    "{obs_block}"
    "Position: {position}  |  Facing: {facing}  |  Goal: {goal}  |  "
    "Manhattan: {manhattan}  |  Step {step_num}/{max_steps} ({steps_left} left)\n"
    "Inventory: {inventory}\n"
    "{budget_warn}"
    "{neighbour_block}"
    "{mechanism_block}"
    "Last result: {last_feedback}\n"
    "What is your next action?"
)

BUDGET_WARNING = "  WARNING: Only {steps_left} steps remaining!\n"
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
