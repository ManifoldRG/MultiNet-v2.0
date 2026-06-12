"""User prompt templates."""

OBSERVATION_SECTION = "Observation:\n{obs_text}\n\n"

ONE_SHOT_EXAMPLE = (
    "Example maze and solution (14x14 maze with a key, switch, and gate):\n"
    "{example_14x14_dense_kr_sg_kb_2_image}\n"
    "actions to solve: {example_solution_14x14_dense_kr_sg_kb_2}\n"
)

STANDARD_USER_PROMPT = (
    "{obs_block}"
    "{status_block}"
    "What is your next action?"
)

VERBOSE_USER_PROMPT = (
    "{obs_block}"
    "{mechanism_block}"
    "{status_block}"
    "What is your next action?"
)

STATUS_BLOCK = (
    "Position: {position}  |  Facing: {facing}  |  Goal: {goal}\n"
    "Last result: {last_feedback}\n"
)

MECHANISM_HINTS_HEADER = "Hints:\n"
KEY_DOOR_HINT = (
    "  - Face an adjacent key and PICKUP (do not walk onto the key). "
    "Face a locked door with the matching key and TOGGLE to open it, then MOVE_FORWARD through."
)
SWITCH_GATE_HINT = (
    "  - MOVE_FORWARD onto a switch, then TOGGLE (hold switches activate on step). "
    "Gates cannot be toggled directly — activate their linked switch(es)."
)
