MIN_TASK_PREFIX = "Task: Solve the maze by reaching the goal."

TASK_PREFIX = "Task: You are the triangular agent trying to navigate this maze. You are facing the pointy end. Move to the green goal cell in the grid."

MECHANISM_LIST = (
    "The environment may contain:\n"
    "Keys, doors, switches, and gates."
)

MECHANISM_RULES = (
    "RULES (domain logic):\n"
    "  - PICKUP: pick up a key while standing in the same cell.\n"
    "  - DROP: put down the key you are carrying, leaving it in your current cell.\n"
    "    You carry one key at a time, so DROP before picking up a different key.\n"
    "  - Doors: face a locked door with the matching key in inventory and TOGGLE to open it, then\n"
    "    MOVE_FORWARD through the open door. MOVE_FORWARD alone does not open a locked door.\n"
    "  - Switches: TOGGLE while standing on them. "
    "    Linked gates are open if its linked switch is on, and closed if it is off.\n"
    "  - Gates: CLOSED gates block movement; OPEN gates do not. TOGGLE linked switches to control them.\n"
    "  - Closed doors you lack a key for block movement like walls until resolved.\n"
    "  - Use DONE only when you are standing on the goal cell."
)

ONE_SHOT_EXAMPLE = (
    "Example maze and solution (14x14 maze with a key, switch, and gate):\n"
    "{example_14x14_dense_kr_sg_kb_2_image}\n"
    "actions to solve: {example_solution_14x14_dense_kr_sg_kb_2}\n"
)

VALID_ACTIONS_TEMPLATE = "Valid actions: {actions_hint}."

INITIAL_MAZE_SECTION = "Initial maze (fixed for this episode):\n{maze_text}"

# `INITIAL_MAZE_SECTION` is used when the observation includes text.
