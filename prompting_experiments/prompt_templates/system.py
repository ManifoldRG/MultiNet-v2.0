
TASK_PREFIX = "Task: You are the triangular agent trying to navigate this maze. You are facing the pointy end. Move to the green goal cell in the grid."

MECHANISM_LIST = (
    "The environment may contain:\n"
    "- Keys: pick them up to open doors of the matching color\n"
    "- Doors: blocked passages that require a matching key\n"
    "- Switches: TOGGLE while standing on them\n"
    "- Gates: blocked passages controlled by switches\n"
)

MECHANISM_RULES = (
    "RULES (domain logic):\n"
    "  - PICKUP: pick up a key from the adjacent cell you are facing. Keys block movement — you\n"
    "    cannot MOVE_FORWARD onto a key; stand beside it, face it, and PICKUP.\n"
    "  - Doors: face a locked door with the matching key in inventory and TOGGLE to open it, then\n"
    "    MOVE_FORWARD through the open door. MOVE_FORWARD alone does not open a locked door.\n"
    "  - Switches: face the switch, then TOGGLE. "
    "    Linked gates are open if its linked switch is on, and closed if it is off.\n"
    "  - Gates: CLOSED gates block movement; OPEN gates do not. TOGGLE linked switches to control them.\n"
    "  - Closed doors you lack a key for block movement like walls until resolved.\n"
    "  - Use DONE only when you are standing on the goal cell."
)

VALID_ACTIONS_TEMPLATE = "Valid actions: {actions_hint}."

INITIAL_MAZE_SECTION = "Initial maze (fixed for this episode):\n{maze_text}"
