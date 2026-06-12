"""System prompt templates."""

TASK_PREFIX = "Task: move to the goal cell in the grid."

MECHANISM_LIST = (
    "The environment may contain:\n"
    "- Keys: pick them up to open doors of the matching color\n"
    "- Doors: blocked passages that require a matching key\n"
    "- Switches: step onto them to activate (hold) or TOGGLE while standing on them\n"
    "- Gates: blocked passages controlled by switches\n"
)

MECHANISM_RULES = (
    "RULES (domain logic):\n"
    "  - PICKUP: pick up a key from the adjacent cell you are facing. Keys block movement — you\n"
    "    cannot MOVE_FORWARD onto a key; stand beside it, face it, and PICKUP.\n"
    "  - Doors: face a locked door with the matching key in inventory and TOGGLE to open it, then\n"
    "    MOVE_FORWARD through the open door. MOVE_FORWARD alone does not open a locked door.\n"
    "  - Switches: MOVE_FORWARD onto the switch cell, then TOGGLE (toggle/one-shot types). Hold-type\n"
    "    switches activate automatically while you stand on them. Only switches are toggled. Linked\n"
    "    gates are open if at least one linked switch is on, and closed if all are off.\n"
    "  - Gates: you cannot TOGGLE a gate. CLOSED gates block movement; OPEN gates do not.\n"
    "  - Closed gates and doors you lack a key for block movement like walls until resolved.\n"
    "  - Use DONE only when you are standing on the goal cell."
)

VALID_ACTIONS_TEMPLATE = "Valid actions: {actions_hint}."

FINAL_OUTPUT_INSTRUCTION = (
    "Do not explain, reason, summarize the map, or include any text before the answer.\n"
    "On the last line, output exactly:\n"
    "FINAL_OUTPUT: <action>  or  FINAL_OUTPUT: <a>, <b>, ...  "
    "(comma-separated; one or more valid actions)"
)

INITIAL_MAZE_SECTION = "Initial maze (fixed for this episode):\n{maze_text}"
