"""User prompt templates."""

CURRENT_IMAGE_PLACEHOLDER = "{current_image}"
NEXT_ACTION_QUESTION = "What is your next action?"

ONE_SHOT_EXAMPLE_INTRO = (
    "Example maze and solution "
    "(14x14 maze with a red key-door, switch-gate, and blue key-door chain):\n"
)
ONE_SHOT_SOLUTION_LINE = "Actions to solve: {actions}"

LAST_N_USER_PROMPT = {
    "header": "Recent steps (oldest first):\n",
    "image_text_step": "Your inventory: {inventory}.\nFINAL_OUTPUT: {action}\n",
    "image_only_step": "Your inventory: {inventory}.\nFINAL_OUTPUT: {action}\n",
}

STANDARD_IMAGE_ONLY_USER_PROMPT = ( # the standard prompt
    f"{CURRENT_IMAGE_PLACEHOLDER}\n"
    "Your inventory: {inventory}.\n"
    f"{NEXT_ACTION_QUESTION}"
)

TEXT_ONLY_USER_PROMPT = (
    "{current_observation_text}"
    f"{NEXT_ACTION_QUESTION}"
)

IMAGE_TEXT_USER_PROMPT = (
    f"{CURRENT_IMAGE_PLACEHOLDER}\n"
    "{current_observation_text}"
    f"{NEXT_ACTION_QUESTION}"
)

# image_text is the only observation mode that triggered Qwen to ramble spatial
# reasoning and never emit FINAL_OUTPUT (parse failures). This reminder is
# appended last, after the standard FINAL_OUTPUT instruction, so it is the final
# thing the model reads.
IMAGE_TEXT_ACTION_FORMAT_REMINDER = (
    "Reminder: decide your single next action without narrating your reasoning. "
    "Your response must end with the line `FINAL_OUTPUT: <action>` and nothing "
    "after it."
)

# --- 3D start map + current-view label (run-config render.start_map) ---------
# NOTE: the 3D visuals are about to be redesigned. Every word describing how
# something LOOKS in a 3D frame lives in this block (plus the two 3D task
# prefixes in system.py, which name the agent and goal), so a redesign is a
# single edit here. Appearances checked against top-down renders of the M10
# all-mechanism maze and an R1-panel maze (2026-10-02).
#
# The map block opens the FIRST user message of every request (episode_step).
START_MAP_MINIMAL = (
    "Map: a top-down view of the whole maze at the start, north up. "
    "It is a snapshot and does not change as you move."
)
START_MAP_STANDARD = (
    "Map: a top-down view of the whole maze at the start, north up. "
    "The red wedge is where you started, pointing the way you faced. "
    "It is a snapshot and does not change as you move."
)
START_MAP_LEGEND = (
    "The map may contain:\n"
    "  - Dark blocks are walls; light gray squares are floor.\n"
    "  - The red wedge is you, at your start.\n"
    "  - The green square is the goal.\n"
    "  - A small key shape is a key.\n"
    "  - A solid bar in a key's color across a passage is a closed door; "
    "two small posts in that color are an open door.\n"
    "  - A dark square holding a round button is a switch; "
    "the button is dim when off and bright when on.\n"
    "  - A thin barred line across a passage is a closed gate; "
    "three small dots in its place are an open gate.\n"
    "  - A colored ring with a dark center is a portal; portals come in pairs of one color, "
    "and a sign above each gives the (row, col) it leads to.\n"
    "  - A dark red disc with a white skull is a death tile.\n"
    "  - A pale blue-white mound is an ice tile.\n"
    "  - An orange disc with a black arrow is a turntable; "
    "the arrow shows the direction it carries you.\n"
    "  - The compass in the top-right corner shows north at the top."
)
START_MAP_VERBOSE = f"{START_MAP_STANDARD}\n{START_MAP_LEGEND}"

# standard/verbose with a start map: a line right before the current image.
CURRENT_VIEW_LABEL = "Your view now ({view}):"
CURRENT_VIEW_NAMES = {
    "first_person": "first person",
    "first_person_narrow": "first person",
}
# verbose, on the views that turn with the agent (render3d/hud.py).
COMPASS_NOTE = (
    "The compass in the top-right corner of your view turns with you: "
    "the yellow letter at the top is the way you face, and the red needle points north."
)
