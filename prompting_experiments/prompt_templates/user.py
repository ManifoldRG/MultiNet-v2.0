"""User prompt templates."""

CURRENT_IMAGE_PLACEHOLDER = "{current_image}"
NEXT_ACTION_QUESTION = "What is your next action?"

ONE_SHOT_EXAMPLE_INTRO = (
    "Example maze and solution "
    "(14x14 maze with a red key-door, switch-gate, and blue key-door chain):\n"
)
ONE_SHOT_SOLUTION_LINE = "Actions to solve: {actions}"

LAST3_USER_PROMPT = {
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
