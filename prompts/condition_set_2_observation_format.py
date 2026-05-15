"""Condition set 2: observation format."""

IMAGE_PLUS_TEXT_PROMPT = """You are the red triangular agent solving a maze. Your mission is to navigate to the green square.

mechanisms present:
{MAZE_SPECIFIC_MECHS}

Available actions:
{DOMAIN_SPECIFIC_ACTION_LIST}

Current observation:
{DOMAIN_SPECIFIC_OBSERVATION}

Observation description:
{OBSERVATION_TEXT_DESCRIPTION}

Inventory:
{INVENTORY}

Choose exactly ONE action to take from the available actions.

Output only the action name.
"""

IMAGE_PROMPT = """You are the red triangular agent solving a maze. Your mission is to navigate to the green square.

mechanisms present:
{MAZE_SPECIFIC_MECHS}

Available actions:
{DOMAIN_SPECIFIC_ACTION_LIST}

Current observation:
{DOMAIN_SPECIFIC_OBSERVATION}

Inventory:
{INVENTORY}

Choose exactly ONE action to take from the available actions.

Output only the action name.
"""

CONDITION_SET = {
	"id": "condition_set_2",
	"name": "Observation format",
	"comparisons": [
		"Image + text prompt",
		"Image only (no text)",
	],
	"decision": "Does text add meaningful signal?",
	"prompts": {
		"image_plus_text": IMAGE_PLUS_TEXT_PROMPT,
		"image": IMAGE_PROMPT,
	},
}
