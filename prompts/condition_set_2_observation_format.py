"""Condition set 2: observation format."""

from .condition_set_1_prompt import STANDARD_PROMPT

IMAGE_PLUS_TEXT_PROMPT = STANDARD_PROMPT

IMAGE_ONLY_PROMPT = """You are the red triangular agent solving a maze. Your mission is to navigate to the green square.

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
	"name": "Observation format",
	"comparisons": [
		"Image + text prompt",
		"Image only (no text)",
	],
	"decision": "Does text add meaningful signal?",
	"prompts": {
		"image_plus_text": IMAGE_PLUS_TEXT_PROMPT,
		"image_only": IMAGE_ONLY_PROMPT,
	},
}

PROMPTS = CONDITION_SET["prompts"]
