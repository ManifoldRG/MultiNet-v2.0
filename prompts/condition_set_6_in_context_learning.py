"""Condition set 6: in-context learning."""

from .condition_set_1_prompt import STANDARD_PROMPT

ONE_SHOT_PROMPT = """Example maze interaction:

mechanisms present:
{ICL_MAZE_SPECIFIC_MECHS}

Available actions:
{DOMAIN_SPECIFIC_ACTION_LIST}

Observation:
{ICL_OBSERVATION_1}

Inventory:
{ICL_INVENTORY_1}

Action:
{ICL_ACTION_1}

Observation:
{ICL_OBSERVATION_2}

Inventory:
{ICL_INVENTORY_2}

Action:
{ICL_ACTION_2}

Observation:
{ICL_OBSERVATION_3}

Inventory:
{ICL_INVENTORY_3}

Action:
{ICL_ACTION_3}

End of example.

Now solve the following maze.

You are the red triangular agent solving a maze. Your mission is to navigate to the green square.

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
	"name": "In-context learning",
	"comparisons": [
		"Zero-shot: no examples",
		"1-shot: one example trajectory (different maze, same mechanism type)",
	],
	"decision": (
		"If 1-shot dramatically improves performance, bottleneck is task "
		"understanding, not capability. Determines zero-shot vs few-shot."
	),
	"constraint": "ICL examples must not use evaluation mazes.",
	"prompts": {
		"zero_shot": STANDARD_PROMPT,
		"one_shot": ONE_SHOT_PROMPT,
	},
}

PROMPTS = CONDITION_SET["prompts"]
