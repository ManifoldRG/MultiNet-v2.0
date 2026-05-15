"""Condition set 3: context window."""

HISTORY_PROMPT = """You are the red triangular agent solving a maze. Your mission is to navigate to the green square.

mechanisms present:
{MAZE_SPECIFIC_MECHS}

Available actions:
{DOMAIN_SPECIFIC_ACTION_LIST}

Observation history:

Frame t-2:
{FRAME_T_MINUS_2}

Frame t-1:
{FRAME_T_MINUS_1}

Current frame:
{CURRENT_FRAME}

Inventory:
{INVENTORY}

Choose exactly ONE action to take from the available actions.

Output only the action name.
"""

TEXT_SUMMARY_PROMPT = """You are the red triangular agent solving a maze. Your mission is to navigate to the green square.

mechanisms present:
{MAZE_SPECIFIC_MECHS}

Available actions:
{DOMAIN_SPECIFIC_ACTION_LIST}

Current observation:
{CURRENT_FRAME}

Exploration summary:
{MECHANISMS_INTERACTED_WITH}.{SUBGOALS_ACHIEVED}.{PATH_IN_LAST_10_FRAMES}.
# Example: you've interacted with the yellow key and the yellow door.
# You've opened the yellow door.
# In the last 10 frames, you've traveled from [1, 1] to [1, 10]

Inventory:
{INVENTORY}

Choose exactly ONE action to take from the available actions.

Output only the action name.
"""

CONDITION_SET = {
	"id": "condition_set_3",
	"name": "Context window",
	"comparisons": [
		"0 history (current frame only)",
		"Last 3 frames",
		"Current frame + text summary of prior actions",
	],
	"decision": "Is there a cheap alternative to feeding multiple frames?",
	"prompts": {
		"history": HISTORY_PROMPT,
		"text_summary": TEXT_SUMMARY_PROMPT,
	},
}
