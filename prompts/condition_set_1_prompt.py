"""Condition set 1: prompt verbosity."""

STANDARD_PROMPT = """You are the red triangular agent solving a maze. Your mission is to navigate to the green square.

mechanisms present:
{MAZE_SPECIFIC_MECHS}

Available actions:
{DOMAIN_SPECIFIC_ACTION_LIST}

Current observation:
{DOMAIN_SPECIFIC_OBSERVATION} # image for 2D and 3D; NL for NL

Inventory:
{INVENTORY}

Choose exactly ONE action to take from the available actions.

Output only the action name.
"""

EXPLICIT_PROMPT = """You are the red triangular agent solving a maze. Your mission is to navigate to the green square.

mechanisms present:
{MAZE_SPECIFIC_MECHS}

Detailed rules:
1. Keys only open doors with matching colors.
2. Keys are consumed immediately after opening a matching door.
3. Opened doors remain open permanently.
4. Switches toggle associated gates between open and closed states.
5. Walls and closed doors cannot be crossed.
6. The agent occupies exactly one cell at a time.
7. Invalid actions do not help progress toward the goal.

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
	"name": "Prompt",
	"comparisons": [
		"Standard: goal + mechanism descriptions + action list",
		"Verbose: Standard + explicit rules",
	],
	"decision": "If delta < 5%, use Standard. If > 5%, use Verbose.",
	"prompts": {
		"standard": STANDARD_PROMPT,
		"explicit": EXPLICIT_PROMPT,
	},
}

PROMPTS = CONDITION_SET["prompts"]
