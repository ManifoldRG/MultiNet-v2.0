"""Condition set 4: action space."""

EGOCENTRIC_PROMPT = """You are the red triangular agent solving a maze. Your mission is to navigate to the green square.

mechanisms present:
{MAZE_SPECIFIC_MECHS}

Available actions:
- TURN_LEFT
- TURN_RIGHT
- MOVE_FORWARD
- INTERACT

Current observation:
{DOMAIN_SPECIFIC_OBSERVATION}

Inventory:
{INVENTORY}

Choose exactly ONE action to take from the available actions.

Output only the action name.
Output only the action name.
"""

CARDINAL_PROMPT = """You are the red triangular agent solving a maze. Your mission is to navigate to the green square.

mechanisms present:
{MAZE_SPECIFIC_MECHS}

Available actions:
- MOVE_NORTH
- MOVE_SOUTH
- MOVE_EAST
- MOVE_WEST
- INTERACT

Current observation:
{DOMAIN_SPECIFIC_OBSERVATION}

Inventory:
{INVENTORY}

Choose exactly ONE action to take from the available actions.

Output only the action name.
"""

CONDITION_SET = {
	"id": "condition_set_4",
	"name": "Action space",
	"comparisons": [
		"Egocentric: TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, INTERACT",
		"Cardinal: MOVE_NORTH/SOUTH/EAST/WEST, INTERACT",
	],
	"decision": "If delta is trivial (<5%), go egocentric. If massive (>15%), reassess.",
	"prompts": {
		"egocentric": EGOCENTRIC_PROMPT,
		"cardinal": CARDINAL_PROMPT,
	},
}
