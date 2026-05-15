"""Condition set 5: querying strategy."""

SUBGOAL_PROMPT = """You are the red triangular agent solving a maze. Your mission is to navigate to the green square.

mechanisms present:
{MAZE_SPECIFIC_MECHS}

Current observation:
{DOMAIN_SPECIFIC_OBSERVATION}

Inventory:
{INVENTORY}

Before acting, produce a short high-level plan.

Your plan should identify:
1. important keys
2. important doors or gates
3. likely exploration order
4. important switches or bottlenecks

Output concise numbered subgoals only.
"""

SUBGOAL_EXECUTION_PROMPT = """You are the red triangular agent solving a maze. Your mission is to navigate to the green square.

mechanisms present:
{MAZE_SPECIFIC_MECHS}

Current high-level plan:
{SUBGOAL_PLAN}

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
	"name": "Querying strategy",
	"comparisons": [
		"Step-by-step: one action per query",
		"Subgoal planning: model outputs plan first, then executes per-subgoal",
	],
	"decision": "Does planning help? If yes, benchmark tests planning or execution?",
	"prompts": {
		"subgoal": SUBGOAL_PROMPT,
		"subgoal_execution": SUBGOAL_EXECUTION_PROMPT,
	},
}

PROMPTS = CONDITION_SET["prompts"]
