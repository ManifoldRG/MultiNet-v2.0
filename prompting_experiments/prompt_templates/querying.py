"""Querying strategy prompt templates."""

SUBGOAL_SUFFIX = (
    "For each turn output:\n"
    "  SUB_GOAL: <short description of your next waypoint>\n"
    "  ACTIONS: <comma-separated action list to reach it>"
)

FULL_TRAJECTORY_QUESTION = (
    "What is the full sequence of actions you will take to complete the task?"
)

FULL_TRAJECTORY_SUFFIX = ""
