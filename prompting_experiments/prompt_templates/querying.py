"""Querying strategy prompt templates."""

SUBGOAL_SUFFIX = ""

FULL_TRAJECTORY_QUESTION = (
    "What is the full sequence of actions you will take to complete the task?"
)

SINGLE_ACTION_FINAL_OUTPUT_INSTRUCTION = (
    "Output exactly:\n"
    "FINAL_OUTPUT: <action>"
)

SUBGOAL_FINAL_OUTPUT_INSTRUCTION = (
    "Output exactly:\n"
    "SUB_GOAL: <short description of your next sub_goal>\n"
    "FINAL_OUTPUT:  <a>, <b>, ...  (comma-separated; one or more valid actions to reach it)"
)

FULL_TRAJECTORY_FINAL_OUTPUT_INSTRUCTION = (
    "Output exactly:\n"
    "FINAL_OUTPUT: <a>, <b>, ...  (comma-separated; one or more valid actions)"
)

FULL_TRAJECTORY_SUFFIX = ""
