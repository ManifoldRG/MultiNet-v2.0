"""Querying strategy prompt templates."""

SUBGOAL_SUFFIX = (
    "For each turn output:\n"
    "  SUB_GOAL: <short description of your next waypoint>\n"
    "  ACTIONS: <comma-separated action list to reach it>"
)

FULL_TRAJECTORY_SUFFIX = (
    "Output your complete trajectory once as:\n"
    "  SUB_GOAL: <short description of the full plan>\n"
    "  ACTIONS: <comma-separated action list from start to finish>\n"
    "The last action in ACTIONS should be DONE (when you expect to be at the goal).\n"
    "You will not be queried again — this is your only planning turn."
)
