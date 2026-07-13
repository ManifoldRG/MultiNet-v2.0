"""QueryingMode.parse_actions must anchor to the designated output line.

The sweep instructed every query mode to answer on a ``FINAL_OUTPUT:`` line,
but the subgoal/full_trajectory parser searched ``ACTIONS:`` anywhere first —
so a reasoning line like "Actions: MOVE_FORWARD x 5" preempted a perfectly
valid FINAL_OUTPUT plan and returned []. 28% (19/67) of full_trajectory
queries were rejected this way (a correct Qwen empty_room plan among them).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interface.querying import QueryingMode


def test_final_output_beats_reasoning_actions_line():
    reply = (
        "I need to cross the room.\n"
        "Actions: MOVE_FORWARD x 5 gets me to the wall.\n"
        "Then I turn and finish.\n"
        "FINAL_OUTPUT: MOVE_FORWARD, MOVE_FORWARD, TURN_LEFT, MOVE_FORWARD"
    )
    mode = QueryingMode("full_trajectory")
    assert mode.parse_actions(reply) == [
        "MOVE_FORWARD",
        "MOVE_FORWARD",
        "TURN_LEFT",
        "MOVE_FORWARD",
    ]


def test_line_anchored_actions_fallback_still_works():
    reply = "SUB_GOAL: reach the key\nACTIONS: MOVE_FORWARD, PICKUP"
    mode = QueryingMode("subgoal")
    assert mode.parse_actions(reply) == ["MOVE_FORWARD", "PICKUP"]
    assert mode.current_subgoal == "reach the key"


def test_last_actions_line_wins_over_earlier_one():
    reply = (
        "ACTIONS: TURN_LEFT (no, wait — that hits the wall)\n"
        "Reconsidering.\n"
        "ACTIONS: MOVE_FORWARD, MOVE_FORWARD"
    )
    mode = QueryingMode("full_trajectory")
    assert mode.parse_actions(reply) == ["MOVE_FORWARD", "MOVE_FORWARD"]


def test_mid_line_actions_mention_is_not_an_output_line():
    reply = (
        "My plan uses these actions: MOVE_FORWARD then TURN_LEFT.\n"
        "FINAL_OUTPUT: TURN_RIGHT"
    )
    mode = QueryingMode("subgoal")
    assert mode.parse_actions(reply) == ["TURN_RIGHT"]


def test_step_by_step_unchanged():
    mode = QueryingMode("step_by_step")
    assert mode.parse_actions("FINAL_OUTPUT: MOVE_FORWARD, TURN_LEFT") == ["MOVE_FORWARD"]
