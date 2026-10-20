"""Shared helpers for the 3D start-map tests (imported as a top-level module,
like render3d_test_utils): an R1-panel maze and scripted replies for it."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from gridworld.baselines import plan_bfs_path
from gridworld.task_spec import TaskSpecification
from interface.parser import ACTION_ORDER

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "gridworld" / "fixtures" / "start_map"
# r1_B1_8x8_corridor_swg_0 from manifest.r1_balanced_03.json, copied out of the
# ogbench submodule so these tests do not need it.
R1_MAZE = FIXTURE_DIR / "r1_B1_8x8_corridor_swg_0.json"

# The R1 fixed cell (run_config.r1.json experiment_config).
R1_CELL = dict(
    prompting="minimal",
    observation="image_only",
    context_window="text_summary_and_last_n",
    context_n=3,
    chat_history="stateless",
    in_context_learning="zero_shot",
    action_space="egocentric",
    querying="step_by_step",
    progress_stall_k=30,
)

UNPARSEABLE = "I am not sure where to go."


def r1_spec(max_steps: int = 9) -> TaskSpecification:
    spec = TaskSpecification.from_json(str(R1_MAZE))
    return dataclasses.replace(spec, max_steps=max_steps)


def scripted_replies(spec: TaskSpecification, *, parse_failure_at: int | None = 3) -> list[str]:
    """The BFS plan as one FINAL_OUTPUT per query, with one unparseable reply
    (it costs a query, not a step) so the parse-failure path is exercised."""
    plan = plan_bfs_path(dataclasses.replace(spec, max_steps=1000))
    assert plan.success
    replies = [f"FINAL_OUTPUT: {ACTION_ORDER[a]}" for a in plan.actions]
    if parse_failure_at is not None:
        replies.insert(parse_failure_at, UNPARSEABLE)
    return replies
