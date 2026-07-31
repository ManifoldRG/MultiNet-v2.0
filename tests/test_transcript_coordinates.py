"""Regression test: transcript step positions must be unambiguously named.

R1 (r1-20260717) post-mortem coordinate bug: step records logged
``position_before``/``position_after`` in (row, col) while the field name and
the co-located ``state.agent_position`` use (x, y). A live operator (and any
downstream trajectory analysis) read ``position_after`` as (x, y) and got
transposed coordinates. Prompts/text_summary are internally consistent in
(row, col); the defect was purely the ambiguous field name in the transcript.

Fix: the fields are named ``position_before_row_col`` / ``position_after_row_col``
so the convention is explicit, and the authoritative (x, y) remains
``state_after.agent_position``. These tests pin both.
"""
from gridworld.task_spec import TaskSpecification
from gridworld.backends.minigrid_backend import MiniGridBackend
from interface.coords import to_row_col
from interface.episode_step import EpisodeStepper
from interface.runner import build_runner
from interface.config import ExperimentConfig
from interface.agents.reply import Reply


_SPEC = {
    "task_id": "coord",
    "seed": 0,
    "difficulty_tier": 1,
    "maze": {"dimensions": [7, 7], "walls": [], "start": [1, 1], "goal": [5, 5]},
    "mechanisms": {},
    "goal": {"type": "reach_position", "target": [5, 5]},
    "max_steps": 30,
}


def _stepper():
    spec = TaskSpecification.from_dict(_SPEC)
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    runner = build_runner(ExperimentConfig(action_space="cardinal"), backend, spec)
    st = EpisodeStepper(runner, verbose=False)
    st.start()
    return st


def test_step_position_field_is_explicitly_row_col():
    st = _stepper()
    st.next_query()
    st.apply_reply(Reply(text="FINAL_OUTPUT: MOVE_SOUTH"))
    st.next_query()  # drain buffered primitives into step records
    step = next(r for r in st.transcript if r.get("kind") == "step")

    # Explicit (row, col) field, matching state_after.agent_position transposed.
    xy = tuple(step["state_after"]["agent_position"])
    assert tuple(step["position_after_row_col"]) == tuple(to_row_col(xy))

    # The ambiguous bare names must NOT exist (they invited the x,y misread).
    assert "position_after" not in step
    assert "position_before" not in step


def test_row_col_field_is_transpose_of_xy():
    st = _stepper()
    st.next_query()
    st.apply_reply(Reply(text="FINAL_OUTPUT: MOVE_SOUTH"))
    st.next_query()  # drain buffered primitives into step records
    step = next(r for r in st.transcript if r.get("kind") == "step")
    row, col = step["position_after_row_col"]
    x, y = step["state_after"]["agent_position"]
    assert (row, col) == (y, x)


def test_facing_forward_movement_has_no_off_by_one():
    """Regression for the R1 'rotation off-by-one' suspicion (§RUN-LEVEL
    ASTERISK). It was an artifact of the transposed-coordinate misread, not a
    code bug: each facing's FORWARD step must move exactly one cell in the
    matching (x, y) direction."""
    from gridworld.custom_env import CustomMiniGridEnv
    from interface.coords import _DIR_TO_FACING

    env = CustomMiniGridEnv(
        width=9, height=9, max_steps=999, agent_start_pos=(4, 4),
        agent_start_dir=0, goal_pos=(7, 7), render_mode="rgb_array", highlight=False,
    )
    env.reset(seed=0)
    expected = {"EAST": (1, 0), "SOUTH": (0, 1), "WEST": (-1, 0), "NORTH": (0, -1)}
    for d in range(4):
        env.agent_pos = (4, 4)
        env.agent_dir = d
        env.step(int(env.actions.forward))
        after = (int(env.agent_pos[0]), int(env.agent_pos[1]))
        delta = (after[0] - 4, after[1] - 4)
        assert delta == expected[_DIR_TO_FACING[d]], (
            f"facing {_DIR_TO_FACING[d]} moved {delta}, expected {expected[_DIR_TO_FACING[d]]}"
        )
