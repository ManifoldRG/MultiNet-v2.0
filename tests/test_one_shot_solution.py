from gridworld.task_spec import TaskSpecification
from gridworld.backends.multigrid_backend import MultiGridBackend
from multigrid.agent import Action
import json


def _action_to_enum(a: str) -> Action:
    if a == "MOVE_FORWARD":
        return Action.FORWARD
    return getattr(Action, a)


def test_one_shot_solution_reaches_goal():
    """Load the one-shot example maze and solution and verify it solves the maze."""
    spec = TaskSpecification.from_json(
        "mazes/one_shot_example/one_shot_example_14x14_dense_kr_sg_kb_2.json"
    )

    with open("mazes/one_shot_example/one_shot_example_solution.json") as f:
        sol = json.load(f)

    actions = sol.get("actions", [])

    backend = MultiGridBackend(tiling="square", render_mode="state_dict")
    backend.configure(spec)
    env = backend.env

    obs, info = env.reset()

    for a in actions:
        if a == "DONE":
            break
        act = _action_to_enum(a)
        obs, reward, terminated, truncated, info = env.step(int(act))
        assert not info.get("invalid_action"), f"Invalid action {a} at step, info={info}"
        if terminated:
            break

    assert env.state.check_goal(), f"Solution failed to reach goal; final cell={env.state.agent.cell_id}"
