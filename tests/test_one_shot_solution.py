from gridworld.task_spec import TaskSpecification
from gridworld.backends.multigrid_backend import MultiGridBackend
from multigrid.agent import Action
import hashlib
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_ONE_SHOT_DIR = _REPO / "mazes" / "one_shot_example"
_OGBENCH_MAZE_DIR = _REPO / "ogbench" / "ogbench" / "procgen" / "maze_jsons"


def _eval_maze_paths():
    paths = [
        p for p in (_REPO / "mazes").rglob("*.json")
        if _ONE_SHOT_DIR not in p.parents
    ]
    if _OGBENCH_MAZE_DIR.exists():
        paths.extend(_OGBENCH_MAZE_DIR.rglob("*.json"))
    return paths


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


def _maze_signature(payload: dict) -> str:
    """Content hash over the maze-defining fields (ignores task_id/metadata) so a
    renamed copy of an evaluation maze is still detected."""
    sig = {key: payload.get(key) for key in ("maze", "mechanisms", "goal")}
    return hashlib.sha256(json.dumps(sig, sort_keys=True).encode("utf-8")).hexdigest()


def test_one_shot_example_is_disjoint_from_evaluation_mazes():
    """The ICL one-shot example must never double as an evaluation maze: it must
    not share a task_id or maze content with any eval maze, and no manifest may
    reference it. Evaluation mazes are every maze under mazes/ outside the
    dedicated one_shot_example/ directory, plus the ogbench procgen corpus
    that the R1 manifests resolve."""
    one_shot_mazes = {
        path: payload
        for path in sorted(_ONE_SHOT_DIR.glob("*.json"))
        if "maze" in (payload := json.loads(path.read_text()))
    }
    assert one_shot_mazes, "expected at least one one-shot example maze"

    eval_payloads = [
        payload
        for path in _eval_maze_paths()
        if "maze" in (payload := json.loads(path.read_text()))
    ]
    assert eval_payloads, "expected to find evaluation mazes under mazes/"

    eval_task_ids = {p.get("task_id") for p in eval_payloads}
    eval_signatures = {_maze_signature(p) for p in eval_payloads}

    for path, payload in one_shot_mazes.items():
        assert payload.get("task_id") not in eval_task_ids, (
            f"one-shot example {path.name} shares a task_id with an evaluation maze"
        )
        assert _maze_signature(payload) not in eval_signatures, (
            f"one-shot example {path.name} has the same maze content as an evaluation maze"
        )

    # No manifest may pull the one-shot example into a run (by file name or task_id).
    one_shot_names = {path.name for path in one_shot_mazes}
    one_shot_ids = {p.get("task_id") for p in one_shot_mazes.values()}
    for manifest in (_REPO / "gridworld" / "fixtures").glob("manifest*.json"):
        text = manifest.read_text()
        for name in one_shot_names:
            assert name not in text, f"{manifest.name} references one-shot maze {name}"
        manifest_ids = {t.get("task_id") for t in json.loads(text).get("tasks", [])}
        assert not (one_shot_ids & manifest_ids), (
            f"{manifest.name} references a one-shot example task_id"
        )
