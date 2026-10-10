"""M10 end to end: BFS onto a death portal, then the BFS solution.

The route crosses every new tile (rotating tiles, two freezes, a death portal,
both teleporters, a ride). The expected minimal feedback is what #61 emitted on
the same route; the world-model label must reproduce it exactly.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.task_spec import TaskSpecification
from gridworld.world_model import TaskPlanningContext, successors
from interface.config import ExperimentConfig
from interface.observation import text_summary_history
from interface.runner import build_runner

MAZE = Path(__file__).parent / "fixtures" / "mechanics" / (
    "M10_18x18_dense_kr_sg_kb_portals_death_rotating_frozen_0.json"
)
ACTION_NAMES = ["TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD", "PICKUP", "DROP", "TOGGLE", "DONE"]

# (step, event_type, minimal feedback) for every step that is not MOVED / TURNED.
EXPECTED = [
    (7, "ROTATING", "ROTATING NORTH"),
    (8, "NOTHING", "NOTHING"),
    (9, "FROZEN", "FROZEN"),
    (10, "FROZEN", "FROZEN"),
    (11, "FROZEN", "FROZEN"),
    (12, "FROZEN", "FROZEN"),
    (13, "FROZEN", "FROZEN"),
    (14, "FROZEN", "FROZEN"),
    (15, "DEATH RESET", "DEATH RESET"),
    (20, "ROTATING", "ROTATING SOUTH"),
    (21, "CARRIED", "CARRIED SOUTH"),
    (22, "CARRIED", "CARRIED SOUTH"),
    (26, "TELEPORTED", "TELEPORTED (11,2)"),
    (37, "TELEPORTED", "TELEPORTED (15,14)"),
    (53, "PICKUP", "PICKUP"),
    (60, "FROZEN", "FROZEN"),
    (61, "FROZEN", "FROZEN"),
    (62, "FROZEN", "FROZEN"),
    (63, "FROZEN", "FROZEN"),
    (64, "FROZEN", "FROZEN"),
    (65, "FROZEN", "FROZEN"),
    (66, "ROTATING", "ROTATING EAST"),
    (68, "OPENED", "OPENED"),
    (69, "NOTHING", "NOTHING"),
    (70, "CARRIED", "CARRIED NORTH"),
    (90, "DONE", "DONE"),
]


def _bfs(ctx, start, done) -> list[int]:
    prev = {start: None}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        for t in successors(ctx, state):
            if done(t):
                path, cur = [t.action], state
                while prev[cur] is not None:
                    cur, action = prev[cur]
                    path.append(action)
                return path[::-1]
            if t.next_state not in prev:
                prev[t.next_state] = (state, t.action)
                queue.append(t.next_state)
    raise AssertionError("no route")


class _Scripted:
    def __init__(self, actions):
        self._actions, self._i = actions, 0
        self.last_usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

    def __call__(self, messages):
        action = self._actions[self._i] if self._i < len(self._actions) else "DONE"
        self._i += 1
        return f"FINAL_OUTPUT: {action}"


def test_m10_route_matches_pr61_feedback_step_for_step():
    spec = TaskSpecification.from_json(str(MAZE))
    ctx = TaskPlanningContext(spec)
    start = ctx.initial_state()
    goal = tuple(spec.resolved_goal())
    plan = _bfs(ctx, start, lambda t: t.label == "kill_reset")
    plan += _bfs(ctx, start, lambda t: t.next_state.agent_pos == goal and t.action == 2)
    assert len(plan) == 90

    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    config = ExperimentConfig(
        observation="text_only",
        prompting="standard",
        context_window="text_summary_and_last_n",
        context_n=3,
        querying="step_by_step",
        chat_history="stateless",
        in_context_learning="zero_shot",
        action_space="egocentric",
        feedback="minimal",
    )
    result = build_runner(config, backend, spec).run(
        _Scripted([ACTION_NAMES[a] for a in plan]), verbose=False
    )
    steps = [r for r in result["transcript"] if r.get("kind") == "step"]

    assert result["end_reason"] == "success"
    got = [
        (i, s["event_type"], s["prompt_feedback"])
        for i, s in enumerate(steps, 1)
        if s["event_type"] not in ("MOVED", "TURNED")
    ]
    assert got == EXPECTED

    summary = text_summary_history(result["transcript"], spec)
    for phrase in (
        "stepped on a death portal at (4,4) and lost your progress",
        "teleported to (11,2)",
        "teleported to (15,14)",
        "stepped on a frozen tile at",
        "and rode it in",
    ):
        assert phrase in summary
