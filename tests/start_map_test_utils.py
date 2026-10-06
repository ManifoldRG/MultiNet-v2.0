"""Shared helpers for the 3D start-map tests (imported as a top-level module,
like render3d_test_utils): scripted episodes over an R1-panel maze and a
GL-independent description of every request the model was sent."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
from pathlib import Path

from gridworld.baselines import plan_bfs_path
from gridworld.task_spec import TaskSpecification
from interface.config import ExperimentConfig
from interface.parser import ACTION_ORDER
from interface.renderer import rgb_to_image_block
from interface.runner import build_runner

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "gridworld" / "fixtures" / "start_map"
# r1_B1_8x8_corridor_swg_0 from manifest.r1_balanced_03.json, copied out of the
# ogbench submodule so these tests do not need it.
R1_MAZE = FIXTURE_DIR / "r1_B1_8x8_corridor_swg_0.json"
GOLDEN = FIXTURE_DIR / "golden_prompts.json"

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


class RecordingAgent:
    """Scripted step_by_step agent keeping a deep copy of every request."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.requests: list[list[dict]] = []
        self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}

    def __call__(self, messages):
        self.requests.append(copy.deepcopy(messages))
        i = len(self.requests) - 1
        return self._replies[i] if i < len(self._replies) else "FINAL_OUTPUT: DONE"


def run_scripted(backend, spec, config_kwargs, replies=None):
    """Configure ``backend``, run one scripted episode; (runner, result, agent)."""
    backend.configure(spec)
    runner = build_runner(ExperimentConfig(**config_kwargs), backend, spec)
    agent = RecordingAgent(scripted_replies(spec) if replies is None else replies)
    result = runner.run(agent, verbose=False)
    return runner, result, agent


def _url(block: dict) -> str:
    return block["image_url"]["url"]


def frame_labels(runner, result) -> dict[str, str]:
    """image url -> which frame of this episode it is ("reset", "step3", ...).

    Labels instead of pixel hashes keep the goldens independent of the GL
    implementation (EGL and OSMesa rasterise slightly differently) while still
    pinning which frame sits where. First label wins for identical frames."""
    labels: dict[str, str] = {}

    def add(rgb, label):
        if rgb is not None:
            labels.setdefault(_url(rgb_to_image_block(rgb)), label)

    add(getattr(runner, "start_map_rgb", None), "start_map")
    for rec in result["transcript"]:
        if rec["kind"] == "reset":
            add(rec.get("_reset_frame_rgb"), "reset")
        elif rec["kind"] == "step":
            add(rec.get("_post_step_rgb"), f"step{rec['step_index']}")
    return labels


def describe_message(message: dict, labels: dict[str, str]) -> dict:
    content = message["content"]
    if isinstance(content, str):
        return {"role": message["role"], "content": content}
    blocks = []
    for block in content:
        if block["type"] == "text":
            blocks.append({"text": block["text"]})
        else:
            url = _url(block)
            blocks.append(
                {"image": labels.get(url) or "sha256:" + hashlib.sha256(url.encode()).hexdigest()}
            )
    return {"role": message["role"], "content": blocks}


def describe_requests(requests, labels) -> list[list[dict]]:
    return [[describe_message(m, labels) for m in request] for request in requests]


def describe_episode(runner, result, agent) -> dict:
    """Every request as sent, plus the per-query agent_messages as logged."""
    labels = frame_labels(runner, result)
    logged = [rec["agent_messages"] for rec in result["transcript"] if rec["kind"] == "query"]
    return {
        "sent": describe_requests(agent.requests, labels),
        "logged": describe_requests(logged, labels),
        "end_reason": result["end_reason"],
        "steps_used": result["steps_used"],
    }
