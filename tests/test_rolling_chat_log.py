"""A rolling-chat query record must log the request that was sent, not a trimmed one."""

from __future__ import annotations

import json

from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.task_spec import TaskSpecification
from interface.agents.reply import Reply
from interface.config import ExperimentConfig
from interface.episode_step import EpisodeStepper
from interface.runner import build_runner

_ROOM = {
    "task_id": "room",
    "seed": 0,
    "difficulty_tier": 1,
    "maze": {"dimensions": [8, 8], "walls": [], "start": [1, 1], "goal": [6, 6]},
    "mechanisms": {},
    "goal": {"type": "reach_position", "target": [6, 6]},
    "max_steps": 8,
}


def test_rolling_chat_logs_the_request_it_sent():
    spec = TaskSpecification.from_dict(_ROOM)
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    config = ExperimentConfig(
        prompting="minimal",
        in_context_learning="zero_shot",
        observation="image_only",
        context_window="current",
        chat_history="rolling",
        chat_turns_max=1,
    )
    stepper = EpisodeStepper(build_runner(config, backend, spec))
    stepper.start()
    sent = []
    walk = ["MOVE_FORWARD", "TURN_RIGHT", "MOVE_FORWARD", "TURN_LEFT"]
    while (messages := stepper.next_query()) is not None:
        sent.append(json.dumps(messages, sort_keys=True))
        stepper.apply_reply(
            Reply(text=f"FINAL_OUTPUT: {walk[len(sent) % 4]}", usage={"input_tokens": 1, "output_tokens": 1})
        )
    logged = [
        json.dumps(rec["agent_messages"], sort_keys=True)
        for rec in stepper.transcript
        if rec.get("kind") == "query"
    ]
    # Trimming starts at the third query; before the fix the log lost a turn pair there.
    assert len(sent) >= 4
    assert logged == sent
