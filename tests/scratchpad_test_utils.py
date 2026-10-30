"""Shared helpers for the scratchpad-memory tests (imported as a top-level
module, like start_map_test_utils): one scripted episode on the R1 corridor
maze, the requests it sends, and those requests as each provider would send
them, with images reduced to a hash of their pixels so the snapshots do not
depend on the PNG encoder."""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from interface.agents.claude import (
    _build_request_body,
    _resolve_cache_markers,
    _to_anthropic_turns,
)
from interface.agents.kimi_k26 import _build_body as kimi_body
from interface.agents.kimi_k26 import _to_openai_messages as kimi_messages
from interface.agents.openai_agent import _build_body as openai_body
from interface.agents.openai_agent import _to_openai_messages as openai_messages
from interface.agents.reply import Reply
from interface.episode_step import EpisodeStepper

SNAPSHOT_DIR = Path(__file__).resolve().parent / "fixtures" / "memory_arms"

# The R2 cell (run_config.mechanics_15_*.json experiment_config), per domain.
R2_CELL = dict(
    prompting="minimal",
    context_window="full",
    chat_history="stateless",
    in_context_learning="zero_shot",
    action_space="egocentric",
    querying="step_by_step",
    progress_stall_k=30,
    prompt_cache=True,
)
NL_CELL = {**R2_CELL, "observation": "text_only"}
IMAGE_CELL = {**R2_CELL, "observation": "image_only"}


def notes(text: str) -> str:
    return f"<scratchpad>\n{text}\n</scratchpad>\n"


# The corridor maze's BFS plan opens MOVE_FORWARD x2, TOGGLE (the switch),
# MOVE_FORWARD x3, TURN_RIGHT, MOVE_FORWARD. Around it, every way a reply can
# carry (or fail to carry) a scratchpad.
REPLIES = [
    # 1: a scratchpad, then the action.
    notes("Start. Plan: follow the corridor to the switch.") + "FINAL_OUTPUT: MOVE_FORWARD",
    # 2: notes that name actions (a FINAL_OUTPUT line and a synonym) must not
    #    be read as the answer.
    notes("FINAL_OUTPUT: TURN_LEFT was a bad idea; do not turn left yet.")
    + "FINAL_OUTPUT: MOVE_FORWARD",
    # 3: a scratchpad but no action: a parse failure keeps the previous notes.
    "<scratchpad>\nNot sure.\n</scratchpad>\nI am not sure where to go.",
    # 4: an action but no scratchpad: the previous notes stay.
    "FINAL_OUTPUT: TOGGLE",
    # 5: notes over the cap (OVERFLOW_CAP): cut, and the model is told.
    notes("Gate should be open now. " + "x" * 100) + "FINAL_OUTPUT: MOVE_FORWARD",
    # 6: an empty scratchpad clears the notes.
    "<scratchpad></scratchpad>\nFINAL_OUTPUT: MOVE_FORWARD",
    # 7: prose around the block.
    "Next cell.\n" + notes("Through the gate.") + "FINAL_OUTPUT: MOVE_FORWARD",
    # 8: the action first, then a long scratchpad (the FINAL_OUTPUT line is
    #    not among the last five lines).
    "FINAL_OUTPUT: TURN_RIGHT\n" + notes("a\nb\nc\nd\ne\nf"),
    # 9-10: plain.
    notes("Corridor east.") + "FINAL_OUTPUT: MOVE_FORWARD",
    notes("Keep going.") + "FINAL_OUTPUT: MOVE_FORWARD",
]
OVERFLOW_CAP = 80

# Each arm's overrides on the R2 cell; the window is K=3 here so the short
# scripted episode slides it.
ARMS = {
    "full_context": {},
    "scratchpad_window": {
        "context_window": "last_n",
        "context_n": 3,
        "memory": "scratchpad_window",
        "scratchpad_max_chars": OVERFLOW_CAP,
    },
    "full_context_scratchpad": {
        "memory": "full_context_scratchpad",
        "scratchpad_max_chars": OVERFLOW_CAP,
    },
}
SCRATCHPAD_ARMS = ("scratchpad_window", "full_context_scratchpad")


def drive(runner, replies=REPLIES, *, usage: dict | None = None) -> tuple[EpisodeStepper, list[list[dict]]]:
    """Run the stepper by hand; return it and every request it sent."""
    stepper = EpisodeStepper(runner, verbose=False)
    stepper.start()
    requests: list[list[dict]] = []
    for reply in replies:
        messages = stepper.next_query()
        if messages is None:
            break
        requests.append(copy.deepcopy(messages))
        stepper.apply_reply(Reply(text=reply, usage=usage))
    return stepper, requests


def _pixels(b64: str) -> str:
    rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
    return "pixels:" + hashlib.sha256(rgb.tobytes() + repr(rgb.shape).encode()).hexdigest()[:16]


def canonical(obj: Any) -> Any:
    """``obj`` with every base64 image replaced by a hash of its pixels."""
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if key == "url" and isinstance(value, str) and value.startswith("data:image"):
                out[key] = _pixels(value.partition(";base64,")[2])
            elif key == "data" and obj.get("type") == "base64" and isinstance(value, str):
                out[key] = _pixels(value)
            else:
                out[key] = canonical(value)
        return out
    if isinstance(obj, list):
        return [canonical(v) for v in obj]
    return obj


def provider_bodies(messages: list[dict]) -> dict[str, dict]:
    """The request bodies the R2 panel's agents build from ``messages``
    (Claude sync with its cache markers, GPT-6 Astra, Kimi K3)."""
    system, turns = _to_anthropic_turns(messages)
    system, turns = _resolve_cache_markers(system, turns, ttl="5m", enabled=True)
    return {
        "claude": _build_request_body(
            model="claude-opus-5-5",
            max_tokens=64000,
            temperature=0.0,
            system=system,
            messages=turns,
            enable_thinking=True,
            effort="xhigh",
        ),
        "openai": openai_body(
            openai_messages(messages, "high", True),
            model="gpt-6-astra",
            max_tokens=64000,
            reasoning_effort="xhigh",
            service_tier="flex",
            prompt_cache=True,
        ),
        "kimi": kimi_body(
            kimi_messages(messages),
            model="kimi-k3",
            max_tokens=64000,
            enable_thinking=True,
            reasoning_effort="max",
            cache_ttl="1h",
        ),
    }


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(canonical(obj), sort_keys=True).encode()).hexdigest()


def snapshot(requests: list[list[dict]]) -> list[dict]:
    """Each request as the runner built it, plus a digest of each provider body."""
    return [
        {
            "messages": canonical(messages),
            "provider_body_sha256": {
                name: _digest(body) for name, body in provider_bodies(messages).items()
            },
        }
        for messages in requests
    ]


def write_snapshot(name: str, data: list[dict]) -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_snapshot(name: str) -> list[dict]:
    return json.loads((SNAPSHOT_DIR / f"{name}.json").read_text(encoding="utf-8"))
