"""Full-history prompts resend every prior frame; our own copies must not grow as steps squared.

Each step's frame is PNG-encoded once and every query shares that string; the
episode log and the lockstep checkpoint store each image once and reference
it. None of this may change a byte the model sees, and the stored request must
rebuild exactly what was sent.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

import interface.episode_step as episode_step
import interface.observation as observation
import interface.renderer as renderer
from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.task_spec import TaskSpecification
from interface.agents.reply import Reply
from interface.config import ExperimentConfig
from interface.episode_checkpoint import (
    discard_checkpoint,
    parts_dir,
    resume_stepper,
    save_checkpoint,
)
from interface.episode_log import _json_safe, flush_episode_log, load_query_messages
from interface.episode_step import EpisodeStepper
from interface.renderer import rgb_to_image_block
from interface.runner import build_runner

# An open room the scripted walk never solves, so every episode runs to max_steps.
_ROOM = {
    "task_id": "room",
    "seed": 0,
    "difficulty_tier": 1,
    "maze": {"dimensions": [8, 8], "walls": [], "start": [1, 1], "goal": [6, 6]},
    "mechanisms": {},
    "goal": {"type": "reach_position", "target": [6, 6]},
    "max_steps": 10,
}
_WALK = ["MOVE_FORWARD", "TURN_RIGHT", "MOVE_FORWARD", "TURN_LEFT"]


def _runner(**config):
    spec = TaskSpecification.from_dict(_ROOM)
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    base = {"prompting": "minimal", "in_context_learning": "zero_shot"}
    return build_runner(ExperimentConfig(**{**base, **config}), backend, spec)


def _reply(i):
    return Reply(
        text=f"FINAL_OUTPUT: {_WALK[i % len(_WALK)]}",
        usage={"input_tokens": 8, "output_tokens": 2},
    )


def _play(runner, *, on_boundary=None):
    """Drive a stepper to the end; return (result, the messages sent per query)."""
    stepper = EpisodeStepper(runner)
    stepper.start()
    sent = []
    while (messages := stepper.next_query()) is not None:
        if on_boundary is not None:
            on_boundary(stepper)
        sent.append(json.dumps(messages, sort_keys=True))
        stepper.apply_reply(_reply(len(sent) - 1))
    return stepper.result(), sent


_MODES = [
    (obs, ctx)
    for obs in ("image_only", "image_text", "text_only")
    for ctx in ("last_n", "text_summary_and_last_n", "full")
]


def _fresh_block(rec):
    """The pre-cache path: encode the frame's pixels anew on every query."""
    rgb = np.asarray(Image.open(io.BytesIO(rec["_decision_frame_png"])).convert("RGB"))
    return rgb_to_image_block(rgb)


@pytest.mark.parametrize("obs,ctx", _MODES)
def test_frame_cache_leaves_prompt_bytes_unchanged(obs, ctx, monkeypatch):
    _, cached = _play(_runner(observation=obs, context_window=ctx))
    monkeypatch.setattr(observation, "_decision_frame_block", _fresh_block)
    _, fresh = _play(_runner(observation=obs, context_window=ctx))
    assert len(cached) > 5
    assert cached == fresh


def test_frames_are_encoded_once(monkeypatch):
    calls = []
    real = renderer.rgb_to_png_bytes

    def counted(rgb):
        calls.append(1)
        return real(rgb)

    monkeypatch.setattr(renderer, "rgb_to_png_bytes", counted)
    monkeypatch.setattr(episode_step, "rgb_to_png_bytes", counted)
    result, sent = _play(_runner(observation="image_only", context_window="full"))
    steps = result["steps_used"]
    # One encode per frame (reset + each step), one per query's current image,
    # one for the opening prompt. Without the cache, query q also re-encoded its
    # q - 1 history frames.
    assert len(sent) == steps and steps >= 8
    assert len(calls) <= 2 * steps + 2


def test_records_hold_png_bytes_not_arrays():
    result, _ = _play(_runner(observation="image_only", context_window="full"))
    frames = 0
    for rec in result["transcript"]:
        for key, value in rec.items():
            assert not isinstance(value, np.ndarray), key
            frames += key.endswith("_png") and isinstance(value, bytes)
    assert frames >= 2 * result["steps_used"] + 1


@pytest.mark.parametrize(
    "config",
    [
        {"observation": "image_only", "context_window": "full"},
        {"observation": "image_text", "context_window": "last_n"},
        {"observation": "image_only", "context_window": "last_n", "in_context_learning": "one_shot"},
        {"observation": "image_only", "context_window": "current", "chat_history": "rolling"},
    ],
)
def test_logged_queries_rebuild_the_sent_messages(config, tmp_path):
    result, sent = _play(_runner(**config))
    out = tmp_path / "episode"
    flush_episode_log(result, out)

    for i, expected in enumerate(sent, start=1):
        rebuilt = load_query_messages(out / "queries" / f"query_{i:03d}")
        assert json.dumps(rebuilt, sort_keys=True) == expected

    # Every decision frame is a step frame: no per-query image copies.
    assert list((out / "queries").rglob("*.png")) == []
    frames = list((out / "frames").glob("*.png"))
    assert len(frames) == result["steps_used"] + 1
    extra = list((out / "images").glob("*.png")) if (out / "images").exists() else []
    if config.get("in_context_learning") == "one_shot":
        assert 1 <= len(extra) <= 2  # the example image(s), once
    else:
        assert extra == []


def test_checkpoint_writes_each_request_and_image_once(tmp_path):
    ckpt = tmp_path / "checkpoint.json"
    sizes = []

    def save(stepper):
        save_checkpoint(ckpt, stepper)
        sizes.append(ckpt.stat().st_size)

    result, _ = _play(_runner(observation="image_only", context_window="full"), on_boundary=save)
    text = ckpt.read_text()
    assert "data:image" not in text
    parts = parts_dir(ckpt)
    images = list((parts / "images").glob("*.png"))
    requests = list(parts.glob("query_*.json"))
    # One request per checkpointed query; one image per distinct frame.
    assert len(requests) == len(sizes) - 1
    assert len(images) <= result["steps_used"] + 1
    # The checkpoint itself grows by a roughly constant amount per query.
    growth = [b - a for a, b in zip(sizes, sizes[1:])]
    assert max(growth[2:]) < 2 * min(growth[2:])


@pytest.mark.parametrize("obs", ["image_only", "image_text"])
def test_full_history_resume_matches_uninterrupted_run(obs, tmp_path):
    baseline, baseline_sent = _play(_runner(observation=obs, context_window="full"))

    ckpt = tmp_path / "checkpoint.json"
    stepper = EpisodeStepper(_runner(observation=obs, context_window="full"))
    stepper.start()
    sent = []
    while (messages := stepper.next_query()) is not None:
        if len(sent) in (3, 6):
            save_checkpoint(ckpt, stepper)
            stepper = resume_stepper(
                ckpt, runner=_runner(observation=obs, context_window="full")
            )
            messages = stepper.next_query()
        sent.append(json.dumps(messages, sort_keys=True))
        stepper.apply_reply(_reply(len(sent) - 1))
    resumed = stepper.result()

    assert sent == baseline_sent

    def scrub(result):
        out = _json_safe(result)
        for rec in out["transcript"]:
            rec.pop("llm_latency_s", None)
        return out

    assert scrub(resumed) == scrub(baseline)


def test_inline_checkpoint_still_resumes(tmp_path):
    # Older checkpoints (and scripts/resume_from_archive.py) store requests inline.
    runner = _runner(observation="image_only", context_window="full")
    stepper = EpisodeStepper(runner)
    stepper.start()
    for i in range(3):
        stepper.next_query()
        stepper.apply_reply(_reply(i))
    stepper.next_query()
    ckpt = tmp_path / "checkpoint.json"
    save_checkpoint(ckpt, stepper)
    data = json.loads(ckpt.read_text())
    data["transcript"] = _json_safe(stepper.transcript)
    discard_checkpoint(ckpt)
    ckpt.write_text(json.dumps(data), encoding="utf-8")

    resumed = resume_stepper(ckpt, runner=_runner(observation="image_only", context_window="full"))
    assert _json_safe(resumed.transcript) == _json_safe(stepper.transcript)


def test_discard_checkpoint_removes_its_parts(tmp_path):
    ckpt = tmp_path / "checkpoint.json"
    _play(
        _runner(observation="image_only", context_window="full"),
        on_boundary=lambda s: save_checkpoint(ckpt, s),
    )
    assert parts_dir(ckpt).exists()
    discard_checkpoint(ckpt)
    assert not ckpt.exists() and not parts_dir(ckpt).exists()
