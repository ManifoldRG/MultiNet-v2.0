"""Tests for query-boundary checkpoint + deterministic resume-by-replay.

``interface.episode_checkpoint`` lets the batch-API lockstep runner survive a
crash mid-round: at a query boundary (both action buffers empty) it writes a
JSON ``checkpoint.json``; on restart ``resume_stepper`` rebuilds a fresh
``EpisodeStepper`` by replaying the recorded action tokens through a fresh
backend so that finishing the episode yields a result indistinguishable (modulo
latency) from an uninterrupted run.
"""

from __future__ import annotations

import json

import pytest

from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.task_spec import TaskSpecification
from interface.agents.reply import Reply
from interface.config import ExperimentConfig
from interface.episode_checkpoint import resume_stepper, save_checkpoint
from interface.episode_log import _json_safe
from interface.episode_step import EpisodeStepper
from interface.runner import build_runner


# A straight 7x3 corridor: agent starts at (1,1) facing EAST, goal at (5,1).
# In the egocentric action space each MOVE_FORWARD is one primitive == one
# query, so [MOVE_FORWARD]*4 is a clean four-query episode ending in success.
_CORRIDOR_SPEC = {
    "task_id": "corridor",
    "seed": 0,
    "difficulty_tier": 1,
    "maze": {"dimensions": [7, 3], "walls": [], "start": [1, 1], "goal": [5, 1]},
    "mechanisms": {},
    "goal": {"type": "reach_position", "target": [5, 1]},
    "max_steps": 30,
}
_CORRIDOR_REPLIES = ["MOVE_FORWARD", "MOVE_FORWARD", "MOVE_FORWARD", "MOVE_FORWARD"]

# A cardinal episode where one query (MOVE_SOUTH while facing EAST) expands into
# two primitives (TURN_RIGHT, MOVE_FORWARD); exercises multi-step-per-query
# replay.
_CARDINAL_SPEC = {
    "task_id": "cardinal_resume",
    "seed": 0,
    "difficulty_tier": 1,
    "maze": {"dimensions": [5, 5], "walls": [], "start": [1, 1], "goal": [3, 3]},
    "mechanisms": {},
    "goal": {"type": "reach_position", "target": [3, 3]},
    "max_steps": 30,
}
_CARDINAL_REPLIES = ["MOVE_EAST", "MOVE_EAST", "MOVE_SOUTH", "MOVE_SOUTH"]


def _build_runner(spec_dict, **config_kwargs):
    spec = TaskSpecification.from_dict(spec_dict)
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    return build_runner(ExperimentConfig(**config_kwargs), backend, spec)


def _reply(text):
    return Reply(text=f"FINAL_OUTPUT: {text}", usage={"input_tokens": 8, "output_tokens": 2})


def _scrub(result):
    """JSON-safe view with non-deterministic latency dropped, for equality."""
    scrubbed = _json_safe(result)
    for rec in scrubbed["transcript"]:
        rec.pop("llm_latency_s", None)
    return scrubbed


def _drive(
    runner,
    replies,
    *,
    checkpoint_after=None,
    checkpoint_path=None,
    resume_builder=None,
    stats=None,
):
    """Hand-drive a stepper to completion.

    If ``checkpoint_after`` is set, at the query boundary reached after that
    many replies have been applied, save a checkpoint and swap in a resumed
    stepper (built from ``resume_builder``) before continuing. ``stats`` (a
    dict, if given) records how many saves/resumes actually happened so tests
    can prove the checkpoint path was exercised, not silently skipped.
    """
    if stats is not None:
        stats.setdefault("saves", 0)
        stats.setdefault("resumes", 0)
    stepper = EpisodeStepper(runner)
    stepper.start()
    applied = 0
    while (messages := stepper.next_query()) is not None:
        if applied == checkpoint_after:
            # We are at a clean query boundary (buffers drained). Persist and
            # resume onto a completely fresh backend/runner.
            save_checkpoint(checkpoint_path, stepper)
            if stats is not None:
                stats["saves"] += 1
            stepper = resume_builder(checkpoint_path)
            if stats is not None:
                stats["resumes"] += 1
            checkpoint_after = None  # only once
            continue
        stepper.apply_reply(_reply(replies[applied]))
        applied += 1
    return stepper.result()


def test_resume_matches_uninterrupted_run(tmp_path):
    # Uninterrupted baseline.
    baseline = _drive(_build_runner(_CORRIDOR_SPEC), _CORRIDOR_REPLIES)
    assert baseline["success"] is True
    assert baseline["query_count"] == 4

    # Save after two applied queries, resume on a fresh backend, finish.
    ckpt = tmp_path / "checkpoint.json"
    stats = {}
    resumed = _drive(
        _build_runner(_CORRIDOR_SPEC),
        _CORRIDOR_REPLIES,
        checkpoint_after=2,
        checkpoint_path=ckpt,
        resume_builder=lambda p: resume_stepper(p, runner=_build_runner(_CORRIDOR_SPEC)),
        stats=stats,
    )

    assert stats == {"saves": 1, "resumes": 1}  # the interruption really happened
    assert resumed["success"] is True
    assert _scrub(resumed) == _scrub(baseline)


def test_resume_multistep_query_expansion(tmp_path):
    # A cardinal MOVE_SOUTH expands into two primitives; resume mid-way must
    # replay the correct number of backend steps.
    baseline = _drive(_build_runner(_CARDINAL_SPEC, action_space="cardinal"), _CARDINAL_REPLIES)
    assert baseline["success"] is True

    ckpt = tmp_path / "checkpoint.json"
    stats = {}
    resumed = _drive(
        _build_runner(_CARDINAL_SPEC, action_space="cardinal"),
        _CARDINAL_REPLIES,
        checkpoint_after=3,
        checkpoint_path=ckpt,
        resume_builder=lambda p: resume_stepper(
            p, runner=_build_runner(_CARDINAL_SPEC, action_space="cardinal")
        ),
        stats=stats,
    )
    assert stats == {"saves": 1, "resumes": 1}
    assert resumed["success"] is True
    assert _scrub(resumed) == _scrub(baseline)


def test_resume_at_first_query_boundary(tmp_path):
    # Checkpoint taken right after start(), before any reply — zero steps to
    # replay — must still resume to an identical finish.
    baseline = _drive(_build_runner(_CORRIDOR_SPEC), _CORRIDOR_REPLIES)
    ckpt = tmp_path / "checkpoint.json"
    stats = {}
    resumed = _drive(
        _build_runner(_CORRIDOR_SPEC),
        _CORRIDOR_REPLIES,
        checkpoint_after=0,
        checkpoint_path=ckpt,
        resume_builder=lambda p: resume_stepper(p, runner=_build_runner(_CORRIDOR_SPEC)),
        stats=stats,
    )
    assert stats == {"saves": 1, "resumes": 1}
    assert resumed["success"] is True
    assert _scrub(resumed) == _scrub(baseline)


def test_resume_r1_like_config(tmp_path):
    # image_only observation + text_summary_and_last3 history is the R1 config:
    # the post-resume query messages embed prior-step frames, so the frame
    # reattach on replay must reproduce them byte-for-byte.
    cfg = dict(observation="image_only", context_window="text_summary_and_last3")
    baseline = _drive(_build_runner(_CORRIDOR_SPEC, **cfg), _CORRIDOR_REPLIES)
    ckpt = tmp_path / "checkpoint.json"
    stats = {}
    resumed = _drive(
        _build_runner(_CORRIDOR_SPEC, **cfg),
        _CORRIDOR_REPLIES,
        checkpoint_after=2,
        checkpoint_path=ckpt,
        resume_builder=lambda p: resume_stepper(p, runner=_build_runner(_CORRIDOR_SPEC, **cfg)),
        stats=stats,
    )
    assert stats == {"saves": 1, "resumes": 1}
    assert resumed["success"] is True
    assert _scrub(resumed) == _scrub(baseline)


def test_resume_rebuilds_stall_state(tmp_path):
    # progress_stall_k must be rebuilt from the replayed states so a stall that
    # would trip after resume trips at the same step as in an uninterrupted run.
    # Agent bumps the west wall repeatedly (no progress) -> stalls.
    spec = dict(_CORRIDOR_SPEC, task_id="stall")
    replies = ["TURN_LEFT", "TURN_LEFT", "MOVE_FORWARD", "MOVE_FORWARD", "MOVE_FORWARD"]
    cfg = dict(progress_stall_k=2)
    baseline = _drive(_build_runner(spec, **cfg), replies)
    assert baseline["end_reason"] == "stalled"
    ckpt = tmp_path / "checkpoint.json"
    stats = {}
    # checkpoint_after=1: the first TURN_LEFT has already accrued stall_count=1,
    # so the resumed stepper must rebuild that from replay for the second
    # TURN_LEFT to trip the stall. (checkpoint_after=2 would never be reached —
    # with stall_k=2 the episode ends inside that drain.)
    resumed = _drive(
        _build_runner(spec, **cfg),
        replies,
        checkpoint_after=1,
        checkpoint_path=ckpt,
        resume_builder=lambda p: resume_stepper(p, runner=_build_runner(spec, **cfg)),
        stats=stats,
    )
    # Makes vacuity impossible: the save/resume really happened.
    assert stats == {"saves": 1, "resumes": 1}
    assert resumed["end_reason"] == "stalled"
    assert _scrub(resumed) == _scrub(baseline)


def test_resume_rejects_multiturn_chat(tmp_path):
    runner = _build_runner(_CORRIDOR_SPEC, chat_history="rolling")
    stepper = EpisodeStepper(runner)
    stepper.start()
    stepper.next_query()
    ckpt = tmp_path / "checkpoint.json"
    save_checkpoint(ckpt, stepper)
    with pytest.raises(ValueError, match="stateless"):
        resume_stepper(ckpt, runner=_build_runner(_CORRIDOR_SPEC, chat_history="rolling"))


def test_save_refuses_mid_action_drain(tmp_path):
    # After apply_reply the parsed action sits in action_queue (not yet drained);
    # that is NOT a query boundary and save must refuse.
    runner = _build_runner(_CORRIDOR_SPEC)
    stepper = EpisodeStepper(runner)
    stepper.start()
    stepper.next_query()
    stepper.apply_reply(_reply("MOVE_FORWARD"))
    assert stepper.action_queue  # queued, undrained
    with pytest.raises(ValueError, match="query boundary"):
        save_checkpoint(tmp_path / "ckpt.json", stepper)


def test_save_refuses_mid_primitive_drain(tmp_path):
    runner = _build_runner(_CORRIDOR_SPEC)
    stepper = EpisodeStepper(runner)
    stepper.start()
    # Simulate a partially-drained cardinal expansion.
    stepper.primitive_buffer = [("MOVE_FORWARD", "MOVE_EAST")]
    with pytest.raises(ValueError, match="query boundary"):
        save_checkpoint(tmp_path / "ckpt.json", stepper)


def test_corrupted_checkpoint_raises_clean_error(tmp_path):
    bad = tmp_path / "checkpoint.json"
    bad.write_text("{ this is not valid json ]", encoding="utf-8")
    with pytest.raises(ValueError):
        resume_stepper(bad, runner=_build_runner(_CORRIDOR_SPEC))


def test_missing_field_checkpoint_raises_clean_error(tmp_path):
    bad = tmp_path / "checkpoint.json"
    bad.write_text(json.dumps({"seed": 0, "transcript": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        resume_stepper(bad, runner=_build_runner(_CORRIDOR_SPEC))


def test_stale_checkpoint_divergence_raises(tmp_path):
    # Save a valid checkpoint, then tamper a recorded position so replay diverges
    # from the recorded state -> resume must reject it rather than continue.
    runner = _build_runner(_CORRIDOR_SPEC)
    stepper = EpisodeStepper(runner)
    stepper.start()
    stepper.next_query()
    stepper.apply_reply(_reply("MOVE_FORWARD"))
    stepper.next_query()  # drains the step -> back at a query boundary
    ckpt = tmp_path / "checkpoint.json"
    save_checkpoint(ckpt, stepper)

    data = json.loads(ckpt.read_text())
    for rec in data["transcript"]:
        if rec.get("kind") == "step":
            rec["position_after_row_col"] = [99, 99]  # corrupt the recorded outcome
    ckpt.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError):
        resume_stepper(ckpt, runner=_build_runner(_CORRIDOR_SPEC))


def test_checkpoint_written_atomically(tmp_path):
    # A successful save leaves no leftover temp file next to the checkpoint.
    runner = _build_runner(_CORRIDOR_SPEC)
    stepper = EpisodeStepper(runner)
    stepper.start()
    stepper.next_query()
    ckpt = tmp_path / "checkpoint.json"
    save_checkpoint(ckpt, stepper)
    assert ckpt.exists()
    assert list(tmp_path.glob("*.tmp")) == []
    # Round-trips as JSON with the documented fields.
    data = json.loads(ckpt.read_text())
    assert set(data) >= {
        "transcript",
        "seed",
        "query_count",
        "parse_failures",
        "consecutive_failures",
        "finished",
        "end_reason",
    }
