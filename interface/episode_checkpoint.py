"""Query-boundary checkpoint + deterministic resume-by-replay for the stepper.

The batch-API lockstep runner drives many ``EpisodeStepper`` instances in
lockstep: each round it collects one query per live episode, dispatches them as
one batch, then feeds the replies back. A crash between "query dispatched" and
"replies applied" would otherwise lose the whole round's work.

``save_checkpoint`` serialises a stepper at a **query boundary** — the point
where both action buffers are empty, i.e. the previous round's actions have been
fully drained and the next query has not yet been applied. The episode path has
no un-replayable RNG (the only randomness is ``backend.reset(seed)``), so the
episode can be reconstructed exactly by re-seeding a fresh backend and replaying
the recorded action tokens through it. ``resume_stepper`` does that and returns a
stepper whose subsequent behaviour is indistinguishable (modulo model latency)
from one that never stopped.

Deliberate scope: only ``chat_history="stateless"`` is supported (the R1 config
and every current batch run). Rolling/full multi-turn chat keeps an in-memory
message list that the checkpoint does not persist, so resuming those modes is
rejected explicitly rather than silently reconstructed wrong.

This module never touches ``interface/episode_log.py`` (final-write semantics
are unchanged); it only *reads* its JSON-safe/serialisation helpers.

Each query's request (``agent_messages``) is written once, to a part file in
``<checkpoint>.parts/`` with its images stored once each, and the checkpoint
refers to it. Rewriting every request with every round grew as steps squared
under full-history prompts; this way each round rewrites only the step and
query metadata. ``resume_stepper`` also accepts requests stored inline.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from interface.actions_map import nlu_action_to_int
from interface.coords import agent_row_col
from interface.episode_log import (
    ImageFiles,
    _json_safe,
    deserialize_messages,
    serialize_messages,
    state_snapshot,
)
from interface.episode_step import EpisodeStepper
from interface.progress_watchdog import ProgressStallWatchdog
from prompting_experiments.prompt_templates import feedback as feedback_templates

_FIELDS = (
    "transcript",
    "seed",
    "query_count",
    "parse_failures",
    "consecutive_failures",
    "finished",
    "end_reason",
)


def save_checkpoint(path: str | Path, stepper: EpisodeStepper) -> None:
    """Atomically write a query-boundary checkpoint for ``stepper``.

    Must be called only at a query boundary: both ``action_queue`` and
    ``primitive_buffer`` empty. Calling it mid-drain (queued-but-unexecuted
    actions, or a partially expanded cardinal move) would persist state that
    replay cannot reconstruct, so it raises ``ValueError`` instead.
    """
    if stepper.action_queue or stepper.primitive_buffer:
        raise ValueError(
            "save_checkpoint must be called at a query boundary "
            "(empty action_queue and primitive_buffer); "
            f"action_queue={stepper.action_queue!r} "
            f"primitive_buffer={stepper.primitive_buffer!r}"
        )
    path = Path(path)
    parts = parts_dir(path)
    payload = {
        # ``_json_safe`` drops the "_"-prefixed frames (PNG bytes, not JSON) and
        # coerces numpy scalars; frames are regenerated deterministically on the
        # post-resume steps and are not needed to replay the backend.
        "transcript": [_checkpoint_record(rec, parts) for rec in stepper.transcript],
        "seed": stepper.seed,
        "query_count": stepper.query_count,
        "parse_failures": stepper.parse_failures,
        "consecutive_failures": stepper.consecutive_failures,
        "finished": stepper.finished,
        "end_reason": stepper.end_reason,
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    # os.replace is atomic against process crash (no torn checkpoint); it is
    # NOT fsync'd, so power loss can still lose the write — out of scope of the
    # threat model (a lost checkpoint just re-runs the episode).
    os.replace(tmp, path)


def parts_dir(path: str | Path) -> Path:
    """Directory holding the checkpoint's write-once request and image files."""
    path = Path(path)
    return path.with_name(path.name + ".parts")


def discard_checkpoint(path: str | Path) -> None:
    """Remove a checkpoint and its parts directory."""
    Path(path).unlink(missing_ok=True)
    shutil.rmtree(parts_dir(path), ignore_errors=True)


def _checkpoint_record(rec: dict, parts: Path) -> dict:
    """JSON-safe record; a query's request is replaced by a reference to its part file.

    A query record never changes after it is appended, so its part is written
    once and remembered on the record ("_" key, never serialized).
    """
    if rec.get("kind") != "query" or not isinstance(rec.get("agent_messages"), list):
        return _json_safe(rec)
    out = _json_safe({k: v for k, v in rec.items() if k != "agent_messages"})
    cached = rec.get("_checkpoint_part")
    if cached is None or cached[0] != parts:
        messages = serialize_messages(
            rec["agent_messages"], parts, "", ImageFiles(parts)
        )
        text = json.dumps(_json_safe(messages), sort_keys=True)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        name = f"query_{rec['query_index']:03d}_{digest}.json"
        part = parts / name
        if not part.exists():
            parts.mkdir(parents=True, exist_ok=True)
            tmp = part.with_name(part.name + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, part)
        cached = rec["_checkpoint_part"] = (parts, name)
    out["agent_messages"] = {"part": cached[1]}
    return out


def _load_parts(transcript: list[dict], parts: Path) -> None:
    """Inline each referenced request again, exactly as it was sent."""
    cache: dict[Path, str] = {}
    for rec in transcript:
        ref = rec.get("agent_messages")
        if rec.get("kind") != "query" or not isinstance(ref, dict):
            continue
        part = parts / ref["part"]
        try:
            messages = json.loads(part.read_text(encoding="utf-8"))
            rec["agent_messages"] = deserialize_messages(messages, parts, cache)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"checkpoint part {part} is missing or corrupted: {exc}") from exc
        rec["_checkpoint_part"] = (parts, ref["part"])


# GridState fields serialized from sets. GridState.to_dict lists them sorted
# now, but older checkpoints and archives list them in their process's
# PYTHONHASHSEED order, so replay compares them as sets.
_SET_VALUED_STATE_FIELDS = (
    "open_doors", "collected_keys", "active_switches", "open_gates", "visible_cells", "explored_cells",
)


def _canonical_state(state):
    if not isinstance(state, dict):
        return state
    out = dict(state)
    for key in _SET_VALUED_STATE_FIELDS:
        if isinstance(out.get(key), list):
            out[key] = sorted(out[key], key=lambda v: json.dumps(v, sort_keys=True))
    return out


def resume_stepper(path: str | Path, *, runner) -> EpisodeStepper:
    """Rebuild an ``EpisodeStepper`` from a checkpoint via deterministic replay.

    ``runner`` is a FRESH ``ExperimentRunner`` (its backend not yet reset). The
    checkpoint's seed must match the runner's task spec. The returned stepper is
    positioned at the same query boundary the checkpoint was taken at: the next
    ``next_query()`` re-issues the round that was in flight when the checkpoint
    was written.
    """
    data = _load_checkpoint(path)

    if runner.config.chat_history != "stateless":
        raise ValueError(
            "resume_stepper only supports chat_history='stateless'; "
            f"got {runner.config.chat_history!r} (multi-turn chat state is not "
            "persisted in the checkpoint)"
        )

    seed = data["seed"]
    if seed != runner.task_spec.seed:
        raise ValueError(
            f"checkpoint seed {seed!r} does not match runner task_spec.seed "
            f"{runner.task_spec.seed!r} (wrong or stale checkpoint)"
        )

    transcript = data["transcript"]
    _load_parts(transcript, parts_dir(path))
    reset_rec = transcript[0]
    if reset_rec.get("kind") != "reset":
        raise ValueError("checkpoint transcript does not start with a reset record")

    # start() re-seeds the backend, seeds its own reset record, and builds the
    # system message + all initial fields exactly as a live episode would.
    stepper = EpisodeStepper(runner)
    stepper.start()
    reset_rgb = stepper._runner.last_rgb  # regenerated reset frame

    if _canonical_state(_json_safe(stepper.initial_state)) != _canonical_state(reset_rec.get("state")):
        raise ValueError(
            "checkpoint initial state does not match a fresh reset "
            "(stale checkpoint or mismatched task/seed)"
        )

    step_records = [r for r in transcript if r.get("kind") == "step"]
    query_records = [r for r in transcript if r.get("kind") == "query"]

    # Replay the recorded action tokens through the fresh backend, verifying each
    # outcome against the record, and rebuild stall-K state from the replayed
    # states using the SAME signature function the stepper uses. The frames
    # were dropped from the JSON checkpoint; they are regenerated here (replay is
    # deterministic) and re-attached so the reconstructed observation history is
    # byte-identical to an uninterrupted run.
    reset_rec["_reset_frame_png"] = stepper._png(reset_rgb)
    if stepper._runner.start_map_rgb is not None:  # regenerated by start() too
        reset_rec["_start_map_rgb"] = stepper._runner.start_map_rgb
    state = stepper.state
    stall_k = stepper.stall_k
    stall_watchdog = ProgressStallWatchdog(stall_k, state) if stall_k is not None else None
    for rec in step_records:
        action = rec["action"]
        decision_rgb = stepper._runner.last_rgb
        try:
            action_int = nlu_action_to_int(action)
        except ValueError:
            # An unconvertible token was a no-op step in the original loop (no
            # backend.step, position unchanged); mirror that and verify.
            if rec["position_after_row_col"] != rec["position_before_row_col"]:
                raise ValueError(
                    "corrupted checkpoint: unconvertible action "
                    f"{action!r} recorded a position change"
                )
            rec["_decision_frame_png"] = stepper._png(decision_rgb)
            rec["_post_step_png"] = stepper._png(decision_rgb)
            continue
        stepper._runner.last_rgb, _reward, terminated, truncated, state, _info = (
            stepper.backend.step(action_int)
        )
        rec["_decision_frame_png"] = stepper._png(decision_rgb)
        rec["_post_step_png"] = stepper._png(stepper._runner.last_rgb)
        if list(agent_row_col(state)) != rec["position_after_row_col"]:
            raise ValueError(
                "corrupted or stale checkpoint: replayed step "
                f"{rec.get('step_index')} ({action!r}) landed at "
                f"{list(agent_row_col(state))} but record has "
                f"{rec['position_after_row_col']}"
            )
        if _canonical_state(_json_safe(state_snapshot(state))) != _canonical_state(rec["state_after"]):
            raise ValueError(
                "corrupted or stale checkpoint: replayed step "
                f"{rec.get('step_index')} ({action!r}) diverged from recorded "
                "state_after"
            )
        if stall_watchdog and not rec["terminated"] and not rec["truncated"]:
            stall_watchdog.observe(state)

    # A query record exists per issued-and-applied query; the query in flight at
    # checkpoint time (if any) has no record yet. Roll query_count back to the
    # applied count so the next next_query() re-issues that in-flight round.
    completed = len(query_records)
    if data["query_count"] not in (completed, completed + 1):
        raise ValueError(
            "corrupted checkpoint: query_count "
            f"{data['query_count']} inconsistent with {completed} query records"
        )

    # Restore live state. Counters/transcript are restored verbatim; indices are
    # rederived from the transcript (they are overwritten by the next query).
    stepper.state = state
    stepper.transcript = transcript
    stepper.query_count = completed
    stepper.current_query_index = completed
    stepper.action_queue_index = 0
    stepper.step_index = len(step_records)
    stepper.parse_failures = data["parse_failures"]
    # Rederived from the last step record. At a parse-failure boundary this can
    # transiently differ for non-step_by_step querying modes (subgoal's
    # should_query consults failures); no effect for R1/step_by_step, where
    # should_query ignores it and the next query resets it to 0.
    stepper.consecutive_failures = (
        step_records[-1]["consecutive_failures_after"] if step_records else 0
    )
    stepper.action_queue = []
    stepper.primitive_buffer = []
    stepper._stall_watchdog = stall_watchdog
    stepper.end_reason = data["end_reason"]
    stepper._finished = bool(data["finished"])
    stepper.last_feedback = _reconstruct_last_feedback(
        stepper, step_records, data["parse_failures"]
    )
    return stepper


def _reconstruct_last_feedback(stepper, step_records, parse_failures):
    """The feedback string the next query's user message would carry.

    A pending parse retry (``parse_failures > 0``) means the last feedback was
    the parse-failure prompt; otherwise it is the last executed step's feedback,
    or the initial feedback if no step has run yet. An infra reply changes
    neither ``parse_failures`` nor the live feedback, so trailing infra query
    records need no case of their own.
    """
    if parse_failures > 0:
        return feedback_templates.PARSE_FAILURE_FEEDBACK.format(
            actions_hint=stepper.actions_hint
        )
    if step_records:
        return step_records[-1]["feedback"]
    return feedback_templates.INITIAL_FEEDBACK


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read checkpoint {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"corrupted checkpoint {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"corrupted checkpoint {path}: top-level object is not a dict")
    missing = [f for f in _FIELDS if f not in data]
    if missing:
        raise ValueError(f"checkpoint {path} missing fields: {missing}")
    if not isinstance(data["transcript"], list) or not data["transcript"]:
        raise ValueError(f"checkpoint {path} has an empty or malformed transcript")
    return data
