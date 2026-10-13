"""Stage 3 — runtime runs on the ``interface/`` stack (Stack A, live models).

Builds the state backend (MiniGrid, optionally drawn by the 3D render layer —
see ``RenderSettings``) + ``ExperimentRunner`` for one task, runs a single
episode with a live-model agent, and flushes the canonical ``episode.json``
artifact (plus PNG frames). Baselines are NOT run here — they feed Stage-2
difficulty/canonical paths via the scorer.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

from interface.config import ExperimentConfig
from interface.episode_log import flush_episode_log
from gridworld.backends import get_backend
from gridworld.render_settings import RenderSettings
from interface.runner import build_runner
from gridworld.task_spec import TaskSpecification

logger = logging.getLogger(__name__)


# An agent is any callable mapping chat messages -> model text (optionally
# exposing a ``last_usage`` attribute for token telemetry).
Agent = Callable[[list[dict]], str]


def _spec_with_seed(spec: TaskSpecification, seed: int) -> TaskSpecification:
    """Return a copy of ``spec`` with ``seed`` overridden (runner seeds from it)."""
    if spec.seed == seed:
        return spec
    return dataclasses.replace(spec, seed=seed)


def build_episode_runner(
    task_source: str | Path,
    config: ExperimentConfig,
    seed: int,
    *,
    max_steps: int | None = None,
    render: RenderSettings | None = None,
):
    """Build a configured ``ExperimentRunner`` for one task (no episode run yet).

    Shared by the serial ``run_episode`` (below) and the batch-API lockstep worker,
    which drives the runner through an ``EpisodeStepper`` instead of ``runner.run``.
    Keeping the backend/spec/seed/max_steps wiring in one place ensures both paths
    build an identical runner. ``render`` selects the backend/camera/resolution
    that draws the frames (2D MiniGrid by default).
    """
    render = render or RenderSettings()
    # Parse once and configure once: a 3D configure compiles the MjModel and
    # opens a GL context, so it must not also run on the pre-seed spec.
    spec = TaskSpecification.from_json(str(task_source))
    backend = get_backend(render.backend, **render.backend_kwargs(spec))
    spec = _spec_with_seed(spec, seed)
    if max_steps is not None and spec.max_steps != max_steps:
        spec = dataclasses.replace(spec, max_steps=int(max_steps))
    backend.configure(spec)
    return build_runner(config, backend, spec)


def run_episode(
    task_source: str | Path,
    config: ExperimentConfig,
    agent: Agent,
    seed: int,
    out_dir: str | Path,
    *,
    max_steps: int | None = None,
    provenance: Optional[dict[str, Any]] = None,
    render: RenderSettings | None = None,
    resume_from: Optional[dict[str, Any]] = None,
    fresh: bool = False,
) -> dict[str, Any]:
    """Run one episode and flush ``episode.json`` into ``out_dir``.

    Returns the in-memory episode dict (the JSON-safe payload written to
    ``out_dir/episode.json``), so callers can derive metrics without re-reading.

    ``provenance`` is additive phase/caps metadata (Task B3: ``pass``,
    ``phase_label``, ``max_tokens``, ``max_model_len``) stamped onto the runner
    result so ``flush_episode_log`` persists it on ``episode.json``. It is never
    part of any input hash.

    Item 17: a stateless episode checkpoints ``out_dir/checkpoint.json`` at every
    query boundary and every episode appends one usage line per query to
    ``out_dir/queries.jsonl``, so a process killed mid-episode leaves a
    resumable partial and its spend record; the next run of the same unit
    resumes from that checkpoint. Both go once ``episode.json`` is written.
    The checkpoint is used only if it was written for these exact inputs and
    this code (``checkpoint_meta.json``) and replays cleanly; otherwise it is
    discarded and the episode starts fresh. ``fresh`` (``--force``) discards it.
    """
    from interface import episode_checkpoint
    from interface.episode_step import EpisodeStepper
    from pipeline import resume as resume_mod
    from scorer.io import stable_hash

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "checkpoint.json"
    meta_path = out_dir / "checkpoint_meta.json"
    ledger = out_dir / "queries.jsonl"
    runner = build_episode_runner(task_source, config, seed, max_steps=max_steps, render=render)
    checkpointed = config.chat_history == "stateless"
    meta = {
        "inputs": stable_hash({
            "config": config.to_dict(),
            "task": Path(task_source).read_text(encoding="utf-8"),
            "seed": int(seed),
            "max_steps": max_steps,
            "render": (render or RenderSettings()).label,
        }),
        "code_sha": resume_mod.current_code_sha(),
    }

    def at_query_boundary(stepper) -> None:
        if not checkpointed:
            return
        try:
            episode_checkpoint.save_checkpoint(checkpoint, stepper)
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
        except (OSError, ValueError) as exc:  # a bad disk must not end a paid episode
            logger.warning("checkpoint write failed (the episode goes on): %s", exc)

    def after_reply(stepper, reply) -> None:
        line = {"query_index": stepper.query_count, "usage": reply.usage, "ts": time.time()}
        try:
            with ledger.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line) + "\n")
        except OSError as exc:
            logger.warning("usage ledger write failed (the episode goes on): %s", exc)

    def usable_checkpoint() -> Optional[Any]:
        """The stepper resumed from a crash checkpoint of this exact run, else None
        (and any other checkpoint is discarded)."""
        if resume_from is not None or not checkpointed or not checkpoint.exists():
            return None
        if (out_dir / "episode.json").exists():
            return None
        reason = None
        try:
            recorded = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            recorded = None
        if fresh:
            reason = "fresh run requested"
        elif recorded != meta or meta["code_sha"] in ("unknown", "unrecorded"):
            reason = "it was written for other inputs or code"
        if reason is None:
            try:
                stepper = episode_checkpoint.resume_stepper(checkpoint, runner=runner)
                stepper.maze_path = str(task_source)
                return stepper
            except ValueError as exc:
                reason = f"it does not replay: {exc}"
        logger.warning("discarding the crash checkpoint in %s (%s); starting the episode fresh", out_dir, reason)
        checkpoint.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        return None

    hooks = {"at_query_boundary": at_query_boundary, "after_reply": after_reply}
    resumed_stepper = usable_checkpoint()
    if resumed_stepper is not None:
        # A previous process died mid-episode: continue from its last boundary.
        killed_at = resumed_stepper.query_count
        result = runner.drive(resumed_stepper, agent, **hooks)
        provenance = {**(provenance or {}), "resumed_from": {"checkpoint": True, "killed_at_query": killed_at}}
    elif resume_from is None:
        stepper = EpisodeStepper(runner, verbose=False, maze_path=str(task_source))
        stepper.start()
        result = runner.drive(stepper, agent, **hooks)
    else:
        # ``resume_from``: {archive_dir, expected_inputs_hash, resume_count, label}
        # (pipeline.resume): replay-verify the killed run, then continue it.
        from pipeline import resume

        stepper, resume_provenance = resume.resumed_stepper(
            runner, Path(resume_from["archive_dir"]),
            expected_inputs_hash=resume_from.get("expected_inputs_hash"),
            resume_count=int(resume_from.get("resume_count", 1)),
            label=str(resume_from.get("label", "")),
        )
        stepper.maze_path = str(task_source)
        result = runner.drive(stepper, agent, **hooks)
        provenance = {**(provenance or {}), **resume_provenance}
    if provenance:
        for key, value in provenance.items():
            if value is not None:
                result[key] = value

    episode_path = flush_episode_log(result, out_dir)  # rewrites out_dir: checkpoint + ledger go
    return json.loads(episode_path.read_text(encoding="utf-8"))
