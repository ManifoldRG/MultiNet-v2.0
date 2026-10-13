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
import time
from pathlib import Path
from typing import Any, Callable, Optional

from interface.config import ExperimentConfig
from interface.episode_log import flush_episode_log
from gridworld.backends import get_backend
from gridworld.render_settings import RenderSettings
from interface.runner import build_runner
from gridworld.task_spec import TaskSpecification


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
    """
    from interface.episode_checkpoint import resume_stepper, save_checkpoint
    from interface.episode_step import EpisodeStepper

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "checkpoint.json"
    ledger = out_dir / "queries.jsonl"
    runner = build_episode_runner(task_source, config, seed, max_steps=max_steps, render=render)
    checkpointed = config.chat_history == "stateless"

    def at_query_boundary(stepper) -> None:
        if checkpointed:
            save_checkpoint(checkpoint, stepper)

    def after_reply(stepper, reply) -> None:
        line = {"query_index": stepper.query_count, "usage": reply.usage, "ts": time.time()}
        with ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")

    hooks = {"at_query_boundary": at_query_boundary, "after_reply": after_reply}
    if resume_from is None and checkpointed and checkpoint.exists() and not (out_dir / "episode.json").exists():
        # A previous process died mid-episode: continue from its last boundary.
        stepper = resume_stepper(checkpoint, runner=runner)
        stepper.maze_path = str(task_source)
        killed_at = stepper.query_count
        result = runner.drive(stepper, agent, **hooks)
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
