"""Stage 3 — runtime runs on the ``interface/`` stack (Stack A, live models).

Builds a MiniGrid backend + ``ExperimentRunner`` for one task, runs a single
episode with a live-model agent, and flushes the canonical ``episode.json``
artifact (plus PNG frames). Baselines are NOT run here — they feed Stage-2
difficulty/canonical paths via the scorer.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Callable, Optional

from interface.config import ExperimentConfig
from interface.episode_log import flush_episode_log
from interface.loader import load_task
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
):
    """Build a configured ``ExperimentRunner`` for one task (no episode run yet).

    Shared by the serial ``run_episode`` (below) and the batch-API lockstep worker,
    which drives the runner through an ``EpisodeStepper`` instead of ``runner.run``.
    Keeping the backend/spec/seed/max_steps wiring in one place ensures both paths
    build an identical runner.
    """
    backend, spec = load_task(task_source)
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
) -> dict[str, Any]:
    """Run one episode and flush ``episode.json`` into ``out_dir``.

    Returns the in-memory episode dict (the JSON-safe payload written to
    ``out_dir/episode.json``), so callers can derive metrics without re-reading.

    ``provenance`` is additive phase/caps metadata (Task B3: ``pass``,
    ``phase_label``, ``max_tokens``, ``max_model_len``) stamped onto the runner
    result so ``flush_episode_log`` persists it on ``episode.json``. It is never
    part of any input hash.
    """
    runner = build_episode_runner(task_source, config, seed, max_steps=max_steps)
    result = runner.run(agent, verbose=False, maze_path=str(task_source))
    if provenance:
        for key, value in provenance.items():
            if value is not None:
                result[key] = value

    out_dir = Path(out_dir)
    episode_path = flush_episode_log(result, out_dir)
    return json.loads(episode_path.read_text(encoding="utf-8"))
