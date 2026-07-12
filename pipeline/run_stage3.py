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
from typing import Any, Callable

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


def run_episode(
    task_source: str | Path,
    config: ExperimentConfig,
    agent: Agent,
    seed: int,
    out_dir: str | Path,
    backend: str = "minigrid",
) -> dict[str, Any]:
    """Run one episode and flush ``episode.json`` into ``out_dir``.

    Returns the in-memory episode dict (the JSON-safe payload written to
    ``out_dir/episode.json``), so callers can derive metrics without re-reading.
    """
    backend_obj, spec = load_task(task_source, backend=backend)
    seeded_spec = _spec_with_seed(spec, seed)
    if seeded_spec is not spec:
        # Only re-configure (rebuilds the backend's env from scratch — cheap
        # for MiniGrid, but a real MuJoCo XML recompile for OgbenchBackend)
        # when the seed actually changed the spec load_task() already
        # configured with.
        backend_obj.configure(seeded_spec)
    spec = seeded_spec

    runner = build_runner(config, backend_obj, spec)
    result = runner.run(agent, verbose=False, maze_path=str(task_source))

    out_dir = Path(out_dir)
    episode_path = flush_episode_log(result, out_dir)
    return json.loads(episode_path.read_text(encoding="utf-8"))
