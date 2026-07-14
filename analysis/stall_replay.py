"""Frozen-replay stall calibration.

`replay_stall` answers, over an already-collected corpus of episode
transcripts, "what would have happened had the progress-stall watchdog
(``interface/runner.py``'s ``_progress_signature`` + K-streak rule) been
active during these runs?" It never re-runs an episode — it walks each
episode's recorded ``state_after`` snapshots in order and approximates the
runner's counting rule (seed ``seen_signatures`` with the initial state's
signature; a signature already seen increments a streak counter, a novel one
resets it to 0; the watchdog "fires" the first time the streak reaches ``K``).

Two known omissions: (1) the runner gates its watchdog check on
``not terminated and not truncated`` (skips the stall count on terminal/
truncated steps), and (2) the replay walks all ``kind == "step"`` records
with a ``state_after``, whereas the runner does not count pre-backend
unparseable-action records toward the streak.

This is a decision aid for picking a calibration table across candidate K
values — NOT a universal-safety assertion that any given K is safe to
enable live. It reuses ``_progress_signature`` rather than reimplementing
it, so the signature stays a single source of truth (see CLAUDE.md).
"""

from __future__ import annotations

from typing import Any

from gridworld.backends.base import GridState
from interface.runner import _progress_signature


def _step_states(episode: dict[str, Any]) -> list[GridState]:
    """Ordered per-step ``state_after`` snapshots as ``GridState`` objects.

    Mirrors ``pipeline/episode_metrics.py``'s ``_step_records`` pattern:
    only ``kind == "step"`` transcript records carry a ``state_after``
    snapshot (the dict shape produced by ``interface/episode_log.py``'s
    ``state_snapshot`` — a superset of ``GridState.to_dict()``, which
    ``GridState.from_dict`` reads back losslessly).
    """
    states = []
    for record in episode.get("transcript", []):
        if not isinstance(record, dict) or record.get("kind") != "step":
            continue
        state_after = record.get("state_after")
        if isinstance(state_after, dict):
            states.append(GridState.from_dict(state_after))
    return states


def replay_stall(episodes: list[dict[str, Any]], K: int) -> dict[str, int]:
    """Replay a frozen episode corpus against the progress-stall watchdog rule.

    For each episode: seed ``seen_signatures`` with the initial state's
    signature (matching ``ExperimentRunner.run``), then walk the per-step
    ``state_after`` snapshots in order, incrementing a streak counter each
    time the signature was already seen and resetting it to 0 on a novel
    signature. The first step where the streak reaches ``K`` is the point
    the watchdog would have fired; an episode with no such step is left
    alone (it never contributes to any of the returned counts).

    Returns:
        A dict with:
        - ``eventual_wins_killed``: eventually-successful episodes
          (``episode["success"]`` is true) whose max non-novel streak
          reached ``K`` — episodes the watchdog would have cut short even
          though they went on to succeed.
        - ``failures_caught``: failed episodes whose max non-novel streak
          reached ``K``.
        - ``failed_primitive_steps_saved``: summed, over caught failures
          only, the number of primitive steps recorded after the point the
          watchdog would have fired (i.e. the steps early termination would
          have saved).
    """
    eventual_wins_killed = 0
    failures_caught = 0
    failed_primitive_steps_saved = 0

    for episode in episodes:
        initial_state = episode.get("initial_state")
        if not isinstance(initial_state, dict):
            continue

        seen_signatures = {_progress_signature(GridState.from_dict(initial_state))}
        stall_count = 0
        kill_index = None

        step_states = _step_states(episode)
        for index, state in enumerate(step_states, start=1):
            sig = _progress_signature(state)
            if sig in seen_signatures:
                stall_count += 1
            else:
                seen_signatures.add(sig)
                stall_count = 0
            if stall_count >= K:
                kill_index = index
                break

        if kill_index is None:
            continue

        if episode.get("success"):
            eventual_wins_killed += 1
        else:
            failures_caught += 1
            failed_primitive_steps_saved += len(step_states) - kill_index

    return {
        "eventual_wins_killed": eventual_wins_killed,
        "failures_caught": failures_caught,
        "failed_primitive_steps_saved": failed_primitive_steps_saved,
    }
