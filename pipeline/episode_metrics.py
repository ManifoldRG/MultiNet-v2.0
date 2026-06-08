"""Stage-3 instrumentation: derive test-2/test-3 signals from an episode log.

Pure post-processing over the ``interface/`` runner's ``episode.json`` (the dict
returned by ``ExperimentRunner.run`` and flushed by ``flush_episode_log``), the
task spec, the canonical paths, and the manifest row. No runner edits required:
each ``kind == "step"`` transcript record already carries ``event_type`` and a
``state_after`` snapshot with the mechanism id sets and agent ``(x, y)`` position.

Coordinate convention: positions here are ``(x, y)`` taken from
``state_after.agent_position`` (NOT the ``(row, col)`` ``position_after`` field),
matching the planner positions in ``canonical_paths.json``.
"""

from __future__ import annotations

from typing import Any, Optional

# Mechanism id sets carried on every state snapshot, in direct-actuation priority
# order (keys/switches the agent acts on, then doors/gates that open as effects).
_MECHANISM_FIELDS = ("collected_keys", "active_switches", "open_doors", "open_gates")


def _position(state: Any) -> Optional[tuple[int, int]]:
    if not isinstance(state, dict):
        return None
    raw = state.get("agent_position") or state.get("position")
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return int(raw[0]), int(raw[1])
    return None


def _mechanism_sets(state: Any) -> dict[str, set[str]]:
    state = state if isinstance(state, dict) else {}
    return {field: set(state.get(field, []) or []) for field in _MECHANISM_FIELDS}


def _step_records(episode: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        rec
        for rec in episode.get("transcript", [])
        if isinstance(rec, dict) and rec.get("kind") == "step"
    ]


def visited_cells(episode: dict[str, Any]) -> list[tuple[int, int]]:
    """Ordered agent cells (x, y), consecutive duplicates collapsed."""
    cells: list[tuple[int, int]] = []
    initial = _position(episode.get("initial_state"))
    if initial is not None:
        cells.append(initial)
    for rec in episode.get("transcript", []):
        if not isinstance(rec, dict):
            continue
        if rec.get("kind") == "reset":
            pos = _position(rec.get("state"))
        elif rec.get("kind") == "step":
            pos = _position(rec.get("state_after"))
        else:
            continue
        if pos is not None:
            cells.append(pos)
    final = _position(episode.get("final_state"))
    if final is not None:
        cells.append(final)

    deduped: list[tuple[int, int]] = []
    for pos in cells:
        if not deduped or deduped[-1] != pos:
            deduped.append(pos)
    return deduped


def mechanism_interaction_order(episode: dict[str, Any]) -> list[str]:
    """Ordered mechanism ids in the order the agent first engaged each one.

    Walks the step records and diffs the ``state_after`` mechanism id sets
    against the previous step; newly-added ids are appended in field-priority
    order (keys, switches, doors, gates) so a single switch toggle that also
    opens a gate records the switch before its downstream gate.
    """
    order: list[str] = []
    seen: set[str] = set()
    prev = _mechanism_sets(episode.get("initial_state"))
    for rec in _step_records(episode):
        current = _mechanism_sets(rec.get("state_after"))
        for field in _MECHANISM_FIELDS:
            for mech_id in sorted(current[field] - prev[field]):
                if mech_id not in seen:
                    seen.add(mech_id)
                    order.append(mech_id)
        prev = current
    return order


def failure_point(
    episode: dict[str, Any],
    expected_mechanisms: list[str],
    mech_order: list[str],
) -> Optional[dict[str, Any]]:
    """First expected mechanism the agent never engaged, with context.

    Returns ``None`` for successful runs. For failed runs, reports the first id
    in ``expected_mechanisms`` missing from ``mech_order`` (``None`` if all were
    engaged but the run still failed), the runner ``end_reason``, the final cell,
    and the engaged-mechanism order for diagnostics.
    """
    if episode.get("success"):
        return None
    engaged = set(mech_order)
    missing = [m for m in expected_mechanisms if m not in engaged]
    cells = visited_cells(episode)
    return {
        "mechanism": missing[0] if missing else None,
        "end_reason": episode.get("end_reason"),
        "final_cell": list(cells[-1]) if cells else None,
        "engaged": list(mech_order),
        "missing": missing,
    }


def path_choice(
    episode: dict[str, Any],
    route_short_cells: Optional[list[Any]],
    route_long_cells: Optional[list[Any]],
) -> Optional[str]:
    """Classify which test-2 route the agent committed to.

    ``route_*_cells`` are discriminator cells unique to each route (cached in the
    manifest by ``validate_fixtures``). Returns ``"short_mech"``, ``"long_open"``,
    ``"mixed"``, or ``"none"``; ``None`` when no route cells are defined (non-test-2).
    """
    if not route_short_cells and not route_long_cells:
        return None
    cells = set(visited_cells(episode))
    short = {tuple(c) for c in (route_short_cells or [])}
    long = {tuple(c) for c in (route_long_cells or [])}
    hit_short = bool(short & cells)
    hit_long = bool(long & cells)
    if hit_short and not hit_long:
        return "short_mech"
    if hit_long and not hit_short:
        return "long_open"
    if hit_short and hit_long:
        return "mixed"
    return "none"


def episode_token_count(episode: dict[str, Any]) -> Optional[int]:
    """Sum token usage over ``kind == "query"`` transcript records."""
    from interface.telemetry import token_count_from_record

    total = 0
    found = False
    for rec in episode.get("transcript", []):
        if not isinstance(rec, dict) or rec.get("kind") != "query":
            continue
        count = token_count_from_record(rec)
        if count is not None:
            total += count
            found = True
    return total if found else None


def _canonical_optimal_steps(canonical_paths: dict[str, Any]) -> Optional[int]:
    bfs = canonical_paths.get("bfs", canonical_paths)
    if isinstance(bfs, dict) and bfs.get("optimal_steps") is not None:
        return int(bfs["optimal_steps"])
    if canonical_paths.get("optimal_steps") is not None:
        return int(canonical_paths["optimal_steps"])
    return None


def _episode_reward(episode: dict[str, Any]) -> Any:
    """Final-state reward, guarding an explicit ``final_state: null``."""
    final = episode.get("final_state")
    return final.get("reward") if isinstance(final, dict) else None


def build_metrics(
    episode: dict[str, Any],
    canonical_paths: dict[str, Any],
    manifest_row: dict[str, Any],
) -> dict[str, Any]:
    """Derive the test-specific signals shared by the run row and the scorer."""
    mech_order = mechanism_interaction_order(episode)
    expected = list(manifest_row.get("expected_mechanisms", []) or [])
    return {
        "mechanism_interaction_order": mech_order,
        "failure_point": failure_point(episode, expected, mech_order),
        "path_choice": path_choice(
            episode,
            manifest_row.get("route_short_cells"),
            manifest_row.get("route_long_cells"),
        ),
    }


def build_run_row(
    episode: dict[str, Any],
    canonical_paths: dict[str, Any],
    manifest_row: dict[str, Any],
    *,
    agent_or_model: str,
    seed: int,
    backend: str = "minigrid",
    raw_output_ref: Optional[str] = None,
    metrics: Optional[dict[str, Any]] = None,
    prompt_variant: str = "default",
) -> dict[str, Any]:
    """Build one ``episode_runs.jsonl`` row (Appendix A.3 fields).

    ``condition`` is the task-intrinsic axis (e.g. the test-3 mechanism order);
    ``prompt_variant`` is the orthogonal prompt axis selected by ``--conditions``.
    The two are kept distinct so prompt variants do not collapse onto the
    manifest condition.
    """
    metrics = metrics if metrics is not None else build_metrics(episode, canonical_paths, manifest_row)
    success = bool(episode.get("success"))
    end_reason = episode.get("end_reason")
    steps = int(episode.get("steps_used", 0))
    optimal_steps = _canonical_optimal_steps(canonical_paths)
    # Mirror scorer.runtime's step_ratio: optimal_steps == 0 is a perfect 0-step
    # solve, not a zero ratio, so the jsonl and run_score.json agree.
    if not success or optimal_steps is None:
        optimality_ratio = 0.0
    elif optimal_steps == 0:
        optimality_ratio = 1.0 if steps == 0 else 0.0
    else:
        optimality_ratio = optimal_steps / max(steps, optimal_steps)
    return {
        "task_id": manifest_row.get("task_id") or episode.get("task_spec", {}).get("task_id"),
        "experiment": manifest_row.get("experiment"),
        "condition": manifest_row.get("condition"),
        "prompt_variant": prompt_variant,
        "backend": backend,
        "agent_or_model": agent_or_model,
        "seed": seed,
        "success": success,
        "terminated": end_reason == "success",
        "truncated": end_reason == "truncated",
        "reward": _episode_reward(episode),
        "steps": steps,
        "optimal_steps": optimal_steps,
        "optimality_ratio": optimality_ratio,
        "path_choice": metrics["path_choice"],
        "mechanism_interaction_order": metrics["mechanism_interaction_order"],
        "failure_point": metrics["failure_point"],
        "tokens": episode_token_count(episode),
        "raw_output_ref": raw_output_ref,
    }


def enrich_run_for_scoring(
    episode: dict[str, Any],
    manifest_row: dict[str, Any],
    *,
    agent_or_model: str,
    seed: int,
    backend: str = "minigrid",
    metrics: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Episode dict + the fields ``scorer.compute_runtime_score`` reads/passes through.

    The scorer already understands the episode transcript (success, steps,
    positions, query-record token usage); this layers on the run identity and
    the derived test-2/test-3 signals so they flow into ``run_score.json``.
    """
    metrics = metrics if metrics is not None else build_metrics(episode, {}, manifest_row)
    run = dict(episode)
    run["task_id"] = manifest_row.get("task_id") or episode.get("task_spec", {}).get("task_id")
    run["backend"] = backend
    run["adapter"] = agent_or_model
    run["agent_or_model"] = agent_or_model
    run["model_id"] = agent_or_model
    run["seed"] = seed
    run["terminated"] = episode.get("end_reason") == "success"
    run["truncated"] = episode.get("end_reason") == "truncated"
    # episode_log nests reward under final_state; the scorer only reads a
    # top-level ``reward``, so lift it (keeps run_score.json reward in sync
    # with the episode_runs.jsonl row).
    if run.get("reward") is None:
        run["reward"] = _episode_reward(episode)
    for key in ("path_choice", "mechanism_interaction_order", "failure_point"):
        if metrics.get(key) is not None:
            run[key] = metrics[key]
    return run
