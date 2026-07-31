"""Phase-1 truncation scanner → phase-2 rerun manifest (two-tier Qwen rerun).

The two-tier rerun (``docs/qwen-two-tier-rerun-design.md``) runs *all* mazes at a
small token cap (8000) in phase 1, then re-runs only the mazes that actually hit
the cap at 64k in phase 2. This module reads phase-1 ``episode.json`` artifacts,
flags the cap-hitters, and writes the phase-2 rerun manifest (a verbatim subset
of the source manifest plus a selection/provenance block).

Detection (design §Detection & scan — load-bearing):
- Token usage lives on **query** records:
  ``episode.json → transcript[i where kind=="query"].usage.output_tokens``.
- An episode is flagged if ANY query record hit the cap. Qwen's truncated steps
  mostly still report ``parse_ok=True`` (the lenient parser salvages an action),
  so a parse-failure trigger would miss ~98% of truncations — the cap-hit
  trigger is the only one that catches the silent, degraded decisions.
- A step record's ``truncated`` field is ENV truncation and must NOT flag.
- The new Reply-based agents additionally stamp ``stop_reason`` /
  ``token_truncated`` on the query record; use them when present.
- The cap is NOT in ``episode.json``; it is read from the sibling
  ``run_inputs.json → model_config.max_tokens`` (unless ``--cap`` overrides).

Usage::

    python -m scripts.scan_truncations \
        --artifacts-root .runs/<run>/qwen_phase1 \
        --source-manifest gridworld/fixtures/manifest.r1_balanced_03.json \
        --out gridworld/fixtures/manifest.r1_qwen_phase2.json \
        [--cap 8000] [--model Qwen_Qwen3.6-27B]

``--model`` is the SANITIZED ``<model>`` run-dir segment
(``_sanitize(model_cfg["model"])`` from run_pipeline), NOT the run-config key
(``qwen36_27b_vllm``). A ``--model`` that matches zero run dirs exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    # Allow ``python scripts/scan_truncations.py`` (bare path) as well as -m.
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_pipeline import load_manifest  # noqa: E402

# Provider stop reasons that mean the output was cut off by the token cap.
_TRUNCATING_STOP_REASONS = ("length", "max_tokens")

# Trigger label recorded in the rerun manifest's provenance block.
TRIGGER = "output_tokens>=cap"


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def _query_records(episode: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        r
        for r in episode.get("transcript", [])
        if isinstance(r, dict) and r.get("kind") == "query"
    ]


def _query_output_tokens(record: dict[str, Any]) -> Optional[int]:
    usage = record.get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get("output_tokens")
    return value if isinstance(value, int) else None


def _query_cap_independent_truncated(record: dict[str, Any]) -> bool:
    """Truncation signals that need NO cap: explicit provider metadata stamped
    by the Reply-based agents. These must fire even when the cap is unresolvable
    (missing/unreadable ``run_inputs.json``)."""
    if record.get("token_truncated") is True:
        return True
    return record.get("stop_reason") in _TRUNCATING_STOP_REASONS


def _query_is_truncated(record: dict[str, Any], cap: int) -> bool:
    if _query_cap_independent_truncated(record):
        return True
    # Cap-compare fallback (the only signal legacy agents leave behind).
    output_tokens = _query_output_tokens(record)
    return output_tokens is not None and output_tokens >= cap


def episode_has_cap_independent_truncation(episode: dict[str, Any]) -> bool:
    """True iff any query record carries an explicit truncation stamp
    (``token_truncated`` / ``stop_reason``) — decidable without a cap."""
    return any(_query_cap_independent_truncated(r) for r in _query_records(episode))


def episode_is_truncated(episode: dict[str, Any], cap: int) -> bool:
    """True iff any ``kind=="query"`` record hit the token cap.

    A query is truncated when it carries ``token_truncated=True``, a
    ``stop_reason`` in ``("length", "max_tokens")``, or a reported
    ``usage.output_tokens >= cap``. Step-record ``truncated`` (env truncation)
    is deliberately ignored.
    """
    return any(_query_is_truncated(r, cap) for r in _query_records(episode))


def episode_queries_missing_usage(episode: dict[str, Any]) -> int:
    """Count query records with no usable ``usage.output_tokens`` — silently
    skipped by the cap-compare fallback; surfaced for operator visibility."""
    return sum(
        1 for r in _query_records(episode) if _query_output_tokens(r) is None
    )


def episode_max_output_tokens(episode: dict[str, Any]) -> int:
    values = [
        t for t in (_query_output_tokens(r) for r in _query_records(episode))
        if t is not None
    ]
    return max(values) if values else 0


# --------------------------------------------------------------------------- #
# Scan
# --------------------------------------------------------------------------- #
def _resolve_cap(run_dir: Path, override: Optional[int]) -> Optional[int]:
    if override is not None:
        return override
    sidecar = run_dir / "run_inputs.json"
    if not sidecar.exists():
        return None
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    cap = data.get("model_config", {}).get("max_tokens")
    return cap if isinstance(cap, int) else None


def _model_segment(episode_path: Path) -> str:
    # runs/<task>/<backend>/<model>/seed_<n>/<variant>/episode.json
    # parents: [0]=<variant> [1]=seed_<n> [2]=<model> [3]=<backend> [4]=<task>
    return episode_path.parents[2].name


def _task_id(run_dir: Path, episode: dict[str, Any]) -> str:
    sidecar = run_dir / "run_inputs.json"
    if sidecar.exists():
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        if data.get("task_id"):
            return str(data["task_id"])
    spec = episode.get("task_spec")
    if isinstance(spec, dict) and spec.get("task_id"):
        return str(spec["task_id"])
    # runs/<task>/<backend>/<model>/seed_<n>/<variant>
    return run_dir.parents[3].name


def scan_runs(
    artifacts_root: Path,
    *,
    cap: Optional[int] = None,
    model: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Walk ``runs/**/episode.json`` and flag cap-hitting episodes.

    ``cap`` is resolved per-run from the sibling ``run_inputs.json``
    (``model_config.max_tokens``) unless an explicit ``cap`` is given, which
    overrides every run. ``model`` filters to run dirs whose ``<model>`` path
    segment matches (Claude/Kimi runs in a shared root are ignored).

    Returns one dict per run dir::

        {"task_id", "run_dir", "max_output_tokens", "flagged"}
    """
    artifacts_root = Path(artifacts_root)
    runs_root = artifacts_root / "runs"
    results: list[dict[str, Any]] = []
    if not runs_root.exists():
        return results
    for episode_path in sorted(runs_root.rglob("episode.json")):
        if model is not None and _model_segment(episode_path) != model:
            continue
        run_dir = episode_path.parent
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        resolved_cap = _resolve_cap(run_dir, cap)

        # Cap-independent signals (explicit provider stamps) always decide, even
        # when the cap is unresolvable — a length-stamped query flags regardless.
        if resolved_cap is not None:
            flagged = episode_is_truncated(episode, resolved_cap)
            status = "flagged" if flagged else "ok"
        elif episode_has_cap_independent_truncation(episode):
            flagged, status = True, "flagged"
        else:
            # Cap unresolved AND no explicit stamp: cannot judge the cap-compare
            # signal. Fail-closed — surfaced as its own status, never silently ok.
            flagged, status = False, "cap_unresolved"

        results.append(
            {
                "task_id": _task_id(run_dir, episode),
                "run_dir": str(run_dir),
                "max_output_tokens": episode_max_output_tokens(episode),
                "flagged": bool(flagged),
                "status": status,
                "cap": resolved_cap,
                "queries_missing_usage": episode_queries_missing_usage(episode),
            }
        )
    return results


def effective_cap(results: list[dict[str, Any]]) -> Optional[int]:
    """The single cap the scanned runs shared, or None if mixed/unknown.

    Used to record a meaningful ``cap`` in the rerun manifest's provenance when
    no explicit ``--cap`` was given (phase-1 runs are uniform at 8000).
    """
    caps = {r.get("cap") for r in results if r.get("cap") is not None}
    return next(iter(caps)) if len(caps) == 1 else None


# --------------------------------------------------------------------------- #
# Rerun manifest
# --------------------------------------------------------------------------- #
def write_rerun_manifest(
    flagged: list[dict[str, Any]],
    source_manifest: Path,
    out_path: Path,
    *,
    cap: Optional[int] = None,
) -> dict[str, Any]:
    """Emit the phase-2 rerun manifest: the flagged subset of the source
    manifest's task rows (preserved verbatim) plus a selection/provenance block.

    ``flagged`` may be the full ``scan_runs`` output (entries with
    ``flagged=False`` are ignored) or an already-filtered list. Task rows are
    copied verbatim from ``source_manifest`` and kept in source order, so the
    output loads and validates through the same manifest loader as the source.
    """
    source_manifest = Path(source_manifest)
    out_path = Path(out_path)

    flagged_ids = {
        entry["task_id"]
        for entry in flagged
        if entry.get("flagged", True) and entry.get("task_id")
    }
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    source_rows = source["tasks"] if isinstance(source, dict) else source

    subset = [row for row in source_rows if row.get("task_id") in flagged_ids]

    manifest: dict[str, Any] = {
        "description": (
            "Phase-2 rerun subset (Qwen two-tier): mazes that hit the phase-1 "
            "token cap, re-run at the 64k ceiling."
        ),
        "selection": {
            "derived_from": str(source_manifest),
            "trigger": TRIGGER,
            "cap": cap,
            "maze_count": len(subset),
        },
        "tasks": subset,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------- #
# Summary + CLI
# --------------------------------------------------------------------------- #
def _summary_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse per-run results to one row per task_id.

    ``flagged`` is any-run-flagged. ``cap_unresolved`` is set when a task has a
    cap_unresolved run that never got overridden by a flag (a task flagged on one
    seed but cap_unresolved on another is still flagged, not silently dropped)."""
    by_task: dict[str, dict[str, Any]] = {}
    for r in results:
        agg = by_task.setdefault(
            r["task_id"], {"task_id": r["task_id"], "flagged": False,
                           "cap_unresolved": False, "max_output_tokens": 0,
                           "run_count": 0}
        )
        agg["flagged"] = agg["flagged"] or r["flagged"]
        agg["cap_unresolved"] = (
            agg["cap_unresolved"] or r.get("status") == "cap_unresolved"
        )
        agg["max_output_tokens"] = max(agg["max_output_tokens"], r["max_output_tokens"])
        agg["run_count"] += 1
    return sorted(by_task.values(), key=lambda a: a["task_id"])


def _status_label(row: dict[str, Any]) -> str:
    if row["flagged"]:
        return "FLAG"
    if row["cap_unresolved"]:
        return "?CAP"
    return "-"


def print_summary(results: list[dict[str, Any]]) -> None:
    rows = _summary_rows(results)
    flagged_tasks = sum(1 for r in rows if r["flagged"])
    unresolved_runs = sum(1 for r in results if r.get("status") == "cap_unresolved")
    missing_usage = sum(r.get("queries_missing_usage", 0) for r in results)
    header = f"{'task_id':<52} {'flag':>5} {'max_out':>8} {'runs':>5}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['task_id']:<52} {_status_label(r):>5} "
            f"{r['max_output_tokens']:>8} {r['run_count']:>5}"
        )
    print("-" * len(header))
    print(
        f"tasks: {len(rows)}  flagged: {flagged_tasks}  "
        f"not-flagged: {len(rows) - flagged_tasks}  "
        f"(runs scanned: {len(results)})"
    )
    print(
        f"cap_unresolved runs: {unresolved_runs}   "
        f"query records with no usable usage: {missing_usage}"
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan phase-1 episodes for token-cap truncation and emit a "
        "phase-2 rerun manifest (two-tier Qwen rerun)."
    )
    parser.add_argument("--artifacts-root", required=True,
                        help="Phase-1 artifacts root (contains runs/).")
    parser.add_argument("--source-manifest", required=True,
                        help="Manifest the flagged rows are copied from.")
    parser.add_argument("--out", help="Path to write the phase-2 rerun manifest.")
    parser.add_argument("--cap", type=int, default=None,
                        help="Override the per-run cap (default: resolve from "
                        "each run_inputs.json model_config.max_tokens).")
    parser.add_argument("--model", default=None,
                        help="Filter run dirs by <model> path segment.")
    parser.add_argument("--allow-unresolved", action="store_true",
                        help="Exit 0 even when some runs have an unresolvable "
                        "cap (default: fail closed — this is a money-deciding tool).")
    args = parser.parse_args(argv)

    results = scan_runs(Path(args.artifacts_root), cap=args.cap, model=args.model)

    # Fail closed on a zero-match --model filter. The run-dir <model> segment is
    # the SANITIZED model id (e.g. 'Qwen_Qwen3.6-27B'), not the run-config key
    # (e.g. 'qwen36_27b_vllm'); a filter typo silently scans 0 runs, flags
    # nothing, and would emit an empty phase-2 rerun manifest — shipping
    # truncated episodes as final data. Never let a mis-typed filter look "clean".
    if args.model is not None and not results:
        print(
            f"ERROR: --model {args.model!r} matched ZERO run dirs under "
            f"{args.artifacts_root}/runs. The <model> path segment is the "
            "sanitized model id (e.g. 'Qwen_Qwen3.6-27B'), not the run-config "
            "key. Refusing to emit an empty rerun manifest from an empty scan.",
            file=sys.stderr,
        )
        return 2

    print_summary(results)

    if args.out:
        flagged = [r for r in results if r["flagged"]]
        cap_for_provenance = args.cap if args.cap is not None else effective_cap(results)
        manifest = write_rerun_manifest(
            flagged, Path(args.source_manifest), Path(args.out),
            cap=cap_for_provenance,
        )
        print(f"\nWrote {len(manifest['tasks'])} flagged task(s) -> {args.out}")

    unresolved = [r for r in results if r.get("status") == "cap_unresolved"]
    if unresolved:
        print(
            f"WARNING: {len(unresolved)} run(s) had an unresolvable token cap "
            "(missing/invalid run_inputs.json model_config.max_tokens and no "
            "explicit truncation stamp); these could NOT be judged and are NOT "
            "in the rerun manifest. Pass --cap or --allow-unresolved.",
            file=sys.stderr,
        )
        for r in unresolved:
            print(f"  cap_unresolved: {r['run_dir']}", file=sys.stderr)
        if not args.allow_unresolved:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
