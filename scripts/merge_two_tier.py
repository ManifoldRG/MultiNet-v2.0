"""Later-pass-wins merge for the two-tier Qwen rerun.

The two-tier rerun (``docs/qwen-two-tier-rerun-design.md`` §Result reconciliation
& provenance) runs *all* mazes at a small token cap in **phase 1**, then re-runs
only the flagged mazes at the 64k ceiling in **phase 2**, writing each phase to a
SEPARATE artifacts root. This module produces the final combined result:

    final = phase-1 episodes for unflagged units
          + phase-2 episodes OVERWRITING phase-1 for flagged units

keyed by ``(task_id, model, seed, condition, prompt_variant)`` — one row per key,
merged EXPLICITLY (the run-dir convention has no phase segment, so relying on the
accidental same-path clobber would double-count / silently lose provenance).

Provenance stamping (B3 review — binding): the distributed workers do NOT
self-stamp ``pass`` onto ``episode.json``. The merge therefore stamps provenance
onto the ``episode.json`` COPIES it writes into ``out_root`` and NEVER mutates the
immutable source phase roots:

- phase-2 winners: ``pass=2`` + ``max_tokens`` / ``max_model_len`` (from that
  run's ``run_inputs.json`` ``model_config`` when absent) + ``truncated_at_ceiling``
  (``episode_is_truncated`` at the phase-2 cap, i.e. 64000).
- phase-1 copies: ``pass=1`` when absent.
- ``run_inputs.json`` / ``run_score.json`` copy verbatim.

Usage::

    python -m scripts.merge_two_tier \
        --phase1 .runs/<run>/qwen_phase1 \
        --phase2 .runs/<run>/qwen_phase2 \
        --out    .runs/<run>/qwen_merged \
        [--expected-rerun-manifest gridworld/fixtures/manifest.r1_qwen_phase2.json]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    # Allow ``python scripts/merge_two_tier.py`` (bare path) as well as -m.
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse Task 2's detection — DO NOT duplicate the truncation logic.
from scripts.scan_truncations import (  # noqa: E402
    episode_has_cap_independent_truncation,
    episode_is_truncated,
)

# Unit identity: the aggregation key shared by phase-1 and phase-2 rows.
_KEY_FIELDS = ("task_id", "agent_or_model", "seed", "condition", "prompt_variant")


class MergeError(RuntimeError):
    """Fail-closed merge violation (missing rerun, non-subset phase 2, ...)."""


# --------------------------------------------------------------------------- #
# Loading phase roots
# --------------------------------------------------------------------------- #
def _row_key(row: dict[str, Any]) -> tuple:
    return tuple(row.get(field) for field in _KEY_FIELDS)


def _load_rows(root: Path) -> list[dict[str, Any]]:
    """Read a phase root's aggregate ``episode_runs.jsonl`` rows (build_run_row
    shaped). This is the single source of unit identity + the winning-dir pointer
    (``raw_output_ref``)."""
    jsonl = root / "episode_runs.jsonl"
    if not jsonl.exists():
        raise MergeError(
            f"phase root {root} has no episode_runs.jsonl (was the phase "
            "aggregated by run_pipeline?)"
        )
    rows: list[dict[str, Any]] = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _run_dir_for_row(root: Path, row: dict[str, Any]) -> Path:
    """The run dir a row was produced from, via its ``raw_output_ref``
    (``runs/<task>/<backend>/<model>/seed_<n>/<variant>/episode.json``)."""
    ref = row.get("raw_output_ref")
    if not ref:
        raise MergeError(
            f"row {_row_key(row)!r} in {root} has no raw_output_ref; cannot "
            "locate its run dir to copy"
        )
    return (root / Path(ref)).parent


def _model_config(run_dir: Path) -> dict[str, Any]:
    sidecar = run_dir / "run_inputs.json"
    if not sidecar.exists():
        return {}
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    cfg = data.get("model_config")
    return cfg if isinstance(cfg, dict) else {}


# --------------------------------------------------------------------------- #
# Ceiling truncation
# --------------------------------------------------------------------------- #
def _ceiling_status(
    episode: dict[str, Any], model_config: dict[str, Any]
) -> tuple[Optional[bool], bool]:
    """Whether a phase-2 episode STILL hit the (64k) ceiling, and whether that was
    decidable.

    Returns ``(ceiling, resolved)``:
    - cap resolvable from ``model_config.max_tokens`` → ``(episode_is_truncated,
      True)``;
    - cap unresolvable but an explicit provider stamp (``stop_reason`` /
      ``token_truncated``) decides it → ``(True, True)``;
    - cap unresolvable AND no explicit stamp → ``(None, False)`` — undecidable.
      The caller fails closed (matching the scanner's ``cap_unresolved`` policy)
      rather than silently recording ``truncated_at_ceiling=False``.
    """
    cap = model_config.get("max_tokens")
    if isinstance(cap, int):
        return episode_is_truncated(episode, cap), True
    if episode_has_cap_independent_truncation(episode):
        return True, True
    return None, False


# --------------------------------------------------------------------------- #
# Copy + stamp
# --------------------------------------------------------------------------- #
def _copy_run_dir(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    # run_inputs.json / run_score.json (and any other sidecars) copy verbatim;
    # episode.json is re-stamped in place afterwards.
    shutil.copytree(src, dst)


def _stamp_episode_copy(
    dst_run_dir: Path,
    *,
    pass_num: int,
    model_config: dict[str, Any],
    ceiling: Optional[bool],
) -> None:
    """Stamp provenance onto the COPY of episode.json in ``dst_run_dir`` only.

    Phase-2 winners (``pass_num == 2``): ``pass`` set unconditionally, plus
    ``max_tokens`` / ``max_model_len`` filled from ``model_config`` when absent
    and ``truncated_at_ceiling``. Phase-1 copies (``pass_num == 1``): ``pass`` set
    only when absent (the source is the ground truth for everything else).
    """
    ep_path = dst_run_dir / "episode.json"
    episode = json.loads(ep_path.read_text(encoding="utf-8"))
    if pass_num == 2:
        episode["pass"] = 2
        if "max_tokens" not in episode and "max_tokens" in model_config:
            episode["max_tokens"] = model_config["max_tokens"]
        if "max_model_len" not in episode and "max_model_len" in model_config:
            episode["max_model_len"] = model_config["max_model_len"]
        if ceiling is not None:
            episode["truncated_at_ceiling"] = bool(ceiling)
    else:
        episode.setdefault("pass", pass_num)
    ep_path.write_text(json.dumps(episode, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #
def _expected_rerun_ids(manifest_path: Path) -> set[str]:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = data["tasks"] if isinstance(data, dict) else data
    return {row["task_id"] for row in rows if isinstance(row, dict) and row.get("task_id")}


def merge_phase_roots(
    phase1_root: Path,
    phase2_root: Path,
    out_root: Path,
    *,
    expected_rerun_manifest: Optional[Path] = None,
    allow_unresolved_cap: bool = False,
) -> dict[str, Any]:
    """Later-pass-wins merge of two phase artifacts roots.

    For every unit key ``(task_id, model, seed, condition, prompt_variant)`` the
    phase-2 run dir wins when present, else phase-1. Winning run dirs are copied
    into ``out_root`` preserving layout (``run_inputs.json`` / ``run_score.json``
    verbatim; ``episode.json`` re-stamped with provenance in the COPY only). The
    combined ``episode_runs.jsonl`` is rebuilt with exactly one row per key.

    Fail-closed:
    - any task_id listed in ``expected_rerun_manifest`` but absent from phase 2
      → :class:`MergeError` listing the missing ids;
    - any phase-2 unit key absent from phase 1 (phase 2 must be a subset)
      → :class:`MergeError` listing the offending task_ids;
    - any phase-2 winner whose ceiling cap is unresolvable AND that carries no
      explicit truncation stamp → :class:`MergeError` listing the offending run
      dirs (matches the scanner's ``cap_unresolved`` policy). Pass
      ``allow_unresolved_cap=True`` to instead record those task_ids under
      ``cap_unresolved`` in the summary and stderr-WARN.

    Returns (and writes to ``out_root/two_tier_merge.json``) the summary
    ``{"total", "from_phase2", "truncated_at_ceiling": [task_ids],
    "cap_unresolved": [task_ids]}``.
    """
    phase1_root = Path(phase1_root)
    phase2_root = Path(phase2_root)
    out_root = Path(out_root)

    p1_rows = {_row_key(r): r for r in _load_rows(phase1_root)}
    p2_rows = {_row_key(r): r for r in _load_rows(phase2_root)}

    # Subset guard: phase 2 re-runs a subset of phase-1 units.
    orphan_keys = [k for k in p2_rows if k not in p1_rows]
    if orphan_keys:
        ids = sorted({k[0] for k in orphan_keys})
        raise MergeError(
            "phase 2 contains unit(s) not present in phase 1 (phase 2 must be a "
            f"subset): {', '.join(ids)}"
        )

    # Expected-rerun guard: every flagged task must have actually re-run.
    if expected_rerun_manifest is not None:
        expected = _expected_rerun_ids(Path(expected_rerun_manifest))
        present = {k[0] for k in p2_rows}
        missing = sorted(expected - present)
        if missing:
            raise MergeError(
                "flagged task(s) missing from phase 2 (fail-closed; a flagged "
                f"maze that never re-ran must not silently keep phase 1): "
                f"{', '.join(missing)}"
            )

    # Pre-pass: resolve each phase-2 winner's ceiling status BEFORE writing any
    # output, so an undecidable cap fails closed with no partial merged root.
    phase2_status: dict[tuple, tuple[dict[str, Any], Optional[bool]]] = {}
    unresolved: list[tuple[str, str]] = []  # (task_id, run_dir)
    for key, row in p2_rows.items():
        src_dir = _run_dir_for_row(phase2_root, row)
        model_config = _model_config(src_dir)
        episode = json.loads((src_dir / "episode.json").read_text(encoding="utf-8"))
        ceiling, resolved = _ceiling_status(episode, model_config)
        phase2_status[key] = (model_config, ceiling)
        if not resolved:
            unresolved.append((row.get("task_id"), str(src_dir)))

    if unresolved and not allow_unresolved_cap:
        listing = "\n".join(f"  cap_unresolved: {d}" for _, d in unresolved)
        raise MergeError(
            "phase-2 winner(s) hit no resolvable ceiling cap "
            "(run_inputs.json model_config.max_tokens missing/invalid) and carry "
            "no explicit truncation stamp, so truncated_at_ceiling cannot be "
            "judged. Fail-closed (a money/interpretability-deciding flag must not "
            "silently default to False). Pass --allow-unresolved-cap to record "
            f"them under cap_unresolved instead:\n{listing}"
        )

    out_root.mkdir(parents=True, exist_ok=True)

    merged_rows: list[dict[str, Any]] = []
    from_phase2 = 0
    ceiling_task_ids: list[str] = []
    cap_unresolved_ids: list[str] = []

    for key in sorted(p1_rows, key=lambda k: tuple("" if v is None else str(v) for v in k)):
        if key in p2_rows:
            from_phase2 += 1
            src_root, row, pass_num = phase2_root, dict(p2_rows[key]), 2
        else:
            src_root, row, pass_num = phase1_root, dict(p1_rows[key]), 1

        src_dir = _run_dir_for_row(src_root, row)
        # Copy under the winning row's raw_output_ref so the merged layout is
        # identical regardless of source phase.
        dst_dir = _run_dir_for_row(out_root, row)
        _copy_run_dir(src_dir, dst_dir)

        model_config: dict[str, Any] = {}
        ceiling: Optional[bool] = None
        row["pass"] = pass_num
        if pass_num == 2:
            model_config, ceiling = phase2_status[key]
            if ceiling is None:
                # Undecidable ceiling (allow_unresolved_cap): leave
                # truncated_at_ceiling ABSENT (unknown, not False) and surface it.
                cap_unresolved_ids.append(row["task_id"])
                print(
                    f"WARNING: cap_unresolved phase-2 winner (ceiling unknown): "
                    f"{src_dir}",
                    file=sys.stderr,
                )
            else:
                row["truncated_at_ceiling"] = ceiling
                if ceiling:
                    ceiling_task_ids.append(row["task_id"])
            if row.get("max_tokens") is None and "max_tokens" in model_config:
                row["max_tokens"] = model_config["max_tokens"]

        _stamp_episode_copy(
            dst_dir, pass_num=pass_num, model_config=model_config, ceiling=ceiling
        )
        merged_rows.append(row)

    (out_root / "episode_runs.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in merged_rows), encoding="utf-8"
    )

    summary = {
        "total": len(merged_rows),
        "from_phase2": from_phase2,
        "truncated_at_ceiling": sorted(set(ceiling_task_ids)),
        "cap_unresolved": sorted(set(cap_unresolved_ids)),
    }
    (out_root / "two_tier_merge.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Later-pass-wins merge of two-tier phase artifacts roots "
        "(phase 2 overwrites phase 1 per unit; provenance stamped on the copies)."
    )
    parser.add_argument("--phase1", required=True, help="Phase-1 artifacts root.")
    parser.add_argument("--phase2", required=True, help="Phase-2 artifacts root.")
    parser.add_argument("--out", required=True, help="Merged output artifacts root.")
    parser.add_argument(
        "--expected-rerun-manifest",
        default=None,
        help="Phase-2 rerun manifest: every task_id it lists MUST appear in "
        "phase 2 (fail-closed).",
    )
    parser.add_argument(
        "--allow-unresolved-cap",
        action="store_true",
        help="Do NOT fail closed when a phase-2 winner's ceiling cap is "
        "unresolvable and unstamped; record such task_ids under cap_unresolved "
        "in the summary and stderr-WARN instead.",
    )
    args = parser.parse_args(argv)

    summary = merge_phase_roots(
        Path(args.phase1),
        Path(args.phase2),
        Path(args.out),
        expected_rerun_manifest=(
            Path(args.expected_rerun_manifest) if args.expected_rerun_manifest else None
        ),
        allow_unresolved_cap=args.allow_unresolved_cap,
    )
    print(json.dumps(summary, indent=2))
    print(
        f"\nMerged {summary['total']} unit(s); {summary['from_phase2']} from "
        f"phase 2; {len(summary['truncated_at_ceiling'])} still truncated at the "
        f"ceiling; {len(summary['cap_unresolved'])} cap_unresolved "
        f"-> {args.out}/episode_runs.jsonl"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
