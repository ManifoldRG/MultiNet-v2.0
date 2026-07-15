from __future__ import annotations
import json
from pathlib import Path

import pandas as pd


def discover_sweeps(root: str | Path = "artifacts-pulled",
                    include_issues: bool = True) -> list[Path]:
    """Every cell dir (contains a ``runs/`` child) under ``root``.

    ``issues_/`` cells (buggy / stale-partial runs) are INCLUDED by default and relabeled
    into the ``bugged_*`` config family (see :data:`_BUGGED_CONFIG`) — retained for mining
    but excluded from :data:`CONDITIONAL_CONFIGS` (the statistical set). Pass
    ``include_issues=False`` to drop them entirely (legacy behavior)."""
    root = Path(root)
    out: list[Path] = []
    for runs in root.rglob("runs"):
        if not runs.is_dir():
            continue
        if not include_issues and any(part == "issues_" for part in runs.parts):
            continue
        out.append(runs.parent)
    return sorted(out)


_EP_COLS = ["task_id", "model", "config", "prompt_variant", "seed", "sweep",
            "success", "terminated", "truncated", "reward", "steps", "optimal_steps",
            "optimality_ratio", "tokens", "end_reason", "failure_point", "run_dir"]

_COND_MASSIVE_VARIANT_CONFIG = {
    "image_only": "obs_image_only",
    "current": "ctx_current",
    "text_summary": "ctx_text_summary",
    "cardinal": "act_cardinal",
    "subgoal": "qry_subgoal",
    "full_trajectory": "qry_full_trajectory",
    "zero_shot": "icl_zero_shot",
    "multiturn": "hist_multiturn",
    # In the Qwen combined job, Prompt/standard is the thinking-on baseline.
    # The Prompt ablation variants live in the separate cond_prompt cell.
    "standard": "baseline_thinking",
}

_CELL_CONFIG_ALIASES = {
    "cond_baseline_thinking": "baseline_thinking",
    "cond_obs_image_only": "obs_image_only",
    "cond_ctx_current": "ctx_current",
    "cond_ctx_text_summary": "ctx_text_summary",
    "cond_act_cardinal": "act_cardinal",
    "cond_qry_subgoal": "qry_subgoal",
    "cond_qry_full_trajectory": "qry_full_trajectory",
    "cond_icl_zero_shot": "icl_zero_shot",
    "cond_hist_multiturn": "hist_multiturn",
}

# The prompt-detail ablation (``cond_prompt`` cell) is ALWAYS split into its three
# variants — never aggregated. The pooled ``cond_prompt`` triple-counted the 15 mazes
# (135 vs 45 episodes), inflating its token totals and making it non-comparable to every
# other 15-maze condition. See :func:`_config_name`.
COND_PROMPT_VARIANTS = ["cond_prompt·minimal", "cond_prompt·standard", "cond_prompt·verbose"]

CONDITIONAL_CONFIGS = [
    "baseline_thinking",
    *COND_PROMPT_VARIANTS,
    "qry_subgoal",
    "icl_zero_shot",
    "ctx_text_summary",
    "ctx_current",
    "hist_multiturn",
    "act_cardinal",
    "qry_full_trajectory",
    "obs_image_only",
]

# ``issues_/`` cells are relabeled into this ``bugged_*`` family: kept in the parquet for
# mining (the buggy-vs-clean contrast is itself a finding), but excluded from the stats set.
# ``bugged_1`` = ICL-dup multiturn (repeats the one-shot example every turn → tanks Claude
# to ~7%, cf. the clean ``hist_multiturn`` at 53%). ``bugged_2/3`` = stale ctx partials
# (superseded by the complete runs; 0 episodes but the queries carry partial-run signal).
_BUGGED_CONFIG = {
    "cond_hist_multiturn_BUGGY": "bugged_1_hist_multiturn_icldup",
    "cond_ctx_current_partial": "bugged_2_ctx_current_partial",
    "cond_ctx_text_summary_partial": "bugged_3_ctx_text_summary_partial",
}


def is_clean(config: str) -> bool:
    """True for the canonical statistical configs (not a ``bugged_*`` / dev cell)."""
    return config in CONDITIONAL_CONFIGS


def split_cond_prompt_config(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize any legacy aggregated ``cond_prompt`` rows into per-variant configs.

    Fresh loads via :func:`load_episodes` / :func:`load_queries` already emit
    ``cond_prompt·<variant>`` — the prompt-detail ablation is never pooled. This repairs
    *derived* caches (behavior/positions/trajectories/waste/features/…) that were built
    before the split. Variant is read from a ``prompt_variant`` column if present, else the
    trailing segment of ``episode_id`` (``.../seed_0/<variant>``). No-op if already split."""
    if "config" not in df.columns:
        return df
    mask = df["config"] == "cond_prompt"
    if not mask.any():
        return df
    df = df.copy()
    if "prompt_variant" in df.columns and df.loc[mask, "prompt_variant"].notna().all():
        variant = df.loc[mask, "prompt_variant"].astype(str)
    else:
        variant = df.loc[mask, "episode_id"].str.rsplit("/", n=1).str[-1]
    df.loc[mask, "config"] = "cond_prompt·" + variant
    return df


def _sweep_name(cell: Path) -> str:
    """Sweep id for a cell, accounting for the extra ``issues_/`` nesting level."""
    return cell.parent.parent.name if cell.parent.name == "issues_" else cell.parent.name


def _config_name(cell: Path, prompt_variant: str | None) -> str:
    if cell.parent.name == "issues_":
        return _BUGGED_CONFIG.get(cell.name, f"bugged_{cell.name}")
    if cell.name == "cond_massive":
        return _COND_MASSIVE_VARIANT_CONFIG.get(str(prompt_variant), cell.name)
    if cell.name == "cond_prompt":
        # Prompt-detail ablation: split into per-variant conditions, never aggregate
        # (the pooled cell triple-counted mazes and skewed every rollup).
        return f"cond_prompt·{prompt_variant}"
    return _CELL_CONFIG_ALIASES.get(cell.name, cell.name)


def _read_end_reason(path: Path) -> str | None:
    try:
        return json.loads(path.read_text()).get("end_reason")
    except (OSError, ValueError):
        return None


def load_episodes(root="artifacts-pulled", cache="analysis/.cache/episodes.parquet",
                  force=False) -> pd.DataFrame:
    """One tidy row per episode across all sweeps (skipping ``issues_``). Cached to parquet.

    ``config`` = the cell dir name (experimental arm, e.g. ``cond_ctx_current``); it is
    distinct from the raw jsonl ``condition`` field. ``end_reason`` is read from the
    aggregate jsonl row when present; older rows that lack it fall back to the
    per-run ``episode.json``.
    """
    cache = Path(cache)
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    rows = []
    for cell in discover_sweeps(root):
        jsonl = cell / "episode_runs.jsonl"
        if not jsonl.exists():
            continue
        for line in jsonl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            run_dir = cell / r["raw_output_ref"]
            prompt_variant = r.get("prompt_variant")
            rows.append({
                "task_id": r["task_id"], "model": r.get("agent_or_model"),
                "config": _config_name(cell, prompt_variant), "prompt_variant": prompt_variant,
                "seed": r.get("seed"), "sweep": _sweep_name(cell),
                "success": bool(r.get("success")), "terminated": bool(r.get("terminated")),
                "truncated": bool(r.get("truncated")), "reward": r.get("reward"),
                "steps": r.get("steps"), "optimal_steps": r.get("optimal_steps"),
                "optimality_ratio": r.get("optimality_ratio"), "tokens": r.get("tokens"),
                "end_reason": r.get("end_reason") or _read_end_reason(run_dir),
                "failure_point": r.get("failure_point"), "run_dir": str(run_dir.parent),
            })
    df = pd.DataFrame(rows, columns=_EP_COLS)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


_Q_COLS = ["episode_id", "task_id", "model", "config", "seed", "query_index",
           "env_step", "parse_ok", "n_actions", "has_image", "latency_s", "query_path"]


def _episode_id(cell: Path, rel_parts) -> str:
    # rel_parts = (task_id, 'minigrid', model, 'seed_N', prompt_variant, 'queries', ...)
    task_id, _, model, seed_dir, variant = rel_parts[:5]
    return f"{_sweep_name(cell)}/{cell.name}/{task_id}/{model}/{seed_dir}/{variant}"


def load_queries(root="artifacts-pulled", cache="analysis/.cache/queries.parquet",
                 force=False) -> pd.DataFrame:
    """One row per model query (step) across all sweeps. ``query_path`` points at the
    ``query.json``; its ``agent_messages``/``assistant_reply`` text is loaded lazily via
    :func:`load_query_text` to keep this table small."""
    cache = Path(cache)
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    rows = []
    for cell in discover_sweeps(root):
        for qjson in (cell / "runs").rglob("queries/query_*/query.json"):
            rel = qjson.relative_to(cell / "runs").parts
            task_id, _, model, seed_dir, variant = rel[:5]
            seed = int(seed_dir.split("_")[1]) if "_" in seed_dir else None
            try:
                q = json.loads(qjson.read_text())
            except ValueError:
                continue
            rows.append({
                "episode_id": _episode_id(cell, rel),
                "task_id": task_id, "model": model, "config": _config_name(cell, variant), "seed": seed,
                "query_index": q.get("query_index"), "env_step": q.get("env_step_count"),
                "parse_ok": bool(q.get("parse_ok")),
                "n_actions": len(q.get("parsed_actions") or []),
                "has_image": bool(q.get("has_image")), "latency_s": q.get("llm_latency_s"),
                "query_path": str(qjson)})
    df = pd.DataFrame(rows, columns=_Q_COLS)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def load_query_text(query_path: str) -> tuple[list[dict], str]:
    """Return ``(agent_messages, assistant_reply)`` for one query.json."""
    q = json.loads(Path(query_path).read_text())
    return q.get("agent_messages") or [], q.get("assistant_reply") or ""
