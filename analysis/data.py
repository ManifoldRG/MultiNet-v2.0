from __future__ import annotations
import json
from pathlib import Path

import pandas as pd


def discover_sweeps(root: str | Path = "artifacts-pulled") -> list[Path]:
    """Every cell dir (contains a ``runs/`` child) under ``root``, skipping ``issues_``."""
    root = Path(root)
    out: list[Path] = []
    for runs in root.rglob("runs"):
        if not runs.is_dir():
            continue
        if any(part == "issues_" for part in runs.parts):
            continue
        out.append(runs.parent)
    return sorted(out)


_EP_COLS = ["task_id", "model", "config", "prompt_variant", "seed", "sweep",
            "success", "terminated", "truncated", "reward", "steps", "optimal_steps",
            "optimality_ratio", "tokens", "end_reason", "failure_point", "run_dir"]


def _read_end_reason(path: Path) -> str | None:
    try:
        return json.loads(path.read_text()).get("end_reason")
    except (OSError, ValueError):
        return None


def load_episodes(root="artifacts-pulled", cache="analysis/.cache/episodes.parquet",
                  force=False) -> pd.DataFrame:
    """One tidy row per episode across all sweeps (skipping ``issues_``). Cached to parquet.

    ``config`` = the cell dir name (experimental arm, e.g. ``cond_ctx_current``); it is
    distinct from the raw jsonl ``condition`` field. ``end_reason`` is read from each
    ``episode.json`` (the jsonl lacks it).
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
            rows.append({
                "task_id": r["task_id"], "model": r.get("agent_or_model"),
                "config": cell.name, "prompt_variant": r.get("prompt_variant"),
                "seed": r.get("seed"), "sweep": cell.parent.name,
                "success": bool(r.get("success")), "terminated": bool(r.get("terminated")),
                "truncated": bool(r.get("truncated")), "reward": r.get("reward"),
                "steps": r.get("steps"), "optimal_steps": r.get("optimal_steps"),
                "optimality_ratio": r.get("optimality_ratio"), "tokens": r.get("tokens"),
                "end_reason": _read_end_reason(run_dir),
                "failure_point": r.get("failure_point"), "run_dir": str(run_dir.parent),
            })
    df = pd.DataFrame(rows, columns=_EP_COLS)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df
