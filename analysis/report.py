from __future__ import annotations
from pathlib import Path

import pandas as pd

from analysis.data import CONDITIONAL_CONFIGS, load_episodes


def summarise(episodes: pd.DataFrame) -> dict:
    """Descriptive rollups: pass rate by config x model, failure taxonomy, token totals."""
    pass_by = episodes.groupby(["config", "model"]).success.mean()
    fail_tax = (episodes[~episodes.success]
                .groupby(["model", "end_reason"]).size().unstack(fill_value=0))
    tokens = episodes.groupby("config").tokens.sum()
    return {"pass_by_config_model": pass_by, "failure_taxonomy": fail_tax,
            "tokens_by_config": tokens}


def build_report(out="artifacts/analysis/REPORT.md", force=False) -> Path:
    ep = load_episodes(force=force)
    ep = ep[ep.config.isin(CONDITIONAL_CONFIGS)].copy()
    s = summarise(ep)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "# Conditional-Eval Results Report\n",
        f"_{len(ep)} episodes, {ep.config.nunique()} configs, "
        f"{ep.model.nunique()} models, {ep.task_id.nunique()} mazes._\n",
        "## Pass rate by config x model\n",
        s["pass_by_config_model"].unstack().round(3).to_markdown(),
        "\n## Failure taxonomy (end_reason count by model)\n",
        s["failure_taxonomy"].to_markdown(),
        "\n## Tokens by config\n",
        s["tokens_by_config"].to_markdown(),
    ]
    out.write_text("\n".join(parts))
    return out
