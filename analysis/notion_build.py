"""Assemble the Experiment 3 Notion import bundle (untracked output).

Injects computed stats into `<!-- STAT: key -->` slots in the Markdown page
templates under ``analysis/notion/``, verifies every referenced figure exists,
and zips the landing page + subpage tree for Notion's *Import -> Markdown & CSV*.

Solve-rate stat keys use the SHORT model names (``solve_claude_pct``,
``solve_kimi_pct``, ``solve_qwen_pct``) via the same ``MODEL_LBL`` mapping
``analysis.plots`` uses, so page authors get predictable, model-id-agnostic keys.
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pandas as pd

SLOT_RE = re.compile(r"<!--\s*STAT:\s*([a-z0-9_]+)\s*-->")
IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
LANDING = "Experiment 3 — MultiNet Conditional Eval"

# Fallback if analysis.plots (matplotlib) is unavailable; kept in sync with it.
_MODEL_LBL_FALLBACK = {
    "claude-opus-4-8": "Claude", "kimi-k2.6": "Kimi", "Qwen_Qwen3.6-27B": "Qwen",
}


def _model_lbl() -> dict:
    try:
        from analysis.plots import MODEL_LBL
        return MODEL_LBL
    except Exception:
        return _MODEL_LBL_FALLBACK


def _short(model: str, lbl: dict) -> str:
    """Full model id -> short slug used in stat keys (e.g. 'claude')."""
    return str(lbl.get(model, model)).lower()


def _usd(v) -> str:
    return f"${v:,.2f}" if v is not None else "pending actual billing"


def compute_stats(ep: pd.DataFrame, cost: dict) -> dict[str, str]:
    lbl = _model_lbl()
    s = {
        "total_episodes": str(len(ep)),
        "total_tokens_m": f"{ep.tokens.sum() / 1e6:.1f}M",
    }
    for m in ep.model.unique():
        s[f"solve_{_short(m, lbl)}_pct"] = f"{ep[ep.model == m].success.mean() * 100:.1f}%"

    # NOTE: `llm_hours` (FINDINGS A2 = 214.9h, total across models) is deliberately
    # NOT emitted from cost_inputs.qwen_a100_hours (53.1h is Qwen-only GPU wall-clock,
    # a different quantity). A page referencing `<!-- STAT: llm_hours -->` will fail
    # loud (KeyError) until a real total-hours source exists — no silent stale number.
    s["cost_claude_usd"] = _usd(cost.get("claude_api_usd"))
    s["cost_kimi_usd"] = _usd(cost.get("kimi_api_usd"))
    s["cost_qwen_usd"] = _usd(cost.get("qwen_a100_usd"))

    claude, kimi, qwen = (cost.get(k) for k in
                          ("claude_api_usd", "kimi_api_usd", "qwen_a100_usd"))
    if all(v is not None for v in (claude, kimi, qwen)):
        s["cost_total_usd"] = _usd(claude + kimi + qwen)
    else:
        # Both API numbers still pending -> anchor on the one known cost (Qwen).
        s["cost_total_usd"] = f"pending (Qwen alone: {_usd(qwen)})"
    return s


def inject(md: str, stats: dict[str, str]) -> str:
    def sub(m):
        key = m.group(1)
        if key not in stats:
            raise KeyError(f"unknown STAT slot: {key}")
        return stats[key]
    return SLOT_RE.sub(sub, md)


def build(src="analysis/notion", out_zip=None, stats=None):
    src = Path(src)
    out_zip = Path(out_zip) if out_zip is not None else src / "multinet-exp3-notion.zip"
    if stats is None:
        from analysis.data import is_clean, load_episodes
        ep = load_episodes()
        ep = ep[ep.config.apply(is_clean)]
        cost = json.loads((src / "cost_inputs.json").read_text())
        stats = compute_stats(ep, cost)

    pages = sorted(src.rglob("*.md"))
    if not pages:
        raise FileNotFoundError(
            f"no .md page templates under {src} — nothing to build "
            "(page authoring is Tasks 6-8; the bundle needs at least the landing page)"
        )

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for page in pages:
            text = inject(page.read_text(), stats)
            for img in IMG_RE.findall(text):
                if not (page.parent / img).exists():
                    raise FileNotFoundError(f"{page.name} references missing figure: {img}")
            zf.writestr(str(page.relative_to(src)), text)
        figdir = src / LANDING / "figures"
        for png in sorted(figdir.glob("*.png")):
            zf.write(png, str(png.relative_to(src)))
    print("wrote", out_zip)
    return out_zip


if __name__ == "__main__":
    build()
