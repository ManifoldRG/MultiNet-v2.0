"""One-pass enrichment of the query index: parse each ``query.json``'s ``agent_messages``
text for structured navigation signal (position, facing, previous-action outcome,
toggle-rejections) plus per-query output tokens. Emits a per-query parquet keyed the same
as :func:`analysis.data.load_queries`.

The prompt's final user block reliably carries ``You are at (x, y) facing DIR`` for every
config except ``obs_image_only`` (text-blind — image only). A ``Recent history`` block lists
prior positions followed by ``FINAL_OUTPUT: ACTION`` and ``Feedback: OUTCOME`` lines
(older corpora use the legacy ``(pos) facing DIR -> ACTION -> OUTCOME`` arrow form). Its
most-recent entry is the outcome of the immediately preceding query's action; we attribute
it back one step (dedup-safe).
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import pandas as pd

from analysis.data import load_queries

_POS = re.compile(r"You are at \((\d+),\s*(\d+)\) facing (\w+)")
# One history item is deliberately shaped like a model response so the prompt
# reinforces the same delimiter required for the next action.
_HIST = re.compile(
    r"Position after:\s*\((\d+),\s*(\d+)\),\s*facing\s+(\w+)\s*\n"
    r"FINAL_OUTPUT:\s*([A-Z_]+)\s*\n"
    r"Feedback:\s*([A-Za-z_]+)"
)
# Corpora collected before the FINAL_OUTPUT-shaped template (all runs through the
# 2026-07 kimictx sweep) embed the old arrow form; without this fallback their
# enrichment silently degrades to prev_action/prev_outcome = None.
_HIST_LEGACY = re.compile(
    r"\((\d+),\s*(\d+)\) facing (\w+)\s*->\s*([A-Z_]+)\s*->\s*([A-Za-z_]+)"
)
_TOGGLE_REJECT = re.compile(r"cannot be toggled directly|Gates cannot be toggled", re.I)


def _agent_text(agent_messages) -> str:
    parts = []
    for m in agent_messages or []:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list):
            for x in c:
                if isinstance(x, dict) and x.get("type") == "text":
                    parts.append(x.get("text", ""))
        elif isinstance(c, str):
            parts.append(c)
    return "\n".join(parts)


def _parse_query(path: str) -> dict:
    d = json.loads(Path(path).read_text())
    txt = _agent_text(d.get("agent_messages"))
    pa = d.get("parsed_actions") or []
    out = {"pos_x": None, "pos_y": None, "facing": None,
           "action": (pa[0] if len(pa) == 1 else ("MULTI" if pa else "NONE")),
           "prev_action": None, "prev_outcome": None,
           "toggle_reject": bool(_TOGGLE_REJECT.search(txt)),
           "out_tokens": None, "reply_len": len(d.get("assistant_reply") or "")}
    # Use the LAST match: history-embedding configs (hist_multiturn) repeat past positions;
    # the current position is always the final "You are at" in the prompt.
    pm = _POS.findall(txt)
    if pm:
        out["pos_x"], out["pos_y"], out["facing"] = int(pm[-1][0]), int(pm[-1][1]), pm[-1][2]
    # last history line = outcome of the previous action
    hs = _HIST.findall(txt) or _HIST_LEGACY.findall(txt)
    if hs:
        out["prev_action"], out["prev_outcome"] = hs[-1][3], hs[-1][4].upper()
    u = d.get("usage") or {}
    for k in ("output_tokens", "completion_tokens", "output"):
        if isinstance(u, dict) and u.get(k) is not None:
            out["out_tokens"] = u[k]
            break
    return out


def enrich_queries(cache="analysis/.cache/queries_enriched.parquet", force=False) -> pd.DataFrame:
    cache = Path(cache)
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    q = load_queries()
    parsed = pd.DataFrame([_parse_query(p) for p in q.query_path], index=q.index)
    out = pd.concat([q.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out


if __name__ == "__main__":
    df = enrich_queries(force=True)
    print("enriched rows:", len(df))
    print("position parsed:", df.pos_x.notna().sum(),
          f"({df.pos_x.notna().mean():.1%})")
    print("prev_outcome parsed:", df.prev_outcome.notna().sum())
    print("out_tokens present:", df.out_tokens.notna().sum())
