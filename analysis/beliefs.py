"""World-model fidelity mining: compare a model's stated beliefs (in its inline reasoning) to
the ground-truth environment state. Reasoning is inline prose terminated by ``FINAL_OUTPUT:``
(no ``<think>`` tags / reasoning field); this is minable for Kimi and Qwen, and for Claude only
where it narrates (``qry_full_trajectory``) — Claude's default reasoning is suppressed/redacted.

Three checks:
  1. self-state echo   — the model's restated CURRENT facing/position vs the env-given state.
  2. mental rotation   — for a planned ``TURN_x (now facing Y)`` chain, does each claimed
                         resulting facing match the true rotation (validated: RIGHT = clockwise)?
  3. wall belief       — ``(x, y) is a wall`` claims vs the true wall set (both coord conventions).

Ground-truth facing/position come from the enriched query table (env-given "You are at (row,col)
facing DIR"); wall sets from the maze specs.
"""
from __future__ import annotations
import glob
import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from analysis.data import is_clean
from analysis.mazes import load_maze_spec

CACHE = Path("analysis/.cache")
_CW = ["NORTH", "EAST", "SOUTH", "WEST"]

_PLAN = re.compile(r"(?:Step\s*1|1\.\s|TURN_LEFT|TURN_RIGHT|MOVE_FORWARD|FINAL_OUTPUT|->)", re.I)
_JOINT = re.compile(r"(?:I'?m|I am|currently|agent is)\s+(?:at\s*)?\(?(\d+),\s*(\d+)\)?\s*(?:,|and)?"
                    r"\s*facing\s+(NORTH|SOUTH|EAST|WEST)", re.I)
_LBLP = re.compile(r"[Pp]osition:\s*\(?(\d+),\s*(\d+)\)?")
_LBLF = re.compile(r"[Ff]acing:\s*(NORTH|SOUTH|EAST|WEST)", re.I)
_TURN = re.compile(r"TURN_(LEFT|RIGHT)\s*[^A-Za-z]{0,6}(?:now\s+)?facing\s+(NORTH|SOUTH|EAST|WEST)", re.I)
_WALL = re.compile(r"\(?(\d+),\s*(\d+)\)?\s+is\s+(?:a\s+)?wall", re.I)
_ARROW = re.compile(r"\((\d+),\s*(\d+)\)(?:\s*(?:->|→|to)\s*\((\d+),\s*(\d+)\))+")
_PAIR = re.compile(r"\((\d+),\s*(\d+)\)")


def _rot(f: str, turn: str) -> str:
    i = _CW.index(f)
    return _CW[(i + 1) % 4] if turn == "RIGHT" else _CW[(i - 1) % 4]


@lru_cache(maxsize=1)
def _wall_map() -> dict[str, set]:
    """spec.task_id -> {(x, y)} wall set, for every maze fixture."""
    out = {}
    for p in glob.glob("mazes/**/*.json", recursive=True):
        try:
            sp = load_maze_spec(p)
            out[sp.task_id] = {(w.x, w.y) for w in sp.maze.walls}
        except Exception:
            continue
    return out


def _walls_for(task_id: str):
    wm = _wall_map()
    hit = next((k for k in sorted(wm, key=len, reverse=True) if task_id.endswith(k)), None)
    return wm.get(hit)


def analyse(models=("kimi-k2.6", "Qwen_Qwen3.6-27B", "claude-opus-4-8"),
            min_len=150, cache=CACHE / "beliefs.parquet", force=False) -> pd.DataFrame:
    cache = Path(cache)
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    q = pd.read_parquet(CACHE / "queries_enriched.parquet")
    q = q[q.config.apply(is_clean) & q.model.isin(models) & (q.reply_len > min_len)
          & q.facing.notna()]
    rows = []
    for _, r in q.iterrows():
        txt = json.loads(Path(r.query_path).read_text()).get("assistant_reply", "")
        gt_f = str(r.facing).upper(); gt_p = (int(r.pos_x), int(r.pos_y))
        rec = {"model": r.model, "config": r.config, "task_id": r.task_id,
               "episode_id": r.episode_id, "query_index": r.query_index, "env_step": r.env_step,
               "echo_face": None, "echo_pos": None,
               "rot_n": 0, "rot_err": 0, "rot_inverted": 0, "rot_noop": 0, "rot_other": 0,
               "wall_n": 0, "wall_err": 0, "trans_n": 0, "trans_err": 0}
        # 1. self-state echo (preamble only)
        pre = txt[:_PLAN.search(txt).start()] if _PLAN.search(txt) else txt
        m = _JOINT.search(pre)
        cf = cp = None
        if m:
            cp = (int(m.group(1)), int(m.group(2))); cf = m.group(3).upper()
        else:
            mf, mp = _LBLF.search(pre), _LBLP.search(pre)
            if mf: cf = mf.group(1).upper()
            if mp: cp = (int(mp.group(1)), int(mp.group(2)))
        if cf is not None: rec["echo_face"] = (cf == gt_f)
        if cp is not None: rec["echo_pos"] = (cp == gt_p)
        # 4. translation: on explicit path traces "(a,b)->(c,d)->...", count adjacent (unit) steps
        #    that route INTO a wall. Only when the (row,col) convention is confirmed by the self-claim
        #    (walls are stored (x,y), so a (row,col) cell maps to (col,row)); waypoint jumps (d>1) skipped.
        walls_t = _walls_for(r.task_id)
        if walls_t is not None and cp == gt_p:  # convention confirmed = (row, col)
            for mm in _ARROW.finditer(txt):
                cells = [(int(a), int(b)) for a, b in _PAIR.findall(mm.group(0))]
                for i in range(len(cells) - 1):
                    if abs(cells[i][0]-cells[i+1][0]) + abs(cells[i][1]-cells[i+1][1]) == 1:
                        rec["trans_n"] += 1
                        rec["trans_err"] += ((cells[i+1][1], cells[i+1][0]) in walls_t)
        # 2. mental rotation (follow the model's own chain so a slip doesn't cascade)
        f = gt_f
        for td, claim in _TURN.findall(txt):
            td = td.upper(); true_f = _rot(f, td); claim = claim.upper()
            rec["rot_n"] += 1
            if claim != true_f:
                rec["rot_err"] += 1
                other = "LEFT" if td == "RIGHT" else "RIGHT"
                if claim == _rot(f, other):   # applied the opposite turn (180 from true)
                    rec["rot_inverted"] += 1
                elif claim == f:              # believed no rotation happened
                    rec["rot_noop"] += 1
                else:                          # the remaining perpendicular facing
                    rec["rot_other"] += 1
            f = claim
        # 3. wall beliefs (accept either coord convention as "correct")
        walls = _walls_for(r.task_id)
        if walls:
            for a, b in _WALL.findall(txt):
                a, b = int(a), int(b)
                rec["wall_n"] += 1
                rec["wall_err"] += 0 if ((a, b) in walls or (b, a) in walls) else 1
        rows.append(rec)
    df = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def summary(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = analyse()
    g = df.groupby("model")
    out = pd.DataFrame({
        "n_queries": g.size(),
        "echo_face_acc": g.echo_face.mean(),
        "echo_pos_acc": g.echo_pos.mean(),
        "rot_claims": g.rot_n.sum(),
        "rot_err_rate": g.rot_err.sum() / g.rot_n.sum().replace(0, pd.NA),
        "wall_claims": g.wall_n.sum(),
        "wall_err_rate": g.wall_err.sum() / g.wall_n.sum().replace(0, pd.NA),
        "trans_claims": g.trans_n.sum(),
        "trans_err_rate": g.trans_err.sum() / g.trans_n.sum().replace(0, pd.NA),
    })
    return out


if __name__ == "__main__":
    df = analyse(force=True)
    print("belief records:", len(df))
    print(summary(df).round(3).to_string())
