from __future__ import annotations
import glob
import json
from pathlib import Path

import pandas as pd

from gridworld.task_spec import TaskSpecification
from gridworld.task_validator import compute_difficulty


def load_maze_spec(path: str) -> TaskSpecification:
    return TaskSpecification.from_dict(json.loads(Path(path).read_text()))


def _max_corridor_len(spec) -> int:
    """Longest straight run of open (non-wall) cells along either axis — a proxy for
    corridor length. Built from ``MazeLayout.dimensions`` + ``walls`` (Position(x,y))."""
    m = getattr(spec, "maze", None)
    dims = getattr(m, "dimensions", None) if m is not None else None
    if not dims:
        return 0
    D0, D1 = int(dims[0]), int(dims[1])
    walls = set()
    for w in (getattr(m, "walls", []) or []):
        x, y = getattr(w, "x", None), getattr(w, "y", None)
        if x is None:
            try:
                x, y = w[0], w[1]
            except (TypeError, IndexError):
                continue
        walls.add((int(x), int(y)))
    best = 0
    for a in range(D0):
        cur = 0
        for b in range(D1):
            cur = cur + 1 if (a, b) not in walls else 0
            best = max(best, cur)
    for b in range(D1):
        cur = 0
        for a in range(D0):
            cur = cur + 1 if (a, b) not in walls else 0
            best = max(best, cur)
    return best


def maze_features(maze_globs=("mazes/**/*.json",),
                  cache="analysis/.cache/maze_features.parquet", force=False) -> pd.DataFrame:
    """One row per maze (all matched JSONs) of structural difficulty features, derived
    from the BFS solver in ``gridworld.task_validator.compute_difficulty``. Cached."""
    cache = Path(cache)
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    paths = sorted({p for g in maze_globs for p in glob.glob(g, recursive=True)})
    rows = []
    for p in paths:
        spec = load_maze_spec(p)
        d = compute_difficulty(spec).to_dict()
        mech = getattr(spec, "mechanisms", None)

        def _n(name: str) -> int:  # MechanismSet exposes keys/doors/... as list attrs
            return len(getattr(mech, name, []) or []) if mech is not None else 0

        rows.append({
            "task_id": d["task_id"], "tier": d.get("tier"), "grid_area": d["grid_area"],
            "is_beatable": d["is_beatable"], "optimal_steps": d["optimal_steps"],
            "states_explored": d["states_explored"], "dependency_depth": d["dependency_depth"],
            "mechanism_count": d["mechanism_count"], "mechanism_types": d["mechanism_types"],
            "backtrack_count": d["backtrack_count"], "difficulty_score": d["difficulty_score"],
            "n_keys": _n("keys"), "n_doors": _n("doors"),
            "n_switches": _n("switches"), "n_gates": _n("gates"),
            "n_blocks": _n("blocks"),
            "max_steps": getattr(spec, "max_steps", None),
            "max_corridor_len": _max_corridor_len(spec)})
    df = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df
