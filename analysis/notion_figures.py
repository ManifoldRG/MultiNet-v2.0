"""Figures for the Experiment 3 Notion bundle. Output is gitignored; code is the record.

Renders the maze-level solve matrices, the two landing figures (solve-rate CI +
per-model spend), and one death-cell dwell map per hard maze, into a caller-supplied
output directory. Styling (palette, rcParams, colormap) is inherited from
``analysis.plots`` so these match the main ``analysis/figures`` set.

Coordinate note: the spatial death-cell maps are drawn in the *spec* ``(x, y)``
convention (x = column, y = row, y-axis inverted so row 1 is at the top) — the same
convention ``plots.fig_spatial_heatmaps`` uses. The enriched ``positions.parquet``
stores ``pos_x = row = y`` and ``pos_y = col = x``, so a visited cell maps to
``(x = pos_y, y = pos_x)``. BFS path/goal/start come straight from the spec as ``(x, y)``.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.data import CONDITIONAL_CONFIGS, is_clean, load_episodes

KIMICTX_ROOT = Path("Multinet-v2-results/kimictx_context_window")
KIMICTX_CONFIGS = ["ctx_last3", "ctx_text_summary", "ctx_text_summary_and_last3"]

# Hard mazes are resolved at render time (pooled clean solve < this threshold).
HARD_SOLVE_THRESHOLD = 0.35


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def maze_matrix(ep: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
    """Per-maze mean-success matrix: rows = task_id sorted by optimal_steps,
    columns = ``CONDITIONAL_CONFIGS`` (NaN where no episode ran)."""
    df = ep[ep.config.apply(is_clean)] if "config" in ep else ep
    if model is not None:
        df = df[df.model == model]
    order = df.groupby("task_id").optimal_steps.max().sort_values().index
    piv = df.pivot_table(index="task_id", columns="config", values="success", aggfunc="mean")
    return piv.reindex(index=order, columns=CONDITIONAL_CONFIGS)


def load_kimictx_episodes(path: str | Path) -> pd.DataFrame:
    """Normalize the kimictx ``episode_runs.jsonl`` to ``_EP_COLS``-compatible columns.

    kimictx schema: ``condition`` is the maze family (S/D/M/B/default), ``prompt_variant``
    is the context variant (last3 / text_summary / text_summary_and_last3), and
    ``agent_or_model`` is the model. The Notion "config" axis is the context variant, so
    ``config = "ctx_" + prompt_variant`` and ``model = agent_or_model``.
    """
    rows = [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]
    df = pd.DataFrame(rows)
    df["config"] = "ctx_" + df["prompt_variant"].astype(str)
    df["model"] = df["agent_or_model"]
    keep = ["task_id", "model", "config", "seed", "success", "steps",
            "optimal_steps", "tokens"]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rendering (styling inherited from analysis.plots)
# ---------------------------------------------------------------------------
from analysis import plots as _p  # noqa: E402  (rcParams + Agg side-effect on import)
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def _order_rows_by_opt(ep: pd.DataFrame, index) -> list:
    """Return ``index`` reordered by each task's max optimal_steps (easy → hard)."""
    opt = ep.groupby("task_id").optimal_steps.max()
    return list(pd.Index(index).to_series().map(lambda t: (opt.get(t, np.inf), t))
                .sort_values().index)


def _render_matrix(m: pd.DataFrame, title: str, out: Path, cbar_label: str = "solve rate"):
    fig, ax = plt.subplots(figsize=(0.62 * len(m.columns) + 3.2, 0.42 * len(m) + 1.8))
    im = ax.imshow(np.ma.masked_invalid(m.values.astype(float)), cmap=_p.SEQ,
                   vmin=0, vmax=1, aspect="auto")
    im.cmap.set_bad(_p.GRID)  # NaN cells read as recessive grey, not colored
    ax.set_xticks(range(len(m.columns)))
    ax.set_xticklabels(m.columns, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels(m.index, fontsize=8.5)
    for i in range(len(m)):
        for j in range(len(m.columns)):
            v = m.values[i, j]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.55 else _p.INK)
    ax.set_title(title, fontweight="bold", color=_p.INK, pad=10)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label(cbar_label, color=_p.INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def _render_landing_ci(ep: pd.DataFrame, out: Path) -> Path:
    """Overall solve rate per model with a 95% bootstrap CI (mirrors plots.fig_model_ci)."""
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for i, mdl in enumerate(_p.MODEL_ORDER):
        x = ep[ep.model == mdl].success.values.astype(float)
        if len(x) == 0:
            continue
        bs = rng.choice(x, (5000, len(x))).mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ax.bar(i, x.mean(), color=_p.MODEL_C[mdl], width=0.62, zorder=2)
        ax.errorbar(i, x.mean(), yerr=[[x.mean() - lo], [hi - x.mean()]], color=_p.INK,
                    capsize=5, lw=1.6, zorder=3)
        ax.text(i, hi + 0.02, f"{x.mean() * 100:.1f}%", ha="center", color=_p.INK,
                fontweight="bold")
    ax.set_xticks(range(len(_p.MODEL_ORDER)))
    ax.set_xticklabels([_p.MODEL_LBL[m] for m in _p.MODEL_ORDER])
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("overall solve rate")
    ax.set_yticks(np.arange(0, 0.76, 0.25))
    ax.set_yticklabels([f"{int(t * 100)}%" for t in np.arange(0, 0.76, 0.25)])
    ax.set_title("Overall solve rate  (95% bootstrap CI)", fontweight="bold", color=_p.INK)
    ax.axhline(0, color=_p.AXIS, lw=1)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def _render_landing_spend(ep: pd.DataFrame, out: Path) -> Path:
    """Per-model spend, as three small multiples (one measure each — no dual axis).

    Wall-clock / LLM-hours are deliberately omitted: build_artifact.py notes Qwen
    wall-clock is local-A100 inference and not comparable to API latency, and the
    per-query timings live only in the heavy queries parquet. Total tokens (M),
    episode count, and solve rate are the comparable, episode-table-native measures.
    """
    g = ep.groupby("model").agg(tokens=("tokens", "sum"),
                                episodes=("success", "size"),
                                solve=("success", "mean"))
    models = [m for m in _p.MODEL_ORDER if m in g.index]
    panels = [("total tokens (M)", [g.loc[m, "tokens"] / 1e6 for m in models], "{:.2f}"),
              ("episodes run", [g.loc[m, "episodes"] for m in models], "{:.0f}"),
              ("solve rate", [g.loc[m, "solve"] for m in models], "{:.0%}")]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0))
    for ax, (label, vals, fmt) in zip(axes, panels):
        for i, m in enumerate(models):
            ax.bar(i, vals[i], color=_p.MODEL_C[m], width=0.62, zorder=2)
            ax.text(i, vals[i], fmt.format(vals[i]), ha="center", va="bottom",
                    fontsize=9, color=_p.INK, fontweight="bold")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([_p.MODEL_LBL[m] for m in models])
        ax.set_ylim(0, max(vals) * 1.18 if max(vals) else 1)
        ax.set_title(label, fontweight="bold", color=_p.INK, fontsize=11)
        ax.grid(axis="y", color=_p.GRID, lw=0.8)
    fig.suptitle("Per-model spend on the conditional sweep", fontweight="bold",
                 color=_p.INK, fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def render_landing(ep: pd.DataFrame, outdir: Path) -> list[Path]:
    return [_render_landing_ci(ep, outdir / "nb_landing_solve_ci.png"),
            _render_landing_spend(ep, outdir / "nb_landing_spend.png")]


def _resolve_maze_path(task_id: str) -> str | None:
    """Map an episode task_id to its maze JSON under ``mazes/`` by matching the spec's
    own task_id as a normalized suffix (episode ids carry a family prefix the spec drops)."""
    from analysis.mazes import load_maze_spec

    def norm(s: str) -> str:
        return s.lower().replace("-", "_")

    nt = norm(task_id)
    best = None
    for p in sorted(glob.glob("mazes/**/*.json", recursive=True)):
        try:
            tid = load_maze_spec(p).task_id
        except Exception:
            continue
        if tid and nt.endswith(norm(tid)):
            # prefer the longest (most specific) matching spec task_id
            if best is None or len(tid) > best[1]:
                best = (p, len(tid))
    return best[0] if best else None


def _render_deathcell(tid: str, spec_path: str, pos: pd.DataFrame, solve: float,
                      out: Path) -> Path:
    """One dwell-density map for a hard maze — visit mass + goal/start + BFS-optimal path.

    Adapts plots.fig_spatial_heatmaps' per-cell coordinate mining. Drawn in spec (x, y):
    x = column, y = row (inverted). pos_x = row = y, pos_y = col = x.
    """
    from analysis.mazes import load_maze_spec
    from gridworld.baselines import plan_bfs_path

    sp = load_maze_spec(spec_path)
    W, H = sp.maze.dimensions
    walls = {(w.x, w.y) for w in sp.maze.walls}
    goal = (sp.goal.target.x, sp.goal.target.y)
    start = (sp.maze.start.x, sp.maze.start.y)

    fig, ax = plt.subplots(figsize=(0.34 * W + 2.4, 0.34 * H + 2.0))
    ax.set_facecolor(_p.SURFACE)
    for wx in range(1, W + 1):
        for wy in range(1, H + 1):
            if (wx, wy) in walls:
                ax.add_patch(plt.Rectangle((wx - 0.5, wy - 0.5), 1, 1, color="#cfcec7",
                                           ec=_p.SURFACE, lw=0.5, zorder=1))
    # dwell overlay (pos_x = row = y, pos_y = col = x) -> cell (x=col, y=row)
    dwell: dict[tuple[int, int], int] = {}
    for _, r in pos[pos.task_id == tid].iterrows():
        gx, gy = int(r.pos_y), int(r.pos_x)
        if 1 <= gx <= W and 1 <= gy <= H and (gx, gy) not in walls:
            dwell[(gx, gy)] = dwell.get((gx, gy), 0) + 1
    vmax = max(dwell.values()) if dwell else 1
    norm = matplotlib.colors.LogNorm(vmin=1, vmax=vmax)
    for (gx, gy), n in dwell.items():
        ax.add_patch(plt.Rectangle((gx - 0.5, gy - 0.5), 1, 1, color=_p.SEQ(norm(n)),
                                   ec=_p.SURFACE, lw=0.5, zorder=2))
    # BFS-optimal path overlay
    try:
        pp = plan_bfs_path(sp)
        if pp.success and pp.positions:
            xs = [p[0] for p in pp.positions]
            ys = [p[1] for p in pp.positions]
            ax.plot(xs, ys, color="#b3312f", lw=1.8, alpha=0.9, zorder=5,
                    solid_capstyle="round", label="BFS-optimal path")
    except Exception as e:  # pragma: no cover - overlay is best-effort
        print("  (BFS overlay skipped:", e, ")")
    ax.scatter(*goal, marker="*", s=300, color="#0ca30c", edgecolor="white",
               linewidth=1.2, zorder=6)
    ax.scatter(*start, marker="s", s=80, facecolor="none", edgecolor=_p.INK,
               linewidth=1.8, zorder=6)
    ax.set_xlim(0.4, W + 0.6)
    ax.set_ylim(H + 0.6, 0.4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{tid}\npooled solve {solve * 100:.0f}%   "
                 "★ goal  ▫ start  red = BFS-optimal  grey = wall",
                 fontsize=9.5, color=_p.INK, pad=6)
    ax.text(0.5, -0.04, "spec (x, y): x = column, y = row (row 1 at top)",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=_p.MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def render_deathcells(outdir: Path) -> list[Path]:
    """One dwell map per hard maze (pooled clean solve < HARD_SOLVE_THRESHOLD)."""
    ep = load_episodes()
    ep = ep[ep.config.apply(is_clean)]
    solve = ep.groupby("task_id").success.mean()
    hard = solve[solve < HARD_SOLVE_THRESHOLD].index
    pos_path = _p.CACHE / "positions.parquet"
    pos = pd.read_parquet(pos_path) if pos_path.exists() else pd.DataFrame(
        columns=["task_id", "pos_x", "pos_y"])
    written: list[Path] = []
    for tid in hard:
        spec_path = _resolve_maze_path(tid)
        if spec_path is None:
            print("  (no maze spec resolved for", tid, "- skipping)")
            continue
        out = outdir / f"nb_deathcells_{tid}.png"
        written.append(_render_deathcell(tid, spec_path, pos, float(solve[tid]), out))
    return written


def render_all(outdir: str | Path) -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ep = load_episodes()
    ep = ep[ep.config.apply(is_clean)]
    written: list[Path] = []

    # Per-maze solve matrices: one per model + pooled
    for model in [*_p.MODEL_ORDER, None]:
        name = f"nb_matrix_{_p.MODEL_LBL.get(model, 'pooled').lower()}.png"
        title = f"Per-maze solve — {_p.MODEL_LBL.get(model, 'pooled (3 models)')}"
        _render_matrix(maze_matrix(ep, model), title, outdir / name)
        written.append(outdir / name)

    # kimictx context-variant matrix (separate run)
    kx = load_kimictx_episodes(KIMICTX_ROOT / "episode_runs.jsonl")
    kxm = kx.pivot_table(index="task_id", columns="config", values="success", aggfunc="mean")
    kxm = kxm.reindex(index=_order_rows_by_opt(kx, kxm.index), columns=KIMICTX_CONFIGS)
    _render_matrix(kxm, "Kimictx run — per-maze solve by context variant",
                   outdir / "nb_matrix_kimictx.png")
    written.append(outdir / "nb_matrix_kimictx.png")

    written += render_landing(ep, outdir)
    written += render_deathcells(outdir)
    return written


if __name__ == "__main__":
    paths = render_all("analysis/notion/Experiment 3 — MultiNet Conditional Eval/figures")
    print(f"\n{len(paths)} figures written")
