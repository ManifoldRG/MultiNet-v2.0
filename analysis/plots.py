"""Reproducible figure generation for the conditional-eval analysis.

Renders all figures under ``analysis/figures/`` from the cached parquets built by
``analysis.data`` / ``analysis.queries_enrich`` / the Phase-1..5 analysis. Palette is the
data-viz reference instance (validated): categorical model hues Claude=blue #2a78d6,
Kimi=orange #eb6834, Qwen=violet #4a3aa7; sequential = blue ramp. Figures render on the
light chart surface (#fcfcfb) with recessive grid/axes and direct labels.

Coordinate note: the prompt reports position as ``(row, col)`` tuples, so the enriched
``pos_x``/``pos_y`` are (row=y, col=x). For maze overlays we plot x=col=pos_y, y=row=pos_x.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from analysis.data import load_episodes, is_clean, CONDITIONAL_CONFIGS
from analysis.mazes import maze_features, load_maze_spec

FIG = Path("analysis/figures")
CACHE = Path("analysis/.cache")

# ---- palette (validated reference instance) ----
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; AXIS = "#c3c2b7"
MODEL_C = {"claude-opus-4-8": "#2a78d6", "kimi-k2.6": "#eb6834", "Qwen_Qwen3.6-27B": "#4a3aa7"}
MODEL_LBL = {"claude-opus-4-8": "Claude", "kimi-k2.6": "Kimi", "Qwen_Qwen3.6-27B": "Qwen"}
MODEL_ORDER = ["claude-opus-4-8", "kimi-k2.6", "Qwen_Qwen3.6-27B"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("bluramp", BLUE_RAMP)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


def _ep():
    ep = load_episodes()
    return ep[ep.config.apply(is_clean)].copy()


# 1 — solve-rate heatmap config × model -----------------------------------------
def fig_solve_heatmap():
    ep = _ep()
    piv = ep.pivot_table(index="config", columns="model", values="success", aggfunc="mean")
    piv["pooled"] = ep.groupby("config").success.mean()
    piv = piv.sort_values("pooled", ascending=False)
    cols = MODEL_ORDER + ["pooled"]
    piv = piv[cols]
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    im = ax.imshow(piv.values, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([MODEL_LBL.get(c, "Pooled") for c in cols], fontsize=10)
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=9.5)
    for i in range(len(piv)):
        for j in range(len(cols)):
            v = piv.values[i, j]
            ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                    color="white" if v > 0.55 else INK, fontsize=9.5,
                    fontweight="bold" if cols[j] == "pooled" else "normal")
    ax.set_title("Solve rate (%) — condition × model", fontweight="bold", color=INK, pad=10)
    ax.axvline(2.5, color=SURFACE, lw=3)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.outline.set_visible(False)
    cb.set_label("solve rate", color=INK2)
    _save(fig, "fig01_solve_heatmap.png")


# 2 — model solve rate with bootstrap CI ----------------------------------------
def fig_model_ci():
    ep = _ep(); rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for i, m in enumerate(MODEL_ORDER):
        x = ep[ep.model == m].success.values.astype(float)
        bs = rng.choice(x, (5000, len(x))).mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ax.bar(i, x.mean(), color=MODEL_C[m], width=0.62, zorder=2)
        ax.errorbar(i, x.mean(), yerr=[[x.mean()-lo], [hi-x.mean()]], color=INK,
                    capsize=5, lw=1.6, zorder=3)
        ax.text(i, hi+0.02, f"{x.mean()*100:.1f}%", ha="center", color=INK, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels([MODEL_LBL[m] for m in MODEL_ORDER])
    ax.set_ylim(0, 0.75); ax.set_ylabel("overall solve rate")
    ax.set_yticks(np.arange(0, 0.76, 0.25)); ax.set_yticklabels([f"{int(t*100)}%" for t in np.arange(0,0.76,0.25)])
    ax.set_title("Overall solve rate  (95% bootstrap CI, n=180)", fontweight="bold", color=INK)
    ax.annotate("Kimi vs Qwen: n.s.\n(+6.7pp [-16,+3])", xy=(1.5, 0.44), ha="center",
                fontsize=8.5, color=INK2)
    ax.axhline(0, color=AXIS, lw=1)
    _save(fig, "fig02_model_ci.png")


# 3 — difficulty curve: solve vs optimal_steps per model ------------------------
def fig_difficulty_curve():
    from sklearn.linear_model import LogisticRegression
    ep = _ep()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    xs = np.linspace(ep.optimal_steps.min(), ep.optimal_steps.max(), 100)
    for m in MODEL_ORDER:
        g = ep[ep.model == m]
        pm = g.groupby("optimal_steps").success.mean()
        ax.scatter(pm.index, pm.values, color=MODEL_C[m], s=34, alpha=0.55, zorder=2,
                   edgecolor=SURFACE, linewidth=0.6)
        clf = LogisticRegression().fit(g[["optimal_steps"]], g.success.astype(int))
        p = clf.predict_proba(xs.reshape(-1, 1))[:, 1]
        ax.plot(xs, p, color=MODEL_C[m], lw=2.2, zorder=3, label=MODEL_LBL[m])
    ax.set_xlabel("optimal path length (BFS steps)"); ax.set_ylabel("P(solve)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks(np.arange(0,1.01,0.25)); ax.set_yticklabels([f"{int(t*100)}%" for t in np.arange(0,1.01,0.25)])
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_title("Difficulty is path length  (r = -0.84)", fontweight="bold", color=INK)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    _save(fig, "fig03_difficulty_curve.png")


# 4 — mechanism confound scatter ------------------------------------------------
def fig_mechanism_confound():
    ep = _ep(); mf = maze_features().drop_duplicates("task_id")
    fids = sorted(mf.task_id, key=len, reverse=True)
    def match(t):
        return next((f for f in fids if t.endswith(f)), None)
    per = ep.groupby("task_id").agg(solve=("success", "mean"), opt=("optimal_steps", "first")).reset_index()
    per["mech"] = per.task_id.map(lambda t: mf.set_index("task_id").mechanism_count.get(match(t), np.nan))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    sc = ax.scatter(per.opt, per.solve, c=per.mech, cmap=SEQ, s=130, vmin=0, vmax=4,
                    edgecolor=INK, linewidth=0.6, zorder=3)
    for _, r in per.iterrows():
        if r.opt >= 60 or r.solve > 0.8:
            lbl = r.task_id.split("_")[-1] if "corridor" not in r.task_id else r.task_id.split("_")[-2]
            ax.annotate(("winding" if "winding" in r.task_id else ("s5-corridor" if "s5" in r.task_id else ("empty" if "empty" in r.task_id else lbl))),
                        (r.opt, r.solve), fontsize=8, color=INK2, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("optimal path length"); ax.set_ylabel("maze solve rate")
    ax.set_yticks(np.arange(0,1.01,0.25)); ax.set_yticklabels([f"{int(t*100)}%" for t in np.arange(0,1.01,0.25)])
    ax.grid(color=GRID, lw=0.8)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04); cb.outline.set_visible(False)
    cb.set_label("mechanism count", color=INK2)
    ax.set_title("The confound: hardest mazes have ZERO mechanisms", fontweight="bold", color=INK, fontsize=11.5)
    _save(fig, "fig04_mechanism_confound.png")


# 5 — token efficiency frontier -------------------------------------------------
def fig_token_frontier():
    ep = _ep()
    g = ep.groupby("config").agg(solve=("success", "mean"), tot=("tokens", "sum"),
                                 sv=("success", "sum"))
    g["tps"] = g.tot / g.sv
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.scatter(g.tps, g.solve, s=90, color="#2a78d6", edgecolor=INK, linewidth=0.6, zorder=3)
    for c, r in g.iterrows():
        ax.annotate(c, (r.tps, r.solve), fontsize=8, color=INK2, xytext=(6, 2),
                    textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("tokens per SUCCESS  (log scale)")
    ax.set_ylabel("solve rate"); ax.grid(color=GRID, lw=0.8, which="both")
    ax.set_yticks(np.arange(0,0.76,0.25)); ax.set_yticklabels([f"{int(t*100)}%" for t in np.arange(0,0.76,0.25)])
    ax.set_title("Efficiency frontier — cost per success vs yield", fontweight="bold", color=INK)
    _save(fig, "fig05_token_frontier.png")


# 6 — failure-cluster scatter ---------------------------------------------------
def fig_failure_clusters():
    m = pd.read_parquet(CACHE / "episode_behavior.parquet")
    m["blocked_rate"] = m.blocked / m.moves.replace(0, np.nan)
    m["revisit_rate"] = m.revisits / m.n_q
    f = m[~m.success].dropna(subset=["blocked_rate"])
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for mo in MODEL_ORDER:
        s = f[f.model == mo]
        ax.scatter(s.blocked_rate, s.revisit_rate, color=MODEL_C[mo], alpha=0.7, s=42,
                   edgecolor=SURFACE, linewidth=0.5, label=MODEL_LBL[mo])
    ax.axvspan(0.5, 1.02, color="#e34948", alpha=0.06)
    ax.text(0.75, 0.05, "WALL-BANGER\nzone", color="#b3312f", fontsize=8.5, ha="center")
    ax.text(0.12, 0.95, "WANDERER\nzone", color=INK2, fontsize=8.5, ha="center")
    ax.set_xlabel("BLOCKED-move rate"); ax.set_ylabel("cell-revisit rate")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(color=GRID, lw=0.8)
    ax.set_title("Failure signatures — every failed episode", fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=9.5, loc="lower left")
    _save(fig, "fig06_failure_clusters.png")


# 7 — spatial death heatmaps ----------------------------------------------------
_HEAT_MAZES = [
    ("conditional_s_s5_14x14_corridor_1", "mazes/exp_maze_jsons/S5/14x14_corridor_1.json", "s5 corridor (opt 89)"),
    ("validation_10_v02_winding_corridor", "mazes/validation_10/V02_winding_corridor.json", "v02 winding (opt 61)"),
    ("conditional_m_m1_10x10_corridor_kr_0", "mazes/exp_maze_jsons/M1/10x10_corridor_kr_0.json", "m1 corridor (opt 44)"),
]
def fig_spatial_heatmaps():
    pos = pd.read_parquet(CACHE / "positions.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    for ax, (tid, path, title) in zip(axes, _HEAT_MAZES):
        sp = load_maze_spec(path); W, H = sp.maze.dimensions
        walls = {(w.x, w.y) for w in sp.maze.walls}
        goal = (sp.goal.target.x, sp.goal.target.y); start = (sp.maze.start.x, sp.maze.start.y)
        ax.set_facecolor(SURFACE)
        # open cells = light surface (drawn as the axes bg); walls = grey blocks
        for wx in range(1, W + 1):
            for wy in range(1, H + 1):
                if (wx, wy) in walls:
                    ax.add_patch(plt.Rectangle((wx-0.5, wy-0.5), 1, 1, color="#cfcec7",
                                               ec=SURFACE, lw=0.5, zorder=1))
        # dwell overlay (pos_x=row=y, pos_y=col=x) -> cell (col, row)
        dwell = {}
        sub = pos[pos.task_id == tid]
        for _, r in sub.iterrows():
            gx, gy = int(r.pos_y), int(r.pos_x)
            if 1 <= gx <= W and 1 <= gy <= H and (gx, gy) not in walls:
                dwell[(gx, gy)] = dwell.get((gx, gy), 0) + 1
        vmax = max(dwell.values()) if dwell else 1
        norm = matplotlib.colors.LogNorm(vmin=1, vmax=vmax)
        for (gx, gy), n in dwell.items():
            ax.add_patch(plt.Rectangle((gx-0.5, gy-0.5), 1, 1, color=SEQ(norm(n)),
                                       ec=SURFACE, lw=0.5, zorder=2))
        ax.scatter(*goal, marker="*", s=300, color="#0ca30c", edgecolor="white", linewidth=1.2, zorder=6)
        ax.scatter(*start, marker="s", s=80, facecolor="none", edgecolor=INK, linewidth=1.8, zorder=6)
        ax.set_xlim(0.4, W+0.6); ax.set_ylim(H+0.6, 0.4); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=11, color=INK, pad=6)
        for s in ax.spines.values(): s.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=SEQ, norm=matplotlib.colors.LogNorm(vmin=1, vmax=vmax))
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01); cb.outline.set_visible(False)
    cb.set_label("visit density (log)", color=INK2)
    fig.suptitle("Where episodes get stuck — dwell density on the hardest mazes\n"
                 "★ goal   ▫ start   grey = wall", fontweight="bold", color=INK, fontsize=13, y=1.04)
    fig.savefig(FIG / "fig07_spatial_heatmaps.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print("wrote", FIG / "fig07_spatial_heatmaps.png")


# 8 — BLOCKED rate + toggle spam ------------------------------------------------
def fig_blocked_toggle():
    tr = pd.read_parquet(CACHE / "trajectories.parquet")
    tr["blocked_rate"] = tr.blocked / tr.moves.replace(0, np.nan)
    piv = tr.pivot_table(index="config", columns="model", values="blocked_rate", aggfunc="median")
    piv = piv.reindex(columns=MODEL_ORDER).loc[[c for c in CONDITIONAL_CONFIGS if c in piv.index]]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    y = np.arange(len(piv)); h = 0.26
    for i, m in enumerate(MODEL_ORDER):
        ax.barh(y + (1-i)*h, piv[m].values, height=h, color=MODEL_C[m], label=MODEL_LBL[m], zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(piv.index, fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel("median BLOCKED-move rate")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_title("Wall-banging by condition  (move blocked, position unchanged)", fontweight="bold", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9.5, ncol=3, loc="lower right")
    _save(fig, "fig08_blocked_by_config.png")


# 9 — parse_ok by step (no decay) -----------------------------------------------
def fig_parse_decay():
    q = pd.read_parquet(CACHE / "queries_enriched.parquet")
    q = q[q.config.apply(is_clean)].copy()
    order = ["1-10", "11-25", "26-50", "51-100", "100+"]
    q["bucket"] = pd.cut(q.env_step, [0, 10, 25, 50, 100, 300], labels=order)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for m in MODEL_ORDER:
        s = q[q.model == m].groupby("bucket", observed=True).parse_ok.mean().reindex(order)
        ax.plot(range(len(order)), s.values, color=MODEL_C[m], lw=2.2, marker="o", ms=6,
                label=MODEL_LBL[m])
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
    ax.set_ylim(0.94, 1.005); ax.set_xlabel("env step within episode"); ax.set_ylabel("parse-OK rate")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_title("No late-episode decay — failures are format-fumbles at the start",
                 fontweight="bold", color=INK, fontsize=10.5)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    _save(fig, "fig09_parse_decay.png")


# 10 — qry_subgoal delegation ---------------------------------------------------
def fig_subgoal_delegation():
    q = pd.read_parquet(CACHE / "queries_enriched.parquet")
    sg = q[q.config == "qry_subgoal"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    data = [np.log10(sg[sg.model == m].n_actions.clip(lower=1)) for m in MODEL_ORDER]
    parts = ax.violinplot(data, showmedians=True, widths=0.8)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(MODEL_C[MODEL_ORDER[i]]); b.set_alpha(0.55); b.set_edgecolor(INK)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        parts[key].set_color(INK2)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels([MODEL_LBL[m] for m in MODEL_ORDER])
    ax.set_ylabel("actions emitted per query  (log10)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_title("qry_subgoal: the MODEL emits the full route\n(harness does not pathfind)",
                 fontweight="bold", color=INK, fontsize=10.5)
    _save(fig, "fig10_subgoal_delegation.png")


# 11 — world-model fidelity: perception vs transformation ----------------------
def fig_world_model():
    from analysis.beliefs import analyse, summary
    s = summary(analyse())
    metrics = [("self-state\necho", "echo_face_acc", None),
               ("wall-map\nrecall", "wall_err_rate", "inv"),
               ("single-step\ntranslation", "trans_err_rate", "inv"),
               ("mental\nrotation", "rot_err_rate", "inv")]
    models = ["kimi-k2.6", "Qwen_Qwen3.6-27B"]  # Claude has ~no minable reasoning
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    x = np.arange(len(metrics)); w = 0.36
    for i, m in enumerate(models):
        vals = []
        for _, col, inv in metrics:
            v = s.loc[m, col]
            vals.append((1 - v) if inv == "inv" else v)
        ax.bar(x + (i - 0.5) * w, vals, width=w, color=MODEL_C[m], label=MODEL_LBL[m], zorder=2)
        for xi, v in zip(x, vals):
            ax.text(xi + (i - 0.5) * w, v + 0.015, f"{v*100:.0f}", ha="center", fontsize=9,
                    color=INK, fontweight="bold")
    ax.axhspan(0, 0.6, color="#e34948", alpha=0.05, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels([m[0] for m in metrics])
    ax.set_ylim(0, 1.08); ax.set_ylabel("belief accuracy vs ground truth")
    ax.set_yticks(np.arange(0, 1.01, 0.25)); ax.set_yticklabels([f"{int(t*100)}%" for t in np.arange(0,1.01,0.25)])
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 4)))
    ax.text(2.35, 0.51, "chance", fontsize=8, color=MUTED)
    ax.set_title("Perception is fine; transformation is not\nmental rotation fails ~46% of planned turns",
                 fontweight="bold", color=INK, fontsize=11.5)
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    _save(fig, "fig11_world_model.png")


# 12 — rotation error by condition + error shape ------------------------------
def fig_rotation_detail():
    b = pd.read_parquet(CACHE / "beliefs.parquet")
    cc = (b.groupby("config").apply(lambda x: pd.Series(
        {"n": x.rot_n.sum(), "err": x.rot_err.sum() / max(x.rot_n.sum(), 1)}), include_groups=False))
    cc = cc[cc.n >= 50].sort_values("err")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.4, 1]})
    y = np.arange(len(cc))
    ax1.barh(y, cc.err.values, color="#4a3aa7", zorder=2)
    for yi, v in zip(y, cc.err.values):
        ax1.text(v + 0.01, yi, f"{v*100:.0f}%", va="center", fontsize=9, color=INK)
    ax1.set_yticks(y); ax1.set_yticklabels(cc.index, fontsize=9); ax1.invert_yaxis()
    ax1.axvline(0.5, color=MUTED, lw=1, ls=(0, (4, 4))); ax1.set_xlim(0, 0.72)
    ax1.set_xlabel("mental-rotation error rate"); ax1.grid(axis="x", color=GRID, lw=0.8)
    ax1.set_title("More in-head planning → more rotation error", fontweight="bold", color=INK, fontsize=11)
    # error shape
    tot = b.rot_err.sum()
    parts = [("inverted turn\n(L/R swap)", b.rot_inverted.sum() / tot, "#e34948"),
             ("claimed no\nrotation", b.rot_noop.sum() / tot, "#eda100"),
             ("perpendicular", b.rot_other.sum() / tot, "#898781")]
    left = 0
    for lbl, frac, col in parts:
        ax2.barh(0, frac, left=left, color=col, zorder=2, edgecolor=SURFACE)
        ax2.text(left + frac / 2, 0, f"{frac*100:.0f}%", ha="center", va="center",
                 color="white", fontweight="bold", fontsize=10)
        ax2.text(left + frac / 2, 0.45, lbl, ha="center", va="bottom", fontsize=8.5, color=INK2)
        left += frac
    ax2.set_xlim(0, 1); ax2.set_ylim(-0.6, 1.1); ax2.axis("off")
    ax2.set_title("How the rotation is wrong", fontweight="bold", color=INK, fontsize=11)
    _save(fig, "fig12_rotation_detail.png")


# 13 — cost: recoverable waste + early-abort tradeoff --------------------------
def fig_cost_waste():
    w = pd.read_parquet(CACHE / "waste.parquet")
    fail = w[~w.success]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    # panel 1: waste-token fraction by model
    wf = fail.groupby("model").apply(lambda x: x.waste_tok.sum() / x.total_tok.sum(), include_groups=False)
    for i, m in enumerate(MODEL_ORDER):
        ax1.bar(i, wf[m], color=MODEL_C[m], width=0.62, zorder=2)
        ax1.text(i, wf[m] + 0.012, f"{wf[m]*100:.0f}%", ha="center", fontweight="bold", color=INK)
    ax1.set_xticks(range(3)); ax1.set_xticklabels([MODEL_LBL[m] for m in MODEL_ORDER])
    ax1.set_ylim(0, 0.65); ax1.set_ylabel("share of failed-episode tokens")
    ax1.set_yticks(np.arange(0, 0.61, 0.2)); ax1.set_yticklabels([f"{int(t*100)}%" for t in np.arange(0,0.61,0.2)])
    ax1.grid(axis="y", color=GRID, lw=0.8)
    ax1.set_title("Tokens burned AFTER the last new cell\n(pure thrashing, recoverable)",
                  fontweight="bold", color=INK, fontsize=10.5)
    # panel 2: abort tradeoff (frozen-in-place K)
    q = pd.read_parquet(CACHE / "queries_enriched.parquet")
    q = q[q.config.apply(is_clean) & q.pos_x.notna()].sort_values(["episode_id", "env_step", "query_index"])
    beh = pd.read_parquet(CACHE / "episode_behavior.parquet")[["episode_id", "success"]]
    q = q.merge(beh, on="episode_id", how="inner")
    q["cell"] = list(zip(q.pos_x.astype(int), q.pos_y.astype(int)))
    eps = []
    for _, g in q.groupby("episode_id"):
        g = g.reset_index(drop=True); cells = g.cell.tolist(); tok = g.out_tokens.fillna(0).astype(float).values
        frozen = 0; ab = {}
        for i in range(len(cells)):
            frozen = frozen + 1 if (i > 0 and cells[i] == cells[i-1]) else 0
            for K in range(4, 26, 2):
                if K not in ab and frozen >= K: ab[K] = i
        eps.append({"success": g.success.iloc[0], "total": tok.sum(), "cum": np.cumsum(tok), "ab": ab})
    d = pd.DataFrame(eps); total = d.total.sum(); nsucc = d.success.sum()
    Ks = list(range(4, 26, 2)); saved = []; lost = []
    for K in Ks:
        s = sum(r["total"] - r["cum"][r["ab"][K]] for _, r in d.iterrows() if K in r["ab"])
        l = sum(1 for _, r in d.iterrows() if K in r["ab"] and r["success"])
        saved.append(s / total * 100); lost.append(l / nsucc * 100)
    ax2.plot(lost, saved, "-o", color="#2a78d6", ms=5, lw=2, zorder=3)
    for K, x, yv in zip(Ks, lost, saved):
        if K in (6, 12, 20): ax2.annotate(f"K={K}", (x, yv), fontsize=8.5, color=INK2, xytext=(5, -2), textcoords="offset points")
    ax2.set_xlabel("successes lost (%)"); ax2.set_ylabel("output tokens saved (%)")
    ax2.grid(color=GRID, lw=0.8)
    ax2.set_title("Early-abort tradeoff (freeze ≥K steps)\nK=20: save 23% for 2% of successes",
                  fontweight="bold", color=INK, fontsize=10.5)
    _save(fig, "fig13_cost_waste.png")


ALL = [fig_solve_heatmap, fig_model_ci, fig_difficulty_curve, fig_mechanism_confound,
       fig_token_frontier, fig_failure_clusters, fig_spatial_heatmaps, fig_blocked_toggle,
       fig_parse_decay, fig_subgoal_delegation, fig_world_model,
       fig_rotation_detail, fig_cost_waste]


if __name__ == "__main__":
    for f in ALL:
        try:
            f()
        except Exception as e:  # keep going; report which figure broke
            import traceback
            print("FAILED", f.__name__, e); traceback.print_exc()
