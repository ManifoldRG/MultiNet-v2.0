# render_dataset.py
from __future__ import annotations

import json
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


CELL = 40  # pixels-ish via figure scale


def _extract_payload_fields(payload: dict):
    maze = payload["maze"]
    mechs = payload.get("mechanisms", {})

    width, height = maze["dimensions"]
    walls = {tuple(w) for w in maze["walls"]}
    start = tuple(maze["start"])
    goal = tuple(maze["goal"])

    keys = mechs.get("keys", [])
    doors = mechs.get("doors", [])
    switches = mechs.get("switches", [])
    gates = mechs.get("gates", [])

    return width, height, walls, start, goal, keys, doors, switches, gates


def _row_col_payload_to_xy_payload(payload: dict) -> dict:
    """Convert a row/col payload to renderer-space (x/y) without mutating input."""
    out = deepcopy(payload)
    maze = out.get("maze", {})
    mechs = out.get("mechanisms", {})

    def rc_to_xy(pos):
        r, c = pos
        # Payloads use 1-based (row, col); drawing uses 0-based (x=col, y=row).
        return [c - 1, r - 1]

    dims = maze.get("dimensions")
    if dims and len(dims) == 2:
        rows, cols = dims
        maze["dimensions"] = [cols, rows]

    maze["walls"] = [rc_to_xy(w) for w in maze.get("walls", [])]
    if "start" in maze:
        maze["start"] = rc_to_xy(maze["start"])
    if "goal" in maze:
        maze["goal"] = rc_to_xy(maze["goal"])

    for k in mechs.get("keys", []):
        if "position" in k:
            k["position"] = rc_to_xy(k["position"])
    for d in mechs.get("doors", []):
        if "position" in d:
            d["position"] = rc_to_xy(d["position"])
    for s in mechs.get("switches", []):
        if "position" in s:
            s["position"] = rc_to_xy(s["position"])
    for g in mechs.get("gates", []):
        if "position" in g:
            g["position"] = rc_to_xy(g["position"])

    validation = out.get("validation", {})
    if "optimal_path" in validation:
        validation["optimal_path"] = [rc_to_xy(p) for p in validation.get("optimal_path", [])]
    return out



def _color_to_facecolor(name: str) -> str:
    mapping = {
        "red": "#e74c3c",
        "blue": "#3498db",
        "green": "#2ecc71",
        "yellow": "#f1c40f",
        "purple": "#9b59b6",
        "orange": "#e67e22",
    }
    return mapping.get(name.lower(), "#95a5a6")


def _draw_centered_text(ax, x: int, y: int, height: int, text: str, fontsize: int = 10, color: str = "black"):
    ax.text(
        x + 0.5,
        height - 1 - y + 0.5,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        fontweight="bold",
    )


def _draw_key(ax, x: int, y: int, height: int, color_name: str):
    face = _color_to_facecolor(color_name)
    cy = height - 1 - y + 0.5

    # colored circle badge
    ax.add_patch(Circle((x + 0.5, cy), 0.28, facecolor=face, edgecolor="black", linewidth=1.0))
    # key icon / fallback letter
    ax.text(
        x + 0.5,
        cy,
        "⚷",   # if this glyph looks odd in your env, replace with "K"
        ha="center",
        va="center",
        fontsize=11,
        color="white",
        fontweight="bold",
    )


def _draw_door(ax, x: int, y: int, height: int, color_name: str):
    face = _color_to_facecolor(color_name)
    by = height - 1 - y

    # colored inner door rectangle
    ax.add_patch(
        Rectangle(
            (x + 0.18, by + 0.12),
            0.64,
            0.76,
            facecolor=face,
            edgecolor="black",
            linewidth=1.0,
        )
    )
    # small doorknob
    ax.add_patch(Circle((x + 0.68, by + 0.5), 0.04, facecolor="white", edgecolor="white"))


def _draw_switch(ax, x: int, y: int, height: int, label: str):
    by = height - 1 - y

    ax.add_patch(
        Rectangle(
            (x + 0.15, by + 0.2),
            0.7,
            0.6,
            facecolor="#dfe6e9",
            edgecolor="black",
            linewidth=1.0,
        )
    )
    ax.text(
        x + 0.5,
        by + 0.5,
        label,
        ha="center",
        va="center",
        fontsize=9,
        color="black",
        fontweight="bold",
    )


def _draw_gate(ax, x: int, y: int, height: int, label: str):
    by = height - 1 - y

    # gate bars
    for dx in [0.22, 0.38, 0.54, 0.70]:
        ax.plot([x + dx, x + dx], [by + 0.15, by + 0.85], color="black", linewidth=1.4)
    ax.plot([x + 0.18, x + 0.74], [by + 0.18, by + 0.18], color="black", linewidth=1.4)
    ax.plot([x + 0.18, x + 0.74], [by + 0.82, by + 0.82], color="black", linewidth=1.4)

    ax.text(
        x + 0.5,
        by + 0.5,
        label,
        ha="center",
        va="center",
        fontsize=8,
        color="black",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.8),
    )


_AGENT_FACING_DELTA = {
    "NORTH": (-1, 0),
    "EAST": (0, 1),
    "SOUTH": (1, 0),
    "WEST": (0, -1),
}


def _draw_agent(ax, ar: int, ac: int, height: int, facing: str) -> None:
    """Overlay current agent (row, col) and facing; same cell coords as ``_draw_centered_text``."""
    # GridState uses (row, col). Rendering uses x=col, y=row (inverted vertical axis).
    cx = ac + 0.5
    cy = height - 1 - ar + 0.5
    ax.plot(
        cx,
        cy,
        "o",
        color="black",
        markersize=10,
        zorder=6,
        markeredgecolor="black",
    )
    dr, dc = _AGENT_FACING_DELTA.get(facing, (0, 0))
    if dr == 0 and dc == 0:
        return
    nr, nc = ar + dr, ac + dc
    tip_x = nc + 0.5
    tip_y = height - 1 - nr + 0.5
    ax.annotate(
        "",
        xy=(tip_x, tip_y),
        xytext=(cx, cy),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        zorder=7,
    )


def _extract_optimal_path(payload: dict):
    validation = payload.get("validation", {})
    return [tuple(p) for p in validation.get("optimal_path", [])]






def _draw_optimal_path(ax, path, height: int):
    if not path:
        return

    xs = [x + 0.5 for x, y in path]
    ys = [height - 1 - y + 0.5 for x, y in path]

    ax.plot(
        xs,
        ys,
        linewidth=3.0,
        alpha=0.45,
        zorder=2,
    )

    # mark start of path a little more clearly
    ax.scatter(
        [xs[0]],
        [ys[0]],
        s=35,
        alpha=0.7,
        zorder=3,
    )



def _figure_from_maze_payload(payload: dict, title: str) -> Tuple[Any, Any, int]:
    """Build figure/axes for a maze JSON payload; caller savesfig and closes."""
    payload = _row_col_payload_to_xy_payload(payload)
    width, height, walls, start, goal, keys, doors, switches, gates = _extract_payload_fields(payload)
    optimal_path = _extract_optimal_path(payload)

    fig_w = max(6, width * 0.55)
    fig_h = max(4, height * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # base grid
    for x in range(width):
        for y in range(height):
            is_wall = (x, y) in walls
            facecolor = "black" if is_wall else "white"
            ax.add_patch(
                Rectangle(
                    (x, height - 1 - y),
                    1,
                    1,
                    facecolor=facecolor,
                    edgecolor="lightgray",
                    linewidth=0.8,
                    zorder=0,
                )
            )

    # path overlay first, so icons remain visible above it
    _draw_optimal_path(ax, optimal_path, height)

    # start / goal
    sx, sy = start
    gx, gy = goal
    ax.add_patch(Rectangle((sx, height - 1 - sy), 1, 1, facecolor="#c8f7c5", edgecolor="black", linewidth=1.2, zorder=4))
    ax.add_patch(Rectangle((gx, height - 1 - gy), 1, 1, facecolor="#f7d6c5", edgecolor="black", linewidth=1.2, zorder=4))
    _draw_centered_text(ax, sx, sy, height, "S", fontsize=11)
    _draw_centered_text(ax, gx, gy, height, "G", fontsize=11)

    # keys
    for key in keys:
        x, y = key["position"]
        color_name = key.get("color", "gray")
        _draw_key(ax, x, y, height, color_name)

    # doors
    for door in doors:
        x, y = door["position"]
        color_name = door.get("requires_key", "gray")
        _draw_door(ax, x, y, height, color_name)

    # switches
    for sw in switches:
        x, y = sw["position"]
        _draw_switch(ax, x, y, height, "S")

    # gates
    for gate in gates:
        x, y = gate["position"]
        _draw_gate(ax, x, y, height, "G")

    ax.set_title(title)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    return fig, ax, height


def render_maze_payload(payload: dict, output_path: Path) -> None:
    title = payload.get("task_id", output_path.stem)
    fig, _ax, _height = _figure_from_maze_payload(payload, title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_maze_payload_bytes(
    payload: dict,
    *,
    dpi: int = 150,
    agent_pos: Optional[Tuple[int, int]] = None,
    facing: str = "NORTH",
) -> bytes:
    """Same layout as ``render_maze_payload``, PNG bytes (e.g. NLU live observations)."""
    title = str(payload.get("task_id", "maze"))
    fig, ax, height = _figure_from_maze_payload(payload, title)
    if agent_pos is not None:
        _draw_agent(ax, agent_pos[0], agent_pos[1], height, facing)
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()




def main() -> None:
    input_dir = Path("generated_mazes")
    # input_dir = Path("../nlu_pipeline/nlu_benchmark/sample mazes")
    output_dir = input_dir / "pngs"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(p for p in input_dir.glob("*.json") if p.name != "manifest.json")
    if not json_files:
        print("No maze JSON files found in generated_mazes/")
        return

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            payload = json.load(f)

        out_path = output_dir / f"{jf.stem}.png"
        render_maze_payload(payload, out_path)
        print(f"[OK] rendered {out_path.name}")

    print(f"\nRendered {len(json_files)} PNGs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()