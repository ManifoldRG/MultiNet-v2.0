"""Side-by-side 2D (MiniGrid) vs 3D (MuJoCo) renders of the PR #57 tiles.

For each maze: the 2D top-down frame beside the 3D top_down frame, then one
row per tile (portal end, kill cell, frozen tile, rotating tile) showing the
2D glyph and the 3D first-person view from the neighbouring cell, facing it.

    python -m scripts.compare_2d_3d_tiles \\
        --mazes 'ogbench/ogbench/procgen/maze_jsons/M7/*.json' --out /tmp/compare
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from gridworld.backends.base import GridState  # noqa: E402
from gridworld.render3d.scene import wall_cells  # noqa: E402
from gridworld.task_spec import TaskSpecification  # noqa: E402

DIRECTION_NAMES = "ESWN"  # GridState.agent_direction 0..3
# Viewer placement preference: stand west facing east, north facing south, ...
_VIEWPOINTS = (((-1, 0), 0), ((0, -1), 1), ((1, 0), 2), ((0, 1), 3))
LABEL_H = 20
GAP = 8


@dataclass(frozen=True)
class TileEntry:
    label: str
    cell: tuple[int, int]


def tile_catalog(spec: TaskSpecification) -> list[TileEntry]:
    mech = spec.mechanisms
    entries: list[TileEntry] = []
    for tp in mech.teleporters:
        for end, pos in (("A", tp.position_a), ("B", tp.position_b)):
            entries.append(TileEntry(f"portal {tp.id} ({tp.color}) {end}", (pos.x, pos.y)))
    for i, p in enumerate(mech.kill_cells):
        entries.append(TileEntry(f"kill cell {i}", (p.x, p.y)))
    for i, p in enumerate(mech.frozen_tiles):
        entries.append(TileEntry(f"frozen tile {i}", (p.x, p.y)))
    for i, (p, d) in enumerate(zip(mech.rotating_tiles, mech.rotating_initial_directions)):
        entries.append(TileEntry(f"rotating tile {i} ({DIRECTION_NAMES[int(d)]})", (p.x, p.y)))
    return entries


def facing_pose(spec: TaskSpecification, cell: tuple[int, int]) -> tuple[tuple[int, int], int]:
    """(viewer cell, direction) of a free neighbour of ``cell`` looking at it."""
    walls = wall_cells(spec)
    width, height = spec.maze.dimensions
    x, y = cell
    for (dx, dy), direction in _VIEWPOINTS:
        viewer = (x + dx, y + dy)
        if 0 <= viewer[0] < width and 0 <= viewer[1] < height and viewer not in walls:
            return viewer, direction
    raise ValueError(f"no free neighbour of {cell} in {spec.task_id!r}")


def initial_state(spec: TaskSpecification) -> GridState:
    """The episode's reset state (keys on the floor, initial switches and gates)."""
    from gridworld.backends.minigrid_backend import MiniGridBackend

    backend = MiniGridBackend(render_mode="rgb_array")
    try:
        backend.configure(spec)
        _frame, state, _info = backend.reset(seed=spec.seed)
        return state
    finally:
        backend.close()


def _render_2d(spec: TaskSpecification, tile_size: int) -> tuple[Image.Image, GridState, dict[str, bool]]:
    from gridworld.backends.minigrid_backend import MiniGridBackend

    backend = MiniGridBackend(render_mode="rgb_array")
    try:
        backend.configure(spec)
        _frame, state, _info = backend.reset(seed=spec.seed)
        backend.env.tile_size = tile_size
        frame = Image.fromarray(backend.env.get_frame(highlight=False, tile_size=tile_size))
        return frame, state, backend.door_states()
    finally:
        backend.close()


def _label_row(images: list[Image.Image], labels: list[str], height: int) -> Image.Image:
    scaled = [im.resize((round(im.width * height / im.height), height), Image.NEAREST) for im in images]
    row = Image.new("RGB", (sum(im.width for im in scaled) + GAP * (len(scaled) - 1), height + LABEL_H), "white")
    draw = ImageDraw.Draw(row)
    x = 0
    for im, text in zip(scaled, labels):
        draw.text((x + 4, 2), text, fill="black")
        row.paste(im, (x, LABEL_H))
        x += im.width + GAP
    return row


def _stack(rows: list[Image.Image]) -> Image.Image:
    width = max(r.width for r in rows)
    sheet = Image.new("RGB", (width, sum(r.height for r in rows) + GAP * (len(rows) - 1)), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + GAP
    return sheet


def compare_maze(path: Path, out_dir: Path, *, resolution: int, tile_size: int) -> dict:
    from gridworld.render3d.renderer import SceneRenderer

    spec = TaskSpecification.from_json(str(path))
    maze_dir = out_dir / spec.task_id
    (maze_dir / "tiles").mkdir(parents=True, exist_ok=True)
    rotators = tuple(int(d) for d in spec.mechanisms.rotating_initial_directions)

    frame_2d, start, doors = _render_2d(spec, tile_size)  # the same reset state feeds both renderers
    frame_2d.save(maze_dir / "topdown_2d.png")
    top = SceneRenderer(spec, camera="top_down", resolution=resolution)
    try:
        frame_3d = Image.fromarray(top.render(start, doors, rotators=rotators))
    finally:
        top.close()
    frame_3d.save(maze_dir / "topdown_3d.png")

    rows = [_label_row([frame_2d, frame_3d], [f"{spec.task_id} | 2D MiniGrid", "3D top_down"], resolution)]
    tiles: list[dict] = []
    eye = SceneRenderer(spec, camera="first_person", resolution=resolution)
    try:
        for entry in tile_catalog(spec):
            viewer, direction = facing_pose(spec, entry.cell)
            state = dataclasses.replace(start, agent_position=viewer, agent_direction=direction)
            fp = Image.fromarray(eye.render(state, doors, rotators=rotators))
            x, y = entry.cell
            glyph = frame_2d.crop((x * tile_size, y * tile_size, (x + 1) * tile_size, (y + 1) * tile_size))
            stem = entry.label.replace(" ", "_").replace("(", "").replace(")", "")
            fp_png = maze_dir / "tiles" / f"{stem}_first_person.png"
            glyph_png = maze_dir / "tiles" / f"{stem}_2d.png"
            fp.save(fp_png)
            glyph.save(glyph_png)
            rows.append(_label_row(
                [glyph, fp],
                [f"{entry.label} @ {entry.cell} | 2D glyph", f"3D first_person from {viewer} facing {DIRECTION_NAMES[direction]}"],
                resolution // 2,
            ))
            tiles.append({
                "label": entry.label,
                "cell": list(entry.cell),
                "viewer_cell": list(viewer),
                "facing": DIRECTION_NAMES[direction],
                "first_person_png": str(fp_png.relative_to(out_dir)),
                "glyph_2d_png": str(glyph_png.relative_to(out_dir)),
            })
    finally:
        eye.close()
    _stack(rows).save(maze_dir / "sheet.png")
    return {
        "maze": str(path),
        "task_id": spec.task_id,
        "sheet_png": str((maze_dir / "sheet.png").relative_to(out_dir)),
        "topdown_2d_png": str((maze_dir / "topdown_2d.png").relative_to(out_dir)),
        "topdown_3d_png": str((maze_dir / "topdown_3d.png").relative_to(out_dir)),
        "tiles": tiles,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="2D vs 3D renders of the PR #57 tiles.")
    parser.add_argument("--mazes", action="append", default=[], required=True, help="glob; repeatable")
    parser.add_argument("--resolution", type=int, default=512, help="3D frame size (px)")
    parser.add_argument("--tile-size", type=int, default=32, help="2D MiniGrid tile size (px)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    paths: list[Path] = []
    for pattern in args.mazes:
        paths.extend(Path(p) for p in sorted(glob.glob(pattern, recursive=True)))
    if not paths:
        parser.error("no mazes matched")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = [compare_maze(p, out_dir, resolution=args.resolution, tile_size=args.tile_size) for p in paths]
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {len(index)} comparison sheets to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
