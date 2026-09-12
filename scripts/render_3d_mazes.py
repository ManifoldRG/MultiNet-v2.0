"""Render task-spec mazes through the 3D backend to PNGs.

Phase 0 pilot input (3D roadmap section 3) and a visual smoke test: start
frames or full BFS-plan replays per camera preset, plus an optional contact
sheet.

    python -m scripts.render_3d_mazes --manifest gridworld/fixtures/manifest.json \
        --experiment r1 --camera top_down --camera chase --contact-sheet --out /tmp/render3d
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from gridworld.backends import get_backend  # noqa: E402
from gridworld.baselines import plan_bfs_path  # noqa: E402
from gridworld.render3d.cameras import PRESETS  # noqa: E402
from gridworld.task_spec import TaskSpecification  # noqa: E402
from interface.parser import ACTION_ORDER  # noqa: E402

THUMB = 256
LABEL_H = 18


def resolve_mazes(manifest: str | None, experiment: str | None, patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    if manifest:
        from scripts.run_pipeline import _resolve_source, load_manifest, resolve_task_rows

        manifest_path = Path(manifest)
        catalog = load_manifest(manifest_path)
        for row in resolve_task_rows([experiment] if experiment else ["all"], catalog, manifest_path):
            try:
                paths.append(Path(_resolve_source(row, manifest_path)))
            except FileNotFoundError as exc:
                print(f"Warning: skipping manifest row {row.get('task_id')!r}: {exc}", file=sys.stderr)
                continue
    for pattern in patterns:
        paths.extend(Path(p) for p in sorted(glob.glob(pattern, recursive=True)))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path.resolve() not in seen:
            seen.add(path.resolve())
            unique.append(path)
    return unique


def render_maze(path: Path, cameras: list[str], resolution: int, replay: str, out_dir: Path) -> list[dict]:
    spec = TaskSpecification.from_json(str(path))
    actions = plan_bfs_path(spec).actions if replay == "bfs" else []
    records: list[dict] = []
    for camera in cameras:
        backend = get_backend("mujoco3d", camera=camera, resolution=resolution)
        try:
            backend.configure(spec)
            frame, _state, _info = backend.reset(seed=spec.seed)
            frames = [(0, None, frame)]
            for step, action in enumerate(actions, start=1):
                frame, _reward, terminated, truncated, _state, _info = backend.step(action)
                frames.append((step, ACTION_ORDER[action], frame))
                if terminated or truncated:
                    break
            cam_dir = out_dir / path.stem / camera
            cam_dir.mkdir(parents=True, exist_ok=True)
            for step, action, image in frames:
                png = cam_dir / f"step_{step:03d}.png"
                Image.fromarray(image).save(png)
                records.append({
                    "maze": str(path),
                    "task_id": spec.task_id,
                    "camera": camera,
                    "step": step,
                    "action": action,
                    "png": str(png.relative_to(out_dir)),
                })
        finally:
            backend.close()
    return records


def contact_sheet(records: list[dict], out_dir: Path, cameras: list[str]) -> Path:
    starts = {(r["maze"], r["camera"]): out_dir / r["png"] for r in records if r["step"] == 0}
    mazes = list(dict.fromkeys(r["maze"] for r in records))
    sheet = Image.new("RGB", (THUMB * len(cameras), (THUMB + LABEL_H) * len(mazes)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, maze in enumerate(mazes):
        for col, camera in enumerate(cameras):
            x, y = col * THUMB, row * (THUMB + LABEL_H)
            draw.text((x + 4, y + 2), f"{Path(maze).stem} | {camera}", fill="black")
            sheet.paste(Image.open(starts[(maze, camera)]).resize((THUMB, THUMB)), (x, y + LABEL_H))
    path = out_dir / "contact_sheet.png"
    sheet.save(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render task-spec mazes through the 3D backend.")
    parser.add_argument("--manifest")
    parser.add_argument("--experiment")
    parser.add_argument("--mazes", action="append", default=[], help="glob; repeatable")
    parser.add_argument("--camera", action="append", choices=PRESETS, help="repeatable; default top_down")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--replay", choices=("start", "bfs"), default="start")
    parser.add_argument("--contact-sheet", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.experiment and not args.manifest:
        parser.error("--experiment requires --manifest")

    mazes = resolve_mazes(args.manifest, args.experiment, args.mazes)
    if not mazes:
        parser.error("no mazes selected (use --manifest and/or --mazes)")
    cameras = args.camera or ["top_down"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for path in mazes:
        records.extend(render_maze(path, cameras, args.resolution, args.replay, out_dir))
    (out_dir / "index.json").write_text(json.dumps(records, indent=1))
    if args.contact_sheet:
        print(f"contact sheet: {contact_sheet(records, out_dir, cameras)}")
    print(f"wrote {len(records)} frames ({len(mazes)} mazes x {len(cameras)} cameras) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
