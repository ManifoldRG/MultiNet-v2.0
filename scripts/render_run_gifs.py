"""Two-panel GIFs for local smoke episodes.

Uses the results-repo NL renderer: left is the coordinate text the model
read, right is the top-down maze and the trail so far.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

from PIL import ImageFont

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "artifacts" / "runs"
OUT = REPO / "artifacts" / "gifs"
RENDERER = (
    Path.home()
    / "MIT Dropbox"
    / "Arushi Jain"
    / "Multinet-v2-results"
    / "r2-20261025"
    / "analysis-nl"
    / "sean"
    / "render_nl_gifs.py"
)

SHORT = {
    "25x25_dense_kr_sg_kb_0": "M11",
    "25x25_dense_kr_sg_kb_irreversible_exclusive_key_0": "M12 exclusive key",
    "25x25_dense_kr_sg_kb_planning_drop_0": "M12 planning drop",
    "25x25_dense_kr_sg_kb_mix_all_0": "M14 mix all",
}


def _load_renderer():
    # The venv's matplotlib native module is blocked. The renderer only needs
    # figstyle.sanitize and a font path, so both imports are stubbed.
    figstyle = types.ModuleType("figstyle")
    figstyle.sanitize = lambda text: text.replace("\u2014", "-").replace("\u2013", "-")
    sys.modules["figstyle"] = figstyle
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.get_data_path = lambda: ""
    sys.modules["matplotlib"] = matplotlib

    spec = importlib.util.spec_from_file_location("render_nl_gifs", RENDERER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def font(size, bold=False):
        name = "arialbd.ttf" if bold else "arial.ttf"
        return ImageFont.truetype(str(Path(r"C:\Windows\Fonts") / name), size)

    module._font = font
    module.r1._font = font
    module.text_panel = lambda observation, width, height: _fitted_text(module, observation, width, height)
    return module


def _fitted_text(gifs, observation, width, height):
    """Same two-column observation panel, shrunk until every line fits."""
    observation = gifs.c.figstyle.sanitize(observation)
    head, maps, wide = [], [], []
    for line in observation.split("\n"):
        if not line.strip():
            continue
        if line.startswith("There is "):
            item = gifs._short_map_line(line)
            (wide if "teleporter" in item else maps).append(item)
        else:
            head.append(line)
    pad = 14
    inner = width - 2 * pad
    col_w = (inner - 18) // 2
    for size in (15, 14, 13, 12, 11, 10):
        img = gifs.Image.new("RGB", (width, height), "#f6f6f4")
        draw = gifs.ImageDraw.Draw(img)
        font, font_b = gifs._font(size), gifs._font(size, bold=True)
        map_size = max(9, size - 2)
        map_font, map_b = gifs._font(map_size), gifs._font(map_size, bold=True)
        head_lines = []
        for line in head:
            head_lines.extend(gifs._wrap_runs(draw, line, font, font_b, inner))
        map_lines = []
        for item in maps:
            map_lines.extend(gifs._wrap_runs(draw, item, map_font, map_b, col_w))
        wide_lines = []
        for item in wide:
            wide_lines.extend(gifs._wrap_runs(draw, item, map_font, map_b, inner))
        line_h = int(font.size * 1.38)
        map_h = int(map_font.size * 1.32)
        rows = (len(map_lines) + 1) // 2 if map_lines else 0
        y = pad + len(head_lines) * line_h
        if map_lines:
            y += 4 + rows * map_h
        if wide_lines:
            y += 4 + len(wide_lines) * map_h
        if y <= height - pad or size == 10:
            y = pad
            for runs in head_lines:
                gifs._paint(draw, pad, y, runs)
                y += line_h
            if map_lines:
                y += 4
                for i, runs in enumerate(map_lines):
                    col = 0 if i < rows else 1
                    row = i if col == 0 else i - rows
                    gifs._paint(draw, pad + col * (col_w + 18), y + row * map_h, runs)
                y += rows * map_h
            if wide_lines:
                y += 4
                for runs in wide_lines:
                    gifs._paint(draw, pad, y, runs)
                    y += map_h
            return img
    return img


def _xy(value):
    return (int(value[0]), int(value[1]))


def _with_live_tiles(spec, state):
    """Walls the static map does not list: sealed floor and closed phased cells.

    Chain portals store every pad in ``positions``; the renderer draws the two ends.
    """
    maze = dict(spec["maze"])
    walls = [list(cell) for cell in maze.get("walls") or []]
    walls.extend(list(cell) for cell in (state.get("sealed") or []))
    tick = int(state.get("rule_tick") or 0)
    mechanisms = dict(spec.get("mechanisms") or {})
    for patch in mechanisms.get("phased_patches") or []:
        period = int(patch["open_steps"]) + int(patch["closed_steps"])
        if (tick + int(patch["offset"])) % period >= int(patch["open_steps"]):
            walls.extend(list(cell) for cell in patch["cells"])
    maze["walls"] = walls
    teleporters = []
    for portal in mechanisms.get("teleporters") or []:
        teleporters.append(portal)
        ends = portal.get("positions") or []
        known = {tuple(portal["position_a"]), tuple(portal["position_b"])}
        for pad in ends:
            if tuple(pad) not in known:
                extra = dict(portal)
                extra["position_a"] = pad
                extra["position_b"] = pad
                teleporters.append(extra)
    mechanisms["teleporters"] = teleporters
    drawn = dict(spec)
    drawn["maze"] = maze
    drawn["mechanisms"] = mechanisms
    return drawn


def _episode(path: Path, task: str, model: str, gifs) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    steps = []
    for rec in raw["transcript"]:
        if rec.get("kind") != "step":
            continue
        after = rec["state_after"]
        steps.append({
            "env_step": int(rec["step_index"]),
            "action": rec["action"],
            "event_type": rec["event_type"],
            "pos": _xy(after["agent_position"]),
            "pos_before": _xy(rec["state_before"]["agent_position"]),
            "facing": rec["facing_after"],
            "state": after,
            "mechanic": (rec.get("backend_info") or {}).get("mechanic") or rec.get("mechanic") or {},
        })
    spec = raw["task_spec"]
    cap = int(spec["max_steps"])
    end = str(raw["end_reason"]).split(":")[0]
    if raw["success"]:
        outcome = "solved"
    elif end == "stuck":
        outcome = "stuck"
    else:
        outcome = gifs.c.base.OUTCOMES.get(end, "infra_terminated")
    return {
        "task": task,
        "model": model,
        "nomap": False,
        "steps": steps,
        "outcome": outcome,
        "end_reason": raw["end_reason"],
        "n_steps": len(steps),
        "bfs": cap // 3,
        "spec": spec,
        "start": _xy(spec["maze"]["start"]),
        "initial_state": raw["initial_state"],
        "start_map": False,
        "render": {},
        "frames": path.parent / "frames",
        "observations": gifs.c.observations_by_step(raw),
    }


def main() -> int:
    gifs = _load_renderer()
    gifs.c.short_task = lambda task: SHORT.get(task, task)
    gifs.r1.OUTCOME_BANNERS["stuck"] = ("× STUCK", "#d32f2f")
    gifs.OUTCOME_WORD["stuck"] = "stuck"
    original = gifs.c.topdown

    def topdown(spec, state, segments, color, panel, ss=2):
        return original(_with_live_tiles(spec, state), state, segments, color, panel, ss)

    gifs.c.topdown = topdown
    gifs.c.ANALYSIS = REPO
    OUT.mkdir(parents=True, exist_ok=True)
    fonts = gifs.r1.Fonts(gifs.MAZE)
    frame_ms = 100
    written = []
    for episode_path in sorted(RUNS.rglob("episode.json")):
        model = episode_path.parents[2].name
        task = episode_path.parents[4].name
        if model not in gifs.c.MODEL:
            continue
        episode = _episode(episode_path, task, model, gifs)
        problems = []
        seq, _shown = gifs.render_segment(episode, fonts, frame_ms, problems)
        dest = OUT / f"{task}_{model}.gif"
        info = gifs.save(seq, dest, problems)
        if problems:
            print(f"{dest.name}: {problems}", flush=True)
        print(
            f"{dest.name}: {episode['outcome']} {episode['n_steps']} steps, "
            f"{info['frames']} frames, {info['bytes'] / 1e6:.1f} MB",
            flush=True,
        )
        written.append(dest)
    print(f"wrote {len(written)} gifs to {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
