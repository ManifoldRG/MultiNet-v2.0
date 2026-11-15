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
    return module


def _xy(value):
    return (int(value[0]), int(value[1]))


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
