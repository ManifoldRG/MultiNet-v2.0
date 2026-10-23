from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from demo.theme import recolor_walls
from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.task_spec import TaskSpecification

from .models import DistractorMode
from .orchestrator import sample_spec, build_valid_maze_with_retries

OUTPUT_DIR = Path(__file__).resolve().parent / "generated_mazes"
TASK_DIR = OUTPUT_DIR / "task_spec"
PNG_DIR = OUTPUT_DIR / "png"


def to_payload(task: TaskSpecification, report: dict) -> dict[str, Any]:
    payload = task.to_dict()
    payload["validation"] = {
        "is_valid": report["is_valid"],
        "reasons": report["reasons"],
        "optimal_cost": report["solver_result"]["optimal_cost"],
        "optimal_path": [list(pos) for pos in report["solver_result"].get("path", [])],
        "interactions": report["solver_result"].get("interactions", []),
    }
    return payload


def save_task_png(backend: MiniGridBackend, task: TaskSpecification, path: Path) -> None:
    """Write the same full-grid frame the 2D demo shows, including wall recolor."""
    backend.configure(task)
    backend.reset(seed=task.seed)
    frame = recolor_walls(backend.render())
    Image.fromarray(frame).save(path)


def main() -> None:
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    backend = MiniGridBackend(render_mode="rgb_array")

    target_n = 200
    accepted = 0
    seed = 1000
    attempts = 0
    max_attempts = 500

    manifest: list[dict[str, Any]] = []

    while accepted < target_n and attempts < max_attempts:
        attempts += 1
        spec = sample_spec(seed)
        seed += 1

        try:
            task, report, final_spec = build_valid_maze_with_retries(spec, max_retries=10)
        except Exception as exc:
            print(f"[ERROR] seed={spec.seed} backbone={spec.backbone.value} logic={spec.logic_chain.value}: {exc}")
            continue


        payload = to_payload(task, report)
        distractor_tag = (
            "none"
            if final_spec.distractor_mode == DistractorMode.NONE
            else f"{final_spec.distractor_mode.value}{final_spec.max_distractors}"
        )

        stem = (
            f"maze_{accepted:03d}_"
            f"{final_spec.backbone.value}_"
            f"{final_spec.logic_chain.value}_"
            f"{distractor_tag}_"
            f"{final_spec.seed}"
        )
        fname = f"{stem}.json"
        with open(TASK_DIR / fname, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        save_task_png(backend, task, PNG_DIR / f"{stem}.png")

        manifest.append(
            {
                "file": f"task_spec/{fname}",
                "image": f"png/{stem}.png",
                "seed": final_spec.seed,
                "backbone": final_spec.backbone.value,
                "logic_chain": final_spec.logic_chain.value,
                "difficulty_tier": final_spec.difficulty_tier,
                "distractor_mode": final_spec.distractor_mode.value,
                "max_distractors": final_spec.max_distractors,
                "optimal_cost": report["solver_result"]["optimal_cost"],
            }
        )

        accepted += 1
        print(f"[OK] {accepted:02d}/{target_n} saved {stem}")

    with open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nFinished. Accepted {accepted} mazes in {attempts} attempts.")
    print(f"Task specs: {TASK_DIR.resolve()}")
    print(f"PNGs: {PNG_DIR.resolve()}")


if __name__ == "__main__":
    main()