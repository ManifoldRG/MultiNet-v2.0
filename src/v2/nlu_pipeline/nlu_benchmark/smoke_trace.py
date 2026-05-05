"""PNG + text artifacts for smoke scripts (``smoke_bfs``, ``smoke_claude``, …).

Writes ``step_000_reset.png``, ``step_NNN_<ACTION>.png``, ``run_log.txt``, ``plan.txt``
under a caller-chosen ``results/…`` directory.
``trace_prepare`` also removes ``*.json`` and ``*.jsonl`` there (e.g. LLM smoke sidecars).

PNG frames omit a figure title by default (see ``task_id=""`` on ``trace_reset`` / ``trace_step``)
so ``tight_layout`` + ``bbox_inches="tight"`` framing stays balanced; pass a non-empty
``task_id`` only if you want a title.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

from nlu_benchmark.renderer import render_maze_image_png_bytes


def trace_prepare(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*.png"):
        p.unlink()
    for p in out_dir.glob("*.txt"):
        p.unlink()
    for p in out_dir.glob("*.json"):
        p.unlink()
    for p in out_dir.glob("*.jsonl"):
        p.unlink()


def trace_reset(out_dir: Path, state: Any, *, task_id: str = "") -> list[str]:
    (out_dir / "step_000_reset.png").write_bytes(render_maze_image_png_bytes(state, task_id=task_id))
    return [f"000 RESET pos={state.agent_pos} facing={state.facing} inv={state.inventory}"]


def trace_step(
    out_dir: Path,
    lines: list[str],
    step: int,
    action: str,
    env: Any,
    *,
    position_before: tuple[Any, ...],
    task_id: str = "",
) -> Tuple[Any, Any]:
    state, event = env.step(action)
    (out_dir / f"step_{step:03d}_{action}.png").write_bytes(
        render_maze_image_png_bytes(state, task_id=task_id)
    )
    line = (
        f"{step:03d} {action:<12} {event.type:<10} from={position_before} "
        f"to={state.agent_pos} facing={state.facing} inv={state.inventory}"
    )
    print(line)
    lines.append(line)
    return state, event


def trace_write_text_artifacts(out_dir: Path, lines: list[str], plan_actions: list[str]) -> None:
    (out_dir / "run_log.txt").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "plan.txt").write_text("\n".join(plan_actions), encoding="utf-8")
