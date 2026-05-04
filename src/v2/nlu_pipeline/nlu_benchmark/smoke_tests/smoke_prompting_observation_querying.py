from __future__ import annotations

import json
import argparse
import base64
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
V2_ROOT = Path(__file__).resolve().parents[3]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from nlu_benchmark.config import ExperimentConfig
from nlu_benchmark.env import FACING_ORDER, FACING_TO_DELTA
from nlu_benchmark.loader import load_maze
from nlu_benchmark.runner import ExperimentRunner
import nlu_benchmark.observation as observation_module
from automatic_maze_generation.mazegen.models import Door, Gate, Key, MazeInstance, Switch
from automatic_maze_generation.mazegen.solver import solve_maze


_POS_RE = re.compile(r"Position:\s*\((\d+),\s*(\d+)\)")
_FACING_RE = re.compile(r"Facing:\s*([A-Z]+)")
_GOAL_RE = re.compile(r"Goal:\s*\((\d+),\s*(\d+)\)")


_ONE_BY_ONE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5nVxUAAAAASUVORK5CYII="
)


def _extract_user_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [blk.get("text", "") for blk in content if isinstance(blk, dict) and blk.get("type") == "text"]
        return "\n".join(texts)
    return ""


def _parse_prompt_state(user_text: str):
    pm = _POS_RE.search(user_text)
    fm = _FACING_RE.search(user_text)
    gm = _GOAL_RE.search(user_text)
    if not (pm and fm and gm):
        return None
    pos = (int(pm.group(1)), int(pm.group(2)))
    facing = fm.group(1)
    goal = (int(gm.group(1)), int(gm.group(2)))
    return pos, facing, goal


def _turn_to_face(cur: str, target: str) -> list[str]:
    ci = FACING_ORDER.index(cur)
    ti = FACING_ORDER.index(target)
    diff = (ti - ci) % 4
    if diff == 0:
        return []
    if diff == 1:
        return ["TURN_RIGHT"]
    if diff == 2:
        return ["TURN_RIGHT", "TURN_RIGHT"]
    return ["TURN_LEFT"]


def _plan_to_goal_from_prompt(user_text: str, budget: int = 6) -> list[str]:
    parsed = _parse_prompt_state(user_text)
    if parsed is None:
        return ["TURN_RIGHT"]
    (r, c), facing, (gr, gc) = parsed
    actions: list[str] = []
    if c != gc:
        target = "EAST" if gc > c else "WEST"
        actions.extend(_turn_to_face(facing, target))
        actions.extend(["MOVE_FORWARD"] * min(abs(gc - c), max(1, budget - len(actions))))
    elif r != gr:
        target = "SOUTH" if gr > r else "NORTH"
        actions.extend(_turn_to_face(facing, target))
        actions.extend(["MOVE_FORWARD"] * min(abs(gr - r), max(1, budget - len(actions))))
    else:
        actions.append("DONE")
    return actions[:budget] if actions else ["DONE"]


def _state_to_maze_instance(st) -> MazeInstance:
    def rc_to_xy(pos):
        r, c = pos
        return (c - 1, r - 1)

    return MazeInstance(
        width=st.cols,
        height=st.rows,
        walls={rc_to_xy(w) for w in st.walls},
        start=rc_to_xy(st.start),
        goal=rc_to_xy(st.goal),
        keys=[
            Key(id=k.get("id", f"key_{i}"), position=rc_to_xy(tuple(k["position"])), color=k["color"])
            for i, k in enumerate(st.keys)
        ],
        doors=[
            Door(
                id=d.get("id", f"door_{i}"),
                position=rc_to_xy(tuple(d["position"])),
                requires_key=d["requires_key"],
                initial_state=d.get("initial_state", "locked"),
            )
            for i, d in enumerate(st.doors)
        ],
        switches=[
            Switch(
                id=s.get("id", f"switch_{i}"),
                position=rc_to_xy(tuple(s["position"])),
                controls=list(s.get("controls", [])),
                switch_type=s.get("switch_type", "toggle"),
                initial_state=s.get("initial_state", "off"),
            )
            for i, s in enumerate(st.switches)
        ],
        gates=[
            Gate(
                id=g.get("id", f"gate_{i}"),
                position=rc_to_xy(tuple(g["position"])),
                initial_state=g.get("initial_state", "closed"),
            )
            for i, g in enumerate(st.gates)
        ],
    )


def _xy_path_to_rc(path_xy) -> list[tuple[int, int]]:
    return [(y + 1, x + 1) for (x, y) in path_xy]


def _path_to_actions(path, start_facing: str = "NORTH") -> list[str]:
    if not path or len(path) < 2:
        return ["DONE"]
    facing = start_facing
    actions: list[str] = []
    for (r, c), (nr, nc) in zip(path, path[1:]):
        dr, dc = nr - r, nc - c
        target = next((f for f, d in FACING_TO_DELTA.items() if d == (dr, dc)), None)
        if target is None:
            continue
        cur_idx = FACING_ORDER.index(facing)
        tgt_idx = FACING_ORDER.index(target)
        diff = (tgt_idx - cur_idx) % 4
        if diff == 1:
            actions.append("TURN_RIGHT")
        elif diff == 2:
            actions.extend(["TURN_RIGHT", "TURN_RIGHT"])
        elif diff == 3:
            actions.append("TURN_LEFT")
        actions.append("MOVE_FORWARD")
        facing = target
    actions.append("DONE")
    return actions


def _inject_pickups(actions: list[str], env, state) -> list[str]:
    out: list[str] = []
    sim_state = state
    for a in actions:
        has_key_here = any(tuple(k["position"]) == sim_state.agent_pos for k in sim_state.keys)
        if has_key_here and a != "PICKUP":
            out.append("PICKUP")
            sim_state, _ = env.step("PICKUP")
        out.append(a)
        sim_state, _ = env.step(a)
    return out


def _full_trajectory_actions_for_maze(maze_path: Path) -> list[str]:
    env = load_maze(maze_path)
    state = env.reset()
    maze_inst = _state_to_maze_instance(state)
    solver_result = solve_maze(maze_inst)
    if not solver_result.get("is_solvable"):
        return ["DONE"]
    path_rc = _xy_path_to_rc(solver_result.get("path", []))
    planned = _path_to_actions(path_rc, start_facing="NORTH")
    return _inject_pickups(planned, env, state)


class ProbeAgent:
    """Deterministic test agent that records message structure on each query."""

    def __init__(self, full_trajectory_actions: list[str]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._full_trajectory_actions = full_trajectory_actions

    def __call__(self, messages: list[dict]) -> str:
        system_text = str(messages[0]["content"])
        user_msg = messages[-1]
        user_content = user_msg.get("content")
        user_text = _extract_user_text(user_content)
        has_image = isinstance(user_content, list) and any(
            isinstance(blk, dict) and blk.get("type") == "image_url" for blk in user_content
        )

        full_mode = "You will not be queried again" in system_text
        subgoal_mode = "SUB_GOAL:" in system_text and "ACTIONS:" in system_text

        reply: str
        if full_mode:
            reply = (
                "SUB_GOAL: Execute maze-aware end-to-end plan.\n"
                f"ACTIONS: {', '.join(self._full_trajectory_actions)}"
            )
        elif subgoal_mode:
            chunk = _plan_to_goal_from_prompt(user_text, budget=4)
            reply = f"SUB_GOAL: Advance toward goal.\nACTIONS: {', '.join(chunk)}"
        else:
            step = _plan_to_goal_from_prompt(user_text, budget=1)[0]
            reply = f"FINAL_OUTPUT: {step}"

        self.calls.append(
            {
                "system": system_text,
                "user_content_type": type(user_content).__name__,
                "has_image": has_image,
                "user_text": user_text,
                "assistant_reply": reply,
            }
        )
        return reply


def _assert(cond: bool, msg: str, errors: list[str]) -> None:
    if not cond:
        errors.append(msg)


def _run_case(base: ExperimentConfig, maze_path: Path, label: str, full_trajectory_actions: list[str], max_steps: int):
    runner = ExperimentRunner.from_json(str(maze_path), config=base)
    runner.env.initial.max_steps = min(runner.env.initial.max_steps, max_steps)
    agent = ProbeAgent(full_trajectory_actions)
    result = runner.run(agent, verbose=False)
    return label, base, result, agent


def _suite_cases(base: ExperimentConfig, suite: str):
    all_cases = [
        (replace(base, prompting="minimal"), "prompting=minimal"),
        (replace(base, prompting="standard"), "prompting=standard"),
        (replace(base, prompting="verbose"), "prompting=verbose"),
        (replace(base, context_window="current"), "context=current"),
        (replace(base, context_window="last3"), "context=last3"),
        (replace(base, observation="text_only", context_window="last3"), "obs=text_only"),
        (replace(base, observation="image_text", context_window="last3"), "obs=image_text"),
        (replace(base, observation="image_only", context_window="last3"), "obs=image_only"),
        (replace(base, querying="step_by_step"), "query=step_by_step"),
        (replace(base, querying="subgoal"), "query=subgoal"),
        (replace(base, querying="full_trajectory"), "query=full_trajectory"),
    ]
    if suite == "all":
        return all_cases
    if suite == "prompting":
        return [c for c in all_cases if c[1].startswith("prompting=")]
    if suite == "observation":
        return [c for c in all_cases if c[1].startswith("obs=") or c[1].startswith("context=")]
    if suite == "querying":
        return [c for c in all_cases if c[1].startswith("query=")]
    raise ValueError(f"Unknown suite: {suite}")


def run_smoke_suite(maze_name: str, tag: str, max_steps: int, suite: str = "all") -> tuple[Path, Path]:
    maze_path = ROOT / "nlu_benchmark" / "sample mazes" / maze_name
    maze_stem = Path(maze_name).stem
    suffix = f"_{tag}" if tag else ""
    full_trajectory_actions = _full_trajectory_actions_for_maze(maze_path)
    # Smoke test already validated rendering elsewhere; use tiny static bytes for speed.
    observation_module.render_maze_image_png_bytes = lambda _state: _ONE_BY_ONE_PNG
    base = ExperimentConfig(prompting="minimal", observation="text_only", context_window="last3", querying="step_by_step")
    selected = _suite_cases(base, suite)
    outputs = [
        _run_case(cfg, maze_path, label, full_trajectory_actions, max_steps)
        for cfg, label in selected
    ]
    errors: list[str] = []
    summary_lines: list[str] = []
    detailed_runs: list[dict[str, Any]] = []

    for label, cfg, result, agent in outputs:
        calls = len(agent.calls)
        first = agent.calls[0]
        summary_lines.append(
            f"{label:<24} success={result['success']!s:<5} steps={result['steps_used']:<3} queries={calls:<3}"
        )

        if cfg.prompting == "minimal":
            _assert("The environment may contain:" not in first["system"], f"{label}: minimal has mechanism list", errors)
        if cfg.prompting == "standard":
            _assert("The environment may contain:" in first["system"], f"{label}: standard missing mechanism list", errors)
            _assert("RULES (domain logic):" not in first["system"], f"{label}: standard unexpectedly has verbose rules", errors)
        if cfg.prompting == "verbose":
            _assert("RULES (domain logic):" in first["system"], f"{label}: verbose missing rules", errors)

        if cfg.observation == "text_only":
            _assert(first["user_content_type"] == "str", f"{label}: text_only should send string content", errors)
            _assert(not first["has_image"], f"{label}: text_only should not include image", errors)
        else:
            _assert(first["user_content_type"] == "list", f"{label}: image mode should send list content", errors)
            _assert(first["has_image"], f"{label}: image mode should include image block", errors)

        if cfg.observation == "image_only":
            _assert("Initial maze (fixed for this episode):" not in first["system"], f"{label}: image_only should omit initial NL map", errors)
        else:
            _assert("Initial maze (fixed for this episode):" in first["system"], f"{label}: expected initial NL map in system prompt", errors)

        if cfg.context_window == "current" and len(agent.calls) > 1:
            second_text = agent.calls[1]["user_text"]
            _assert("Recent history (last 3 steps" not in second_text, f"{label}: current unexpectedly includes history", errors)
            _assert("Recent steps (oldest first, action only):" not in second_text, f"{label}: current unexpectedly includes action history", errors)
        if cfg.context_window == "last3" and len(agent.calls) > 1:
            second_text = agent.calls[1]["user_text"]
            if cfg.observation == "image_only":
                _assert("Recent steps (oldest first, action only):" in second_text, f"{label}: last3 image_only should include action-only history", errors)
            else:
                _assert("Recent history (last 3 steps, oldest first):" in second_text, f"{label}: last3 should include full history", errors)

        if cfg.querying == "full_trajectory":
            _assert(calls == 1, f"{label}: full_trajectory should query once, got {calls}", errors)
        if cfg.querying == "step_by_step":
            _assert(calls >= 3, f"{label}: step_by_step should query repeatedly, got {calls}", errors)
        if cfg.querying == "subgoal":
            _assert(calls >= 2, f"{label}: subgoal should query at least twice, got {calls}", errors)
            has_subgoal_meta = any("subgoal" in t for t in result["transcript"])
            _assert(has_subgoal_meta, f"{label}: transcript missing subgoal metadata", errors)

        detailed_runs.append(
            {
                "label": label,
                "config": cfg.to_dict(),
                "summary": {
                    "success": result["success"],
                    "steps_used": result["steps_used"],
                    "query_count": calls,
                },
                "system_prompt": first["system"],
                "queries": [
                    {
                        "call_idx": i + 1,
                        "user_content_type": call["user_content_type"],
                        "has_image": call["has_image"],
                        "user_text": call["user_text"],
                        "assistant_reply": call["assistant_reply"],
                    }
                    for i, call in enumerate(agent.calls)
                ],
                "transcript": result["transcript"],
            }
        )

    out_dir = Path(__file__).resolve().parent / "results" / f"smoke_{suite}_{maze_stem}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "report.txt"
    details_json = out_dir / "detailed_logs.json"
    body = ["Runner/Prompt/Observation/Querying smoke report", ""] + summary_lines + [""]
    if errors:
        body.append("FAILURES:")
        body.extend(f"- {e}" for e in errors)
    else:
        body.append("All checks passed.")
    report.write_text("\n".join(body), encoding="utf-8")
    details_json.write_text(json.dumps({"maze": str(maze_path), "runs": detailed_runs}, indent=2), encoding="utf-8")

    print("\n".join(summary_lines))
    print("")
    if errors:
        print(f"FAILED checks: {len(errors)}")
        for e in errors:
            print(f"- {e}")
    else:
        print("All checks passed.")
    print(f"report={report}")
    print(f"details={details_json}")
    return report, details_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test prompting/context/querying/observation workflow.")
    parser.add_argument("--maze", default="V01_empty_room.json", help="Maze JSON filename under sample mazes/")
    parser.add_argument("--tag", default="", help="Optional output tag suffix.")
    parser.add_argument("--max-steps", type=int, default=40, help="Cap per-episode steps for faster smoke runs.")
    parser.add_argument("--suite", choices=["all", "prompting", "observation", "querying"], default="all")
    args = parser.parse_args()
    run_smoke_suite(args.maze, args.tag, args.max_steps, args.suite)


if __name__ == "__main__":
    main()
