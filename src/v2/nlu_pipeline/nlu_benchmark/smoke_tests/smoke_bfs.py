from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
V2_ROOT = Path(__file__).resolve().parents[3]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from nlu_benchmark.env import FACING_ORDER, FACING_TO_DELTA
from nlu_benchmark.loader import load_maze
from nlu_benchmark.smoke_trace import trace_prepare, trace_reset, trace_step, trace_write_text_artifacts
from automatic_maze_generation.mazegen.models import Door, Gate, Key, MazeInstance, Switch
from automatic_maze_generation.mazegen.solver import solve_maze


def _state_to_maze_instance(st) -> MazeInstance:
    def rc_to_xy(pos):
        r, c = pos
        # NLU grids are 1-based (row, col); mazegen solver uses 0-based (x, y).
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


def _xy_path_to_rc(path_xy) -> list[tuple[int, int]]:
    return [(y + 1, x + 1) for (x, y) in path_xy]


def _inject_pickups(actions: list[str], env, state) -> list[str]:
    """Nlu env needs explicit PICKUP; solver assumes pickup-on-entry."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test: mazegen solver plan replayed in NLU env (PNG trace under results/smoke_*_bfs/).")
    parser.add_argument("--maze", default="V04_single_key.json", help="Maze JSON filename under sample mazes/")
    parser.add_argument("--tag", default="", help="Optional output tag suffix.")
    args = parser.parse_args()

    maze_path = ROOT / "nlu_benchmark" / "sample mazes" / args.maze
    maze_stem = Path(args.maze).stem
    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = Path(__file__).resolve().parent / "results" / f"smoke_{maze_stem}_bfs{suffix}"
    trace_prepare(out_dir)
    env_plan = load_maze(maze_path)
    plan_state = env_plan.reset()
    maze_inst = _state_to_maze_instance(plan_state)
    solver_result = solve_maze(maze_inst)
    if not solver_result.get("is_solvable"):
        print("Solver reported unsolvable maze.")
        return
    path_rc = _xy_path_to_rc(solver_result.get("path", []))
    planned_actions = _path_to_actions(path_rc, start_facing="NORTH")
    executable_actions = _inject_pickups(planned_actions, env_plan, plan_state)

    env = load_maze(maze_path)
    state = env.reset()

    lines = trace_reset(out_dir, state)

    for step, action in enumerate(executable_actions, start=1):
        before = state.agent_pos
        state, event = trace_step(out_dir, lines, step, action, env, position_before=before)
        if event.type == "DONE":
            break

    trace_write_text_artifacts(out_dir, lines, executable_actions)
    print(f"\nsuccess={state.agent_pos == state.goal}")
    print(f"steps_used={state.step_count}")
    print(f"out={out_dir}")


if __name__ == "__main__":
    main()
