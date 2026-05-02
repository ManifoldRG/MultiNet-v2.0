"""Run the built-in gridworld baselines on a small mechanism task.

The default task includes a key, a locked door, a switch, and a gate. A
different task JSON can be supplied with ``--task`` when benchmark files are
available in the checkout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation_harness import EvaluationHarness
from gridworld.actions import ACTION_NAMES
from gridworld.baselines import BFSModelInterface, GreedyModelInterface
from gridworld.task_spec import TaskSpecification
from gridworld.task_validator import compute_difficulty
from model_interface import RandomModelInterface


def built_in_demo_task() -> TaskSpecification:
    """Return a compact task with the same dependency objects used in demos."""
    return TaskSpecification.from_dict({
        "task_id": "demo_key_door_switch_gate",
        "seed": 7,
        "difficulty_tier": 3,
        "description": "Pick up a key, open a door, toggle a switch, then cross a gate.",
        "maze": {
            "dimensions": [8, 5],
            "walls": [],
            "start": [1, 1],
            "goal": [6, 1],
        },
        "mechanisms": {
            "keys": [{"id": "red_key", "position": [2, 1], "color": "red"}],
            "doors": [{"id": "red_door", "position": [3, 1], "requires_key": "red"}],
            "switches": [
                {
                    "id": "gate_switch",
                    "position": [4, 2],
                    "controls": ["exit_gate"],
                    "switch_type": "toggle",
                    "initial_state": "off",
                }
            ],
            "gates": [{"id": "exit_gate", "position": [5, 1], "initial_state": "closed"}],
            "blocks": [],
            "teleporters": [],
            "hazards": [],
        },
        "rules": {
            "key_consumption": True,
            "switch_type": "toggle",
            "hidden_mechanisms": [],
            "observability": "full",
            "view_size": 7,
        },
        "goal": {"type": "reach_position", "target": [6, 1], "auxiliary_conditions": []},
        "max_steps": 30,
    })


def load_task(path: str | None) -> TaskSpecification:
    """Load a task JSON, or fall back to the built-in mechanism task."""
    if path is None:
        return built_in_demo_task()
    return TaskSpecification.from_json(path)


def make_agent(name: str):
    """Construct one baseline agent by CLI name."""
    if name == "bfs":
        return BFSModelInterface()
    if name == "greedy":
        return GreedyModelInterface()
    if name == "random":
        return RandomModelInterface(seed=42)
    raise ValueError(f"Unknown agent: {name}")


def run_agent(agent_name: str, spec: TaskSpecification, verbose: bool) -> None:
    """Run one baseline and print a compact result summary."""
    model = make_agent(agent_name)
    harness = EvaluationHarness(model, history_images=0, history_text=False)
    try:
        result = harness.evaluate_task(spec, seed=spec.seed, verbose=verbose)
    finally:
        harness.close()

    difficulty = compute_difficulty(spec)
    action_names = [ACTION_NAMES.get(step.action, str(step.action)) for step in result.trajectory]
    preview = ", ".join(action_names[:12])
    if len(action_names) > 12:
        preview += ", ..."

    print(f"\nAgent: {agent_name}")
    print(f"Task: {spec.task_id}")
    print(f"Success: {result.success}")
    print(f"Steps: {result.steps_taken}/{spec.max_steps}")
    print(f"Validator optimal steps: {difficulty.optimal_steps}")
    print(f"Actions: {preview or 'none'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BFS, greedy, and random baselines on a demo task.")
    parser.add_argument("--agent", choices=["all", "bfs", "greedy", "random"], default="all")
    parser.add_argument("--task", default=None, help="Optional task JSON path. Defaults to built-in demo task.")
    parser.add_argument("--verbose", action="store_true", help="Print each environment step.")
    args = parser.parse_args()

    spec = load_task(args.task)
    agents = ["bfs", "greedy", "random"] if args.agent == "all" else [args.agent]
    for agent_name in agents:
        run_agent(agent_name, spec, args.verbose)


if __name__ == "__main__":
    main()
