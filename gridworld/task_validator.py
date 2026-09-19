"""Task validator — beatability and related checks over the R1 rulebook.

Search uses ``world_model.successors`` / ``shortest_plan``. There is no second
physics model here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Optional

from .task_spec import TaskSpecification
from .world_model import (
    PlannerState,
    TaskPlanningContext,
    Transition,
    shortest_plan,
    successors,
)


class TaskValidator:
    """Validate that a task is beatable by BFS over the R1 action graph."""

    def __init__(self, spec: TaskSpecification, *, drop_available: bool = False):
        self.spec = spec
        self.ctx = TaskPlanningContext(spec, drop_available=drop_available)
        self.goal = self.ctx.goal
        self.start = self.ctx.start
        self.switches_by_id = {
            sw["id"]: sw for sw in self.ctx.switches_by_pos.values()
        }

    def validate(self, max_states: int = 500_000) -> tuple[bool, Optional[list[tuple[int, int]]], str]:
        beatable, path, states_explored = self._find_solution(
            self.ctx.initial_state(), max_states=max_states
        )
        if beatable:
            step_count = len(path) - 1 if path else 0
            return True, path, (
                f"Solution found in {step_count} steps ({states_explored} states explored)"
            )
        if states_explored >= max_states:
            return False, None, f"State space exceeded {max_states} states without finding solution"
        return False, None, (
            f"No solution found ({states_explored} states explored, all reachable states checked)"
        )

    def _find_solution(
        self,
        initial_state: PlannerState,
        goal: Optional[tuple[int, int]] = None,
        max_states: int = 500_000,
    ) -> tuple[bool, Optional[list[tuple[int, int]]], int]:
        target = self.goal if goal is None else goal
        actions, end, states_explored = shortest_plan(
            self.ctx,
            initial_state,
            lambda st: st.agent_pos == target,
        )
        if end is None:
            return False, None, min(states_explored, max_states)

        state = initial_state
        path = [state.agent_pos]
        for action in actions:
            nxt = next(
                (t.next_state for t in successors(self.ctx, state) if t.action == action),
                state,
            )
            state = nxt
            path.append(state.agent_pos)
        return True, path, states_explored

    def _successors(self, state: PlannerState) -> list[Transition]:
        return list(successors(self.ctx, state))

    def _spec_without_mechanism(self, mechanism_id: str) -> TaskSpecification:
        data = self.spec.to_dict()
        mechanisms = data.get("mechanisms", {})
        for key in ("keys", "doors", "switches", "gates", "blocks", "teleporters", "hazards"):
            mechanisms[key] = [
                item for item in mechanisms.get(key, [])
                if item.get("id") != mechanism_id
            ]
        if data.get("dependency_chain"):
            data["dependency_chain"]["sequence"] = [
                step for step in data["dependency_chain"].get("sequence", [])
                if step.get("element") != mechanism_id and step.get("unlocks") != mechanism_id
            ]
            data["dependency_chain"]["depth"] = len(data["dependency_chain"]["sequence"])
        return TaskSpecification.from_dict(data)

    def validate_mechanism_necessity(self) -> list[str]:
        if self.spec.dependency_chain is not None:
            mechanism_ids = [step.element for step in self.spec.dependency_chain.sequence]
        else:
            mechanism_ids = [
                obj.id
                for group in (
                    self.spec.mechanisms.keys,
                    self.spec.mechanisms.doors,
                    self.spec.mechanisms.switches,
                    self.spec.mechanisms.gates,
                    self.spec.mechanisms.blocks,
                    self.spec.mechanisms.teleporters,
                    self.spec.mechanisms.hazards,
                )
                for obj in group
            ]

        violations = []
        for mechanism_id in dict.fromkeys(mechanism_ids):
            stripped_spec = self._spec_without_mechanism(mechanism_id)
            beatable, _, _ = TaskValidator(stripped_spec).validate()
            if beatable:
                violations.append(f"Mechanism {mechanism_id} is not necessary")
        return violations

    def _spec_with_steps_triggered(self, steps: list) -> TaskSpecification:
        data = self.spec.to_dict()
        mechanisms = data.get("mechanisms", {})

        for step in steps:
            if step.type == "key-door":
                for door in mechanisms.get("doors", []):
                    if door.get("id") == step.unlocks:
                        door["initial_state"] = "open"
            elif step.type == "switch-gate":
                for switch in mechanisms.get("switches", []):
                    if switch.get("id") == step.element:
                        switch["initial_state"] = "on"
                for gate in mechanisms.get("gates", []):
                    if gate.get("id") == step.unlocks:
                        gate["initial_state"] = "open"
        return TaskSpecification.from_dict(data)

    def _get_element_position(self, element_id: str) -> Optional[tuple[int, int]]:
        for group in (
            self.spec.mechanisms.keys,
            self.spec.mechanisms.doors,
            self.spec.mechanisms.switches,
            self.spec.mechanisms.gates,
            self.spec.mechanisms.blocks,
            self.spec.mechanisms.hazards,
        ):
            for obj in group:
                if obj.id == element_id:
                    return obj.position.to_tuple()
        return None

    def validate_chain_ordering(self) -> bool:
        if self.spec.dependency_chain is None or len(self.spec.dependency_chain.sequence) <= 1:
            return True

        sequence = self.spec.dependency_chain.sequence
        for idx in range(len(sequence) - 1):
            current_step = sequence[idx]
            prior_steps = sequence[:idx]
            next_step = sequence[idx + 1]
            next_pos = self._get_element_position(next_step.element)
            if next_pos is None:
                return False
            staged_spec = self._spec_with_steps_triggered(prior_steps)
            staged_spec = TaskValidator(staged_spec)._spec_without_mechanism(current_step.element)
            staged_data = staged_spec.to_dict()
            staged_data["maze"]["goal"] = list(next_pos)
            staged_data["goal"] = {"type": "reach_position", "target": list(next_pos)}
            staged_target_spec = TaskSpecification.from_dict(staged_data)
            beatable, _, _ = TaskValidator(staged_target_spec).validate()
            if beatable:
                return False
        return True

    def validate_distractor_safety(self, base_beatable: bool | None = None) -> list[str]:
        if not self.spec.distractors:
            return []

        if base_beatable is None:
            base_beatable, _, _ = self.validate()
        if not base_beatable:
            return ["Base task is not solvable"]

        initial_state = self.ctx.initial_state()
        violations = []
        for distractor in self.spec.distractors:
            relevant_ids = self._distractor_candidate_ids(distractor)
            queue = deque([initial_state])
            visited = {initial_state}
            found_interaction = False
            unsafe = False

            while queue:
                state = queue.popleft()
                for transition in successors(self.ctx, state):
                    if transition.next_state not in visited:
                        visited.add(transition.next_state)
                        queue.append(transition.next_state)

                    if not any(
                        self._transition_matches_distractor(transition.label, candidate_id)
                        for candidate_id in relevant_ids
                    ):
                        continue

                    found_interaction = True
                    beatable, _, _ = self._find_solution(transition.next_state)
                    if (
                        not beatable
                        and distractor.type == "wrong_color_key"
                        and transition.label.startswith("pickup:")
                    ):
                        dropped_state = replace(transition.next_state, carrying_key=None)
                        beatable, _, _ = self._find_solution(dropped_state)
                    if not beatable:
                        unsafe = True
                        queue.clear()
                        break

                if unsafe:
                    break

            if unsafe or not found_interaction:
                violations.append(f"Distractor {distractor.element_id} can break solvability")

        return violations

    def compute_fragility(self, depth_limit: int = 5) -> "FragilityReport":
        initial_state = self.ctx.initial_state()
        queue = deque([(initial_state, [])])
        visited: dict[PlannerState, int] = {initial_state: 0}
        breaking_sequences: list[list[str]] = []
        min_steps_to_break = None

        while queue:
            state, sequence = queue.popleft()
            if min_steps_to_break is not None and len(sequence) >= min_steps_to_break:
                continue
            if len(sequence) >= depth_limit:
                continue

            for transition in successors(self.ctx, state):
                next_sequence = list(sequence)
                if self._is_irreversible_transition(state, transition):
                    next_sequence = sequence + [transition.label]
                next_irrev = len(next_sequence)
                if next_irrev > depth_limit:
                    continue
                if transition.next_state in visited and visited[transition.next_state] <= next_irrev:
                    continue
                visited[transition.next_state] = next_irrev

                beatable, _, _ = self._find_solution(transition.next_state)
                if not beatable and self._is_irreversible_transition(state, transition):
                    min_steps_to_break = (
                        len(next_sequence)
                        if min_steps_to_break is None
                        else min(min_steps_to_break, len(next_sequence))
                    )
                    if len(next_sequence) == min_steps_to_break:
                        breaking_sequences.append(next_sequence)
                    continue

                queue.append((transition.next_state, next_sequence))

        if min_steps_to_break is None:
            return FragilityReport(
                min_steps_to_break=-1,
                breaking_sequences=[],
                is_fragile=False,
            )

        return FragilityReport(
            min_steps_to_break=min_steps_to_break,
            breaking_sequences=breaking_sequences[:depth_limit],
            is_fragile=min_steps_to_break <= 3,
        )

    def _transition_matches_distractor(self, action_label: str, element_id: str) -> bool:
        if action_label.startswith(("pickup:", "toggle:", "open_door:", "drop:")):
            return action_label.split(":", 1)[1] == element_id
        return False

    def _distractor_candidate_ids(self, distractor) -> list[str]:
        if any(
            distractor.element_id == obj.id
            for group in (
                self.spec.mechanisms.keys,
                self.spec.mechanisms.doors,
                self.spec.mechanisms.switches,
                self.spec.mechanisms.gates,
                self.spec.mechanisms.blocks,
                self.spec.mechanisms.teleporters,
                self.spec.mechanisms.hazards,
            )
            for obj in group
        ):
            return [distractor.element_id]

        if distractor.type == "distractor_chain":
            critical_ids = set()
            if self.spec.dependency_chain is not None:
                for step in self.spec.dependency_chain.sequence:
                    critical_ids.add(step.element)
                    critical_ids.add(step.unlocks)
            candidate_ids = [
                obj.id
                for group in (
                    self.spec.mechanisms.keys,
                    self.spec.mechanisms.doors,
                    self.spec.mechanisms.switches,
                    self.spec.mechanisms.gates,
                )
                for obj in group
                if obj.id not in critical_ids
            ]
            return candidate_ids or [distractor.element_id]

        return [distractor.element_id]

    def _is_irreversible_transition(self, state: PlannerState, transition: Transition) -> bool:
        label = transition.label
        if label.startswith("open_door:") and self.spec.rules.key_consumption:
            return True
        if label.startswith("toggle:"):
            switch_id = label.split(":", 1)[1]
            switch_info = self.switches_by_id.get(switch_id, {})
            return switch_info.get("switch_type") == "one_shot"
        if label.startswith("drop:"):
            return True
        return False


@dataclass
class FragilityReport:
    """Minimum wrong-step analysis for a task."""
    min_steps_to_break: int
    breaking_sequences: list[list[str]]
    is_fragile: bool

    def to_dict(self) -> dict:
        return {
            "min_steps_to_break": self.min_steps_to_break,
            "breaking_sequences": self.breaking_sequences,
            "is_fragile": self.is_fragile,
        }


@dataclass
class DifficultyReport:
    """Difficulty metrics for a task."""
    task_id: str
    tier: int
    is_beatable: bool
    optimal_steps: int
    states_explored: int
    mechanism_count: int
    mechanism_types: int
    dependency_depth: int
    grid_area: int
    optimal_path: list[tuple[int, int]]
    backtrack_count: int
    difficulty_score: float

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "tier": self.tier,
            "is_beatable": self.is_beatable,
            "optimal_steps": self.optimal_steps,
            "states_explored": self.states_explored,
            "mechanism_count": self.mechanism_count,
            "mechanism_types": self.mechanism_types,
            "dependency_depth": self.dependency_depth,
            "grid_area": self.grid_area,
            "optimal_path": [list(pos) for pos in self.optimal_path],
            "backtrack_count": self.backtrack_count,
            "difficulty_score": round(self.difficulty_score, 2),
        }


def compute_difficulty(
    spec: TaskSpecification,
    validator: TaskValidator | None = None,
    validation_result: tuple[bool, Optional[list[tuple[int, int]]], str] | None = None,
    bfs_path=None,
) -> DifficultyReport:
    """Solver-derived difficulty metrics. Beatability and step count share one graph."""
    task_validator = validator or TaskValidator(spec)
    if validation_result is None:
        validation_result = task_validator.validate()
    is_beatable, solution, message = validation_result

    if bfs_path is None:
        from gridworld.baselines import plan_bfs_path

        bfs_path = plan_bfs_path(spec)
    if bfs_path is not None and bfs_path.success:
        optimal_steps = len(bfs_path.action_labels)
        solution = [tuple(pos) for pos in bfs_path.positions]
        states_explored = bfs_path.states_explored
    else:
        optimal_steps = len(solution) - 1 if solution else 0
        import re
        match = re.search(r"(\d+) states explored", message)
        states_explored = int(match.group(1)) if match else 0

    seen = set()
    backtrack_count = 0
    previous_pos = None
    for pos in solution or []:
        if pos == previous_pos:
            continue
        if pos in seen:
            backtrack_count += 1
        seen.add(pos)
        previous_pos = pos

    m = spec.mechanisms
    keys_count = len(m.keys)
    doors_count = len(m.doors)
    switches_count = len(m.switches)
    gates_count = len(m.gates)
    blocks_count = len(m.blocks)
    teleporters_count = len(m.teleporters)
    hazards_count = len(m.hazards)
    kill_cells_count = len(m.kill_cells)
    frozen_tiles_count = len(m.frozen_tiles)
    mechanism_count = (
        keys_count + doors_count + switches_count
        + gates_count + blocks_count + teleporters_count + hazards_count
        + kill_cells_count + frozen_tiles_count
    )

    type_flags = [
        keys_count > 0,
        doors_count > 0,
        switches_count > 0,
        gates_count > 0,
        blocks_count > 0,
        teleporters_count > 0,
        hazards_count > 0,
        kill_cells_count > 0,
        frozen_tiles_count > 0,
    ]
    mechanism_types = sum(type_flags)

    depth = spec.dependency_chain.depth if spec.dependency_chain is not None else 0
    if depth == 0:
        if doors_count > 0 and keys_count > 0:
            depth = max(depth, 1)
        if gates_count > 0 and switches_count > 0:
            depth = max(depth, 1)
        if doors_count > 0 and keys_count > 0 and gates_count > 0 and switches_count > 0:
            depth = max(depth, 2)
        if blocks_count > 0:
            depth = max(depth, 1)
        if teleporters_count > 0:
            depth = max(depth, 1)
        if (teleporters_count > 0 or blocks_count > 0) and (gates_count > 0 or doors_count > 0):
            depth = max(depth, 2)

    w, h = spec.maze.dimensions
    grid_area = w * h

    score = (
        optimal_steps * 1.0
        + mechanism_count * 2.0
        + mechanism_types * 3.0
        + depth * 5.0
        + backtrack_count * 2.0
        + (states_explored / 100.0)
        + (grid_area / 50.0)
    )

    return DifficultyReport(
        task_id=spec.task_id,
        tier=spec.difficulty_tier,
        is_beatable=is_beatable,
        optimal_steps=optimal_steps,
        states_explored=states_explored,
        mechanism_count=mechanism_count,
        mechanism_types=mechanism_types,
        dependency_depth=depth,
        grid_area=grid_area,
        optimal_path=solution or [],
        backtrack_count=backtrack_count,
        difficulty_score=score,
    )


def validate_task_file(path: str, verbose: bool = True) -> bool:
    spec = TaskSpecification.from_json(path)
    report = compute_difficulty(spec)

    if verbose:
        status = "PASS" if report.is_beatable else "FAIL"
        print(
            f"[{status}] {spec.task_id}: optimal={report.optimal_steps} steps, "
            f"mechanisms={report.mechanism_count} ({report.mechanism_types} types), "
            f"depth={report.dependency_depth}, score={report.difficulty_score}"
        )

    return report.is_beatable


def validate_all_tasks(tasks_dir: str = "gridworld/tasks", verbose: bool = True) -> dict:
    from pathlib import Path

    results = {"pass": [], "fail": [], "reports": []}
    tasks_path = Path(tasks_dir)

    for tier in range(1, 6):
        tier_dir = tasks_path / f"tier{tier}"
        if not tier_dir.exists():
            continue

        if verbose:
            print(f"\n=== Tier {tier} ===")

        for task_file in sorted(tier_dir.glob("*.json")):
            spec = TaskSpecification.from_json(str(task_file))
            report = compute_difficulty(spec)
            results["reports"].append(report.to_dict())

            if verbose:
                status = "PASS" if report.is_beatable else "FAIL"
                print(
                    f"  [{status}] {report.task_id}: optimal={report.optimal_steps} steps, "
                    f"mechanisms={report.mechanism_count}, score={report.difficulty_score}"
                )

            if report.is_beatable:
                results["pass"].append(str(task_file))
            else:
                results["fail"].append(str(task_file))

    if verbose:
        total = len(results["pass"]) + len(results["fail"])
        print(f"\n=== Summary: {len(results['pass'])}/{total} tasks beatable ===")
        if results["fail"]:
            print("Failed tasks:")
            for f in results["fail"]:
                print(f"  - {f}")

        print("\n=== Difficulty Ranking ===")
        sorted_reports = sorted(results["reports"], key=lambda r: r["difficulty_score"])
        for r in sorted_reports:
            print(f"  {r['difficulty_score']:6.1f}  T{r['tier']}  {r['task_id']}")

    return results


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    validate_all_tasks()
