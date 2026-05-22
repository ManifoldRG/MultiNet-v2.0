"""Prompt strategies for the NLU interface."""

from __future__ import annotations

from gridworld.backends.base import GridState
from gridworld.task_spec import TaskSpecification

from interface.coords import (
    FACING_ORDER,
    FACING_TO_DELTA,
    agent_facing,
    agent_row_col,
    describe_cell,
    goal_row_col,
    inventory_list,
    maze_rows_cols,
    wall_cells,
)

MECHANISM_LIST = (
    "The environment may contain:\n"
    "- Keys: pick them up to open doors of the matching color\n"
    "- Doors: blocked passages that require a matching key\n"
    "- Switches: step onto them to activate (hold) or TOGGLE while standing on them\n"
    "- Gates: blocked passages controlled by switches\n"
)

MECHANISM_RULES = (
    "RULES (domain logic):\n"
    "  - PICKUP: pick up a key from the adjacent cell you are facing. Keys block movement — you\n"
    "    cannot MOVE_FORWARD onto a key; stand beside it, face it, and PICKUP.\n"
    "  - Doors: face a locked door with the matching key in inventory and TOGGLE to open it, then\n"
    "    MOVE_FORWARD through the open door. MOVE_FORWARD alone does not open a locked door.\n"
    "  - Switches: MOVE_FORWARD onto the switch cell, then TOGGLE (toggle/one-shot types). Hold-type\n"
    "    switches activate automatically while you stand on them. Only switches are toggled. Linked\n"
    "    gates are open if at least one linked switch is on, and closed if all are off.\n"
    "  - Gates: you cannot TOGGLE a gate. CLOSED gates block movement; OPEN gates do not.\n"
    "  - Closed gates and doors you lack a key for block movement like walls until resolved.\n"
    "  - Use DONE only when you are standing on the goal cell."
)

FINAL_OUTPUT_INSTRUCTION = (
    "On the last line, output exactly:\n"
    "FINAL_OUTPUT: <action>  or  FINAL_OUTPUT: <a>, <b>, ...  "
    "(comma-separated; one or more valid actions)"
)


class MinimalPromptStrategy:
    def __init__(self, actions_hint: str) -> None:
        self._actions_hint = actions_hint

    def build_system_prompt(self, querying_suffix: str = "") -> str:
        return (
            "Task: move to the goal cell in the grid.\n"
            f"Valid actions: {self._actions_hint}.\n"
            f"{FINAL_OUTPUT_INSTRUCTION}"
            + (f"\n\n{querying_suffix}" if querying_suffix else "")
        )

    def build_user_prompt(
        self,
        obs_text: str,
        history_text: str,
        task_spec: TaskSpecification,
        state: GridState,
        last_feedback: str,
    ) -> str:
        history_block = f"{history_text}\n\n" if history_text else ""
        obs_block = f"Observation:\n{obs_text}\n\n" if obs_text else ""
        pos = agent_row_col(state)
        goal = goal_row_col(task_spec)
        return (
            f"{history_block}"
            f"{obs_block}"
            f"Position: {pos}  |  Facing: {agent_facing(state)}  |  Goal: {goal}  |  "
            f"Step {state.step_count + 1}/{state.max_steps}\n"
            f"Last result: {last_feedback}\n"
            "What is your next action?"
        )


class StandardPromptStrategy(MinimalPromptStrategy):
    def build_system_prompt(self, querying_suffix: str = "") -> str:
        return (
            "Task: move to the goal cell in the grid.\n"
            f"{MECHANISM_LIST}\n"
            f"Valid actions: {self._actions_hint}.\n"
            f"{FINAL_OUTPUT_INSTRUCTION}"
            + (f"\n\n{querying_suffix}" if querying_suffix else "")
        )


class VerbosePromptStrategy(StandardPromptStrategy):
    def build_system_prompt(self, querying_suffix: str = "") -> str:
        std = StandardPromptStrategy.build_system_prompt(self, "").rstrip()
        chunks = [std, MECHANISM_RULES]
        if querying_suffix:
            chunks.append(querying_suffix)
        return "\n\n".join(chunks)

    def build_user_prompt(
        self,
        obs_text: str,
        history_text: str,
        task_spec: TaskSpecification,
        state: GridState,
        last_feedback: str,
    ) -> str:
        steps_left = state.max_steps - state.step_count
        budget_warn = (
            f"  WARNING: Only {steps_left} steps remaining!\n"
            if steps_left <= max(5, state.max_steps // 5)
            else ""
        )
        row, col = agent_row_col(state)
        grow, gcol = goal_row_col(task_spec)
        manhattan = abs(row - grow) + abs(col - gcol)
        rows, cols = maze_rows_cols(task_spec)
        walls = wall_cells(task_spec)

        facing_idx = FACING_ORDER.index(agent_facing(state))
        rel_dirs = [
            ("AHEAD", FACING_ORDER[facing_idx % 4]),
            ("RIGHT", FACING_ORDER[(facing_idx + 1) % 4]),
            ("BEHIND", FACING_ORDER[(facing_idx + 2) % 4]),
            ("LEFT", FACING_ORDER[(facing_idx + 3) % 4]),
        ]
        neighbour_lines = []
        for rel, cardinal in rel_dirs:
            dr, dc = FACING_TO_DELTA[cardinal]
            nr, nc = row + dr, col + dc
            desc = describe_cell(
                task_spec,
                state,
                nr,
                nc,
                walls=walls,
                goal=(grow, gcol),
                rows=rows,
                cols=cols,
            )
            neighbour_lines.append(f"  {rel}: {desc}")
        neighbour_block = "From your perspective:\n" + "\n".join(neighbour_lines) + "\n"
        mechanism_block = _mechanism_hints_text(task_spec)
        history_block = f"{history_text}\n\n" if history_text else ""
        obs_block = f"Observation:\n{obs_text}\n\n" if obs_text else ""
        inventory_str = ", ".join(inventory_list(state)) or "none"

        return (
            f"{history_block}"
            f"{obs_block}"
            f"Position: {row, col}  |  Facing: {agent_facing(state)}  |  Goal: {(grow, gcol)}  |  "
            f"Manhattan: {manhattan}  |  Step {state.step_count + 1}/{state.max_steps} ({steps_left} left)\n"
            f"Inventory: {inventory_str}\n"
            f"{budget_warn}"
            f"{neighbour_block}"
            f"{mechanism_block}"
            f"Last result: {last_feedback}\n"
            "What is your next action?"
        )


PromptStrategy = MinimalPromptStrategy


def _mechanism_hints_text(task_spec: TaskSpecification) -> str:
    lines = []
    if task_spec.mechanisms.keys or task_spec.mechanisms.doors:
        lines.append(
            "  - Face an adjacent key and PICKUP (do not walk onto the key). "
            "Face a locked door with the matching key and TOGGLE to open it, then MOVE_FORWARD through."
        )
    if task_spec.mechanisms.switches or task_spec.mechanisms.gates:
        lines.append(
            "  - MOVE_FORWARD onto a switch, then TOGGLE (hold switches activate on step). "
            "Gates cannot be toggled — activate their linked switch(es)."
        )
    if not lines:
        return ""
    return "Hints:\n" + "\n".join(lines) + "\n"
