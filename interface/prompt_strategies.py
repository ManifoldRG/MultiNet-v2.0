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
from prompting_experiments.prompt_templates import system as system_templates
from prompting_experiments.prompt_templates import user as user_templates

MECHANISM_LIST = system_templates.MECHANISM_LIST
MECHANISM_RULES = system_templates.MECHANISM_RULES
FINAL_OUTPUT_INSTRUCTION = system_templates.FINAL_OUTPUT_INSTRUCTION


class MinimalPromptStrategy:
    def __init__(self, actions_hint: str) -> None:
        self._actions_hint = actions_hint

    def build_system_prompt(self, querying_suffix: str = "") -> str:
        chunks = [
            system_templates.TASK_PREFIX,
            system_templates.VALID_ACTIONS_TEMPLATE.format(actions_hint=self._actions_hint),
            FINAL_OUTPUT_INSTRUCTION,
        ]
        if querying_suffix:
            chunks.append(querying_suffix)
        return "\n".join(chunks[:2]) + "\n" + "\n\n".join(chunks[2:])

    def build_user_prompt(
        self,
        obs_text: str,
        history_text: str,
        task_spec: TaskSpecification,
        state: GridState,
        last_feedback: str,
    ) -> str:
        history_block = f"{history_text}\n\n" if history_text else ""
        obs_block = (
            user_templates.OBSERVATION_SECTION.format(obs_text=obs_text)
            if obs_text
            else ""
        )
        pos = agent_row_col(state)
        goal = goal_row_col(task_spec)
        return user_templates.MINIMAL_USER_PROMPT.format(
            history_block=history_block,
            obs_block=obs_block,
            position=pos,
            facing=agent_facing(state),
            goal=goal,
            step_num=state.step_count + 1,
            max_steps=state.max_steps,
            last_feedback=last_feedback,
        )


class StandardPromptStrategy(MinimalPromptStrategy):
    def build_system_prompt(self, querying_suffix: str = "") -> str:
        chunks = [
            system_templates.TASK_PREFIX,
            MECHANISM_LIST,
            system_templates.VALID_ACTIONS_TEMPLATE.format(actions_hint=self._actions_hint),
            FINAL_OUTPUT_INSTRUCTION,
        ]
        if querying_suffix:
            chunks.append(querying_suffix)
        return "\n".join(chunks[:3]) + "\n" + "\n\n".join(chunks[3:])


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
            user_templates.BUDGET_WARNING.format(steps_left=steps_left)
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
            neighbour_lines.append(
                user_templates.NEIGHBOUR_LINE.format(
                    relative_direction=rel,
                    description=desc,
                )
            )
        neighbour_block = (
            user_templates.NEIGHBOUR_BLOCK_HEADER + "\n".join(neighbour_lines) + "\n"
        )
        mechanism_block = _mechanism_hints_text(task_spec)
        history_block = f"{history_text}\n\n" if history_text else ""
        obs_block = (
            user_templates.OBSERVATION_SECTION.format(obs_text=obs_text)
            if obs_text
            else ""
        )
        inventory_str = ", ".join(inventory_list(state)) or "none"

        return user_templates.VERBOSE_USER_PROMPT.format(
            history_block=history_block,
            obs_block=obs_block,
            position=(row, col),
            facing=agent_facing(state),
            goal=(grow, gcol),
            manhattan=manhattan,
            step_num=state.step_count + 1,
            max_steps=state.max_steps,
            steps_left=steps_left,
            inventory=inventory_str,
            budget_warn=budget_warn,
            neighbour_block=neighbour_block,
            mechanism_block=mechanism_block,
            last_feedback=last_feedback,
        )


PromptStrategy = MinimalPromptStrategy


def _mechanism_hints_text(task_spec: TaskSpecification) -> str:
    lines = []
    if task_spec.mechanisms.keys or task_spec.mechanisms.doors:
        lines.append(user_templates.KEY_DOOR_HINT)
    if task_spec.mechanisms.switches or task_spec.mechanisms.gates:
        lines.append(user_templates.SWITCH_GATE_HINT)
    if not lines:
        return ""
    return user_templates.MECHANISM_HINTS_HEADER + "\n".join(lines) + "\n"
