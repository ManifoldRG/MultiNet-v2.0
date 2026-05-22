"""Prompt strategies for the NLU benchmark.

  minimal   – goal + action list. Per-turn user text is ``render_user_observation_text``;
               initial layout is in the system message when observation is text or image+text.
  standard  – system prompt adds the static ``MECHANISM_LIST``; user layout same as
               minimal for text observation content.
  verbose   – system: mechanism list + domain rules; user: neighbour view,
               inventory, per-step mechanism hints.

Initial maze NL is ``render_initial_maze_text`` in the system prompt; each user
turn includes ``render_user_observation_text`` (when text or image+text), not
here.
"""

from __future__ import annotations

from nlu_benchmark.env import FACING_ORDER, FACING_TO_DELTA

# Standard system prompt: high-level object types. Verbose reuses this and adds
# MECHANISM_RULES + per-step hints in the user turn.
MECHANISM_LIST = (
    "The environment may contain:\n"
    "- Keys: pick them up to open doors of the matching color\n"
    "- Doors: blocked passages that require a matching key\n"
    "- Switches: toggle these to open or close linked gates\n"
    "- Gates: blocked passages controlled by switches\n"
)

# Verbose system prompt: operational rules (action semantics). Not in Standard.
MECHANISM_RULES = (
    "RULES (domain logic):\n"
    "  - PICKUP: take a key on your current cell and store it in your inventory.\n"
    "  - Doors: keys and doors are color-matched. With the matching key in your inventory, move onto\n"
    "    the door to open it\n"
    "  - Switches: face a switch and TOGGLE to flip it on or off. Only switches are toggled. Linked\n"
    "    gates are open if at least one linked switch is on, and closed if all are off.\n"
    "  - Gates: you cannot TOGGLE a gate. CLOSED gates block movement; OPEN gates do not.\n"
    "  - Closed gates and doors you lack a key for block movement like walls until resolved.\n"
    "  - Use DONE only when you are standing on the goal cell."
)

# How models must terminate the reply (Minimal + Standard + Verbose base).
FINAL_OUTPUT_INSTRUCTION = (
    "On the last line, output exactly:\n"
    "FINAL_OUTPUT: <action>  or  FINAL_OUTPUT: <a>, <b>, ...  "
    "(comma-separated; one or more valid actions)"
)


class PromptStrategy:
    """Base: shared action hint injection."""

    def __init__(self, actions_hint: str) -> None:
        self._actions_hint = actions_hint

    def build_system_prompt(self, querying_suffix: str = "") -> str:
        raise NotImplementedError

    def build_user_prompt(
        self,
        obs_text: str,
        history_text: str,
        state,
        last_feedback: str,
    ) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Minimal — goal + action list only
# ---------------------------------------------------------------------------

class MinimalPromptStrategy(PromptStrategy):
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
        state,
        last_feedback: str,
    ) -> str:
        history_block = f"{history_text}\n\n" if history_text else ""
        obs_block     = f"Observation:\n{obs_text}\n\n" if obs_text else ""
        return (
            f"{history_block}"
            f"{obs_block}"
            f"Position: {state.agent_pos}  |  Facing: {state.facing}  |  Goal: {state.goal}  |  "
            f"Step {state.step_count + 1}/{state.max_steps}\n"
            f"Last result: {last_feedback}\n"
            "What is your next action?"
        )


# ---------------------------------------------------------------------------
# Standard — mechanism list only (user prompt same as Minimal)
# ---------------------------------------------------------------------------

class StandardPromptStrategy(MinimalPromptStrategy):
    def build_system_prompt(self, querying_suffix: str = "") -> str:
        return (
            "Task: move to the goal cell in the grid.\n"
            f"{MECHANISM_LIST}\n"
            f"Valid actions: {self._actions_hint}.\n"
            f"{FINAL_OUTPUT_INSTRUCTION}"
            + (f"\n\n{querying_suffix}" if querying_suffix else "")
        )


# ---------------------------------------------------------------------------
# Verbose — mechanism list + rules (system); optional hint lines (user)
# ---------------------------------------------------------------------------

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
        state,
        last_feedback: str,
    ) -> str:
        steps_left  = state.max_steps - state.step_count
        budget_warn = (
            f"  WARNING: Only {steps_left} steps remaining!\n"
            if steps_left <= max(5, state.max_steps // 5)
            else ""
        )
        row, col = state.agent_pos
        grow, gcol = state.goal
        manhattan = abs(row - grow) + abs(col - gcol)

        facing_idx = FACING_ORDER.index(state.facing)
        rel_dirs = [
            ("AHEAD",  FACING_ORDER[facing_idx % 4]),
            ("RIGHT",  FACING_ORDER[(facing_idx + 1) % 4]),
            ("BEHIND", FACING_ORDER[(facing_idx + 2) % 4]),
            ("LEFT",   FACING_ORDER[(facing_idx + 3) % 4]),
        ]
        neighbour_lines = []
        for rel, cardinal in rel_dirs:
            dr, dc = FACING_TO_DELTA[cardinal]
            nr, nc = row + dr, col + dc
            if nr < 1 or nr > state.rows or nc < 1 or nc > state.cols:
                desc = "out of bounds"
            elif (nr, nc) in state.walls:
                desc = "wall"
            elif (nr, nc) == state.goal:
                desc = f"GOAL ({nr},{nc})"
            else:
                desc = f"open ({nr},{nc})"
            neighbour_lines.append(f"  {rel}: {desc}")
        neighbour_block = "From your perspective:\n" + "\n".join(neighbour_lines) + "\n"

        mechanism_block = _mechanism_hints_text(state)

        history_block = f"{history_text}\n\n" if history_text else ""
        obs_block     = f"Observation:\n{obs_text}\n\n" if obs_text else ""
        inventory_str = ", ".join(state.inventory) if state.inventory else "none"

        return (
            f"{history_block}"
            f"{obs_block}"
            f"Position: {state.agent_pos}  |  Facing: {state.facing}  |  Goal: {state.goal}  |  "
            f"Manhattan: {manhattan}  |  Step {state.step_count + 1}/{state.max_steps} ({steps_left} left)\n"
            f"Inventory: {inventory_str}\n"
            f"{budget_warn}"
            f"{neighbour_block}"
            f"{mechanism_block}"
            f"Last result: {last_feedback}\n"
            "What is your next action?"
        )


def _mechanism_hints_text(state) -> str:
    """Short reminders when the map has interactive objects; observation still has details."""
    lines = []
    if state.keys or state.doors:
        lines.append("  - PICKUP keys; with the right key, MOVE_FORWARD into a door to open it.")
    if state.switches or state.gates:
        lines.append("  - Face a switch and TOGGLE; gates follow linked switches (do not TOGGLE gates).")
    if not lines:
        return ""
    return "Hints:\n" + "\n".join(lines) + "\n"
