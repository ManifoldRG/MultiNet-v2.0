"""Progress stall detection shared by the LLM runner and playable demo."""

from __future__ import annotations


def _progress_signature(state) -> tuple:
    """A hashable summary of durable episode progress. Facing is excluded so
    spinning is not progress; explored_cells counts only under partial
    observation, where a turn can reveal genuinely new cells."""
    explored = (
        frozenset(tuple(int(coord) for coord in cell) for cell in state.explored_cells)
        if getattr(state, "observability_mode", "full") != "full"
        else frozenset()
    )
    blocks = frozenset(
        (
            block_id,
            tuple(int(coord) for coord in position),
        )
        for block_id, position in state.block_positions.items()
    )
    # Keys move: DROP puts a held key back on the grid, so a retrace over
    # previously visited cells with the key elsewhere is genuine progress.
    # Without DROP the key layout is a pure function of collected_keys, so
    # this axis never splits states a DROP-free episode saw as equal.
    keys = frozenset(
        (
            key_id,
            tuple(int(coord) for coord in position),
        )
        for key_id, position in (getattr(state, "key_positions", None) or {}).items()
    )
    return (
        tuple(int(coord) for coord in state.agent_position),
        state.agent_carrying,
        frozenset(state.collected_keys),
        frozenset(state.open_doors),
        frozenset(state.active_switches),
        frozenset(state.open_gates),
        blocks,
        keys,
        explored,
    )


class ProgressStallWatchdog:
    """End an episode after K consecutive steps with no new progress signature."""

    def __init__(self, k: int, initial_state) -> None:
        self.k = k
        self.seen_signatures = {_progress_signature(initial_state)}
        self.stall_count = 0

    def observe(self, state) -> bool:
        """Record ``state`` and return True once ``stall_count`` reaches ``k``."""
        sig = _progress_signature(state)
        if sig in self.seen_signatures:
            self.stall_count += 1
        else:
            self.seen_signatures.add(sig)
            self.stall_count = 0
        return self.stall_count >= self.k
