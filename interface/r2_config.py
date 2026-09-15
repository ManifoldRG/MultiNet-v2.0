"""interface/r2_config.py - R2 sprint experiment profile.

Scopes the R2 protocol changes to a dedicated profile rather than changing
ExperimentConfig's existing defaults, so R1 configs/results stay reproducible.
"""
from __future__ import annotations

from interface.config import ExperimentConfig

# Placeholder pending team confirmation. If this ends up "rolling" or "full":
# checkpoint/resume breaks, since episode_checkpoint.py currently rejects
# non-stateless chat_history. Do not treat this profile as final until that's
# either confirmed acceptable or checkpoint support is extended.
R2_CHAT_HISTORY = "rolling"  # TODO(team): confirm - see PR description

R2_CONFIG = ExperimentConfig(
    context_window="full",
    max_history_tokens=100_000,
    chat_history=R2_CHAT_HISTORY,
    # Everything else intentionally left at ExperimentConfig's existing
    # defaults so this profile only deviates on the R2-scoped axes.
)