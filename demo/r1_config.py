"""Shared R1 human-play experiment config (desktop CLI + web API)."""

from __future__ import annotations

from interface.config import ExperimentConfig

DEFAULT_MANIFEST = "gridworld/fixtures/manifest.json"
DEFAULT_EXPERIMENT = "r1"

R1_CONFIG = ExperimentConfig(
    observation="image_only",
    context_window="text_summary_and_last3",
    include_current_observation_description=True,
    observation_text_includes_facing=True,
    action_space="egocentric",
    prompting="minimal",
    in_context_learning="zero_shot",
    progress_stall_k=30,
)
