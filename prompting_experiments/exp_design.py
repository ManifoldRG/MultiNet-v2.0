"""Experiment prompt condition-set registry.

Each condition set is split into its own module to mirror the PR #12 experiment
design while keeping runnable prompt behavior centralized in ``interface``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Mapping

if TYPE_CHECKING:
    from interface.config import ExperimentConfig

from .condition_set_1_prompt import CONDITION_SET as CONDITION_SET_1
from .condition_set_2_observation_format import CONDITION_SET as CONDITION_SET_2
from .condition_set_3_context_window import CONDITION_SET as CONDITION_SET_3
from .condition_set_4_action_space import CONDITION_SET as CONDITION_SET_4
from .condition_set_5_querying_strategy import CONDITION_SET as CONDITION_SET_5
from .condition_set_6_in_context_learning import CONDITION_SET as CONDITION_SET_6
from .core import ConditionSet, Variant, iter_condition_configs as _iter_condition_configs


CONDITION_SETS: Mapping[str, ConditionSet] = {
    CONDITION_SET_1.name: CONDITION_SET_1,
    CONDITION_SET_2.name: CONDITION_SET_2,
    CONDITION_SET_3.name: CONDITION_SET_3,
    CONDITION_SET_4.name: CONDITION_SET_4,
    CONDITION_SET_5.name: CONDITION_SET_5,
    CONDITION_SET_6.name: CONDITION_SET_6,
}


def iter_condition_configs(
    condition_name: str,
    base: ExperimentConfig | None = None,
) -> Iterator[tuple[str, ExperimentConfig]]:
    """Yield runnable ``(variant_name, config)`` pairs for one condition set."""

    yield from _iter_condition_configs(CONDITION_SETS[condition_name], base)


__all__ = ["CONDITION_SETS", "ConditionSet", "Variant", "iter_condition_configs"]
