"""Shared types for prompt experiment condition registries."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Iterator, Mapping

if TYPE_CHECKING:
    from interface.config import ExperimentConfig


@dataclass(frozen=True)
class Variant:
    """One experiment variant expressed as overrides to ``ExperimentConfig``."""

    name: str
    description: str
    config_overrides: Mapping[str, object] | None = None
    implemented: bool = True

    def build_config(self, base: ExperimentConfig | None = None) -> ExperimentConfig:
        if not self.implemented:
            raise ValueError(f"Variant is not implemented in ExperimentConfig: {self.name}")
        from interface.config import ExperimentConfig

        cfg = base or ExperimentConfig()
        return replace(cfg, **dict(self.config_overrides or {}))


@dataclass(frozen=True)
class ConditionSet:
    """A named experimental axis and its comparable variants."""

    name: str
    comparisons: tuple[str, ...]
    decision: str
    variants: Mapping[str, Variant]
    implemented: bool = True
    notes: str = ""


def iter_condition_configs(
    condition: ConditionSet,
    base: ExperimentConfig | None = None,
) -> Iterator[tuple[str, ExperimentConfig]]:
    """Yield ``(variant_name, config)`` pairs for implemented variants."""

    for variant_name, variant in condition.variants.items():
        if variant.implemented:
            yield variant_name, variant.build_config(base)
