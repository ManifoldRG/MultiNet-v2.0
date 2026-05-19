"""
Cross-Domain Interface for MultiNet-v2.0

Provides canonical task specification and domain adapter abstractions
for evaluating models across different action domains.
"""

from .canonical_task_spec import (
    CanonicalGoal,
    CanonicalObject,
    CanonicalRules,
    CanonicalTaskSpec,
)
from .domain_adapter import DomainAdapter

__all__ = [
    "CanonicalTaskSpec",
    "CanonicalGoal",
    "CanonicalObject",
    "CanonicalRules",
    "DomainAdapter",
]
