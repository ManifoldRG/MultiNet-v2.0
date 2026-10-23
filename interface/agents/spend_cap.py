"""Per-model hard spend caps shared by the API agents (Claude, Kimi, OpenAI).

``spend_cap_usd`` on a run-config model entry caps that model's priced usage in
this process: once the model's running total reaches the cap, further requests
raise ``SpendCapReached`` instead of spending. Totals are kept per model name,
so models sharing a process (a local multi-model run) never draw on each other's
cap, while every agent instance of one model (a worker's concurrent episodes)
shares one total. A request already in flight when the cap is reached still
completes, so a run can end slightly over its cap. Restarting the process resets
the totals, so pair each cap with a provider-side budget limit.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional


class SpendCapReached(RuntimeError):
    """Raised before a request once the model's spend cap is reached."""


class SpendLedger:
    """Process-wide priced-usage totals, one per model name."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._usd: Dict[str, float] = {}

    def add(self, model: str, usd: float) -> float:
        with self._lock:
            self._usd[model] = self._usd.get(model, 0.0) + usd
            return self._usd[model]

    def total(self, model: str) -> float:
        with self._lock:
            return self._usd.get(model, 0.0)

    def reset(self) -> None:
        with self._lock:
            self._usd.clear()


SPEND = SpendLedger()


def check_spend_cap(
    agent_name: str,
    model: str,
    cap: Optional[float],
    error: type[SpendCapReached] = SpendCapReached,
) -> None:
    """Raise ``error`` if ``model`` has already spent ``cap`` USD in this process."""
    if cap is None:
        return
    spent = SPEND.total(model)
    if spent >= cap:
        raise error(
            f"{agent_name} spend cap ${cap:.2f} for {model} reached "
            f"(running total ${spent:.2f}); refusing request."
        )
