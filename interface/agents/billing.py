"""Billing exhaustion: the provider account is out of credit or a spend cap was hit.

Agents raise ``BillingExhausted`` instead of a generic error so the run can pause
(next-run-fixes item 2): the runner ends the episode as
``agent_error:BillingExhausted: <provider>: ...``, the coordinator keeps it
resumable without counting it against MAX_RESUMES, and parks every model group
on that provider until someone tops up and unparks.
"""

from __future__ import annotations

BILLING_END_REASON_PREFIX = "agent_error:BillingExhausted:"


class BillingExhausted(RuntimeError):
    def __init__(self, provider: str, detail: str):
        self.provider = provider
        super().__init__(f"{provider}: {detail}")


class BillingParked(BillingExhausted):
    """Raised at an episode's next query once its provider is parked: another
    episode hit the provider's billing limit. The episode ends resumable."""

    def __init__(self, provider: str):
        super().__init__(provider, "parked: another episode hit this provider's billing limit")


def is_park_echo(end_reason: str | None) -> bool:
    """A BillingParked end_reason: the episode stopped because its provider was
    already parked. It must not (re)park the provider after an unpark."""
    if billing_provider(end_reason) is None:
        return False
    rest = end_reason[len(BILLING_END_REASON_PREFIX):].strip()
    return rest.split(":", 1)[-1].strip().startswith("parked:")


def billing_provider(end_reason: str | None) -> str | None:
    """The provider named by a billing end_reason, else None."""
    if not end_reason or not end_reason.startswith(BILLING_END_REASON_PREFIX):
        return None
    rest = end_reason[len(BILLING_END_REASON_PREFIX):].strip()
    return rest.split(":", 1)[0].strip() or None
