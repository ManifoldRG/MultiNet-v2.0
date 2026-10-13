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


def billing_provider(end_reason: str | None) -> str | None:
    """The provider named by a billing end_reason, else None."""
    if not end_reason or not end_reason.startswith(BILLING_END_REASON_PREFIX):
        return None
    rest = end_reason[len(BILLING_END_REASON_PREFIX):].strip()
    return rest.split(":", 1)[0].strip() or None
