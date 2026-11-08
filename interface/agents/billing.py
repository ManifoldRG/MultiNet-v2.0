"""Billing exhaustion: the provider account is out of credit or a spend cap was hit.

Agents raise ``BillingExhausted`` instead of a generic error so the run can pause:
the runner ends the episode as
``agent_error:BillingExhausted: <provider>: ...``, the coordinator keeps it
resumable without counting it against MAX_RESUMES, and parks every model group
on that provider until someone tops up and unparks.

A spend cap is a per-model budget, not an empty account, so its error names the
model (``BillingExhausted(provider, detail, model=...)``) and the end reason
reads ``agent_error:BillingExhausted: <provider>: model <model>: ...``; the
coordinator then parks only that model's groups. ``OpenAISpendCapReached`` sets
it. A per-model cap class plugs in the same way: subclass ``BillingExhausted``
(or a ``SpendCapReached`` that does), take the model, and pass ``model=`` up.
"""

from __future__ import annotations

import re

BILLING_END_REASON_PREFIX = "agent_error:BillingExhausted:"
# "<provider>: model <model>: <detail>". A model id has no whitespace but may hold
# a colon (us.anthropic.x-v2:0), so the model ends at the first ": ".
_MODEL_SCOPE = re.compile(r"[^:]*:\s*model (?P<model>\S+?): ")


class BillingExhausted(RuntimeError):
    def __init__(self, provider: str, detail: str, *, model: str | None = None):
        self.provider = provider
        self.model = model
        scope = provider if model is None else f"{provider}: model {model}"
        super().__init__(f"{scope}: {detail}")


class BillingParked(BillingExhausted):
    """Raised at an episode's next query once its provider (or its model) is
    parked: another episode hit a billing limit. The episode ends resumable."""

    def __init__(self, provider: str):
        super().__init__(provider, "parked: another episode hit this provider's billing limit or this model's spend cap")


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


def billing_model(end_reason: str | None) -> str | None:
    """The model a billing end_reason is scoped to (a spend cap), else None: the
    whole provider is out of credit."""
    if billing_provider(end_reason) is None:
        return None
    match = _MODEL_SCOPE.match(end_reason[len(BILLING_END_REASON_PREFIX):].strip())
    return match.group("model") if match else None
