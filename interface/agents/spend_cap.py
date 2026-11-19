"""Per-model hard spend caps shared by the API agents (Claude, Kimi, OpenAI).

``spend_cap_usd`` on a run-config model entry caps that model's priced usage in
this process: once the model's running total reaches the cap, further requests
raise ``SpendCapReached`` instead of spending. Totals are kept per model name,
so models sharing a process (a local multi-model run) never draw on each other's
cap, while every agent instance of one model (a worker's concurrent episodes)
shares one total. A request already in flight when the cap is reached still
completes, so a run can end slightly over its cap.

A restart does not hand the cap back: the first time a process builds a capped
model's agent, the fleet worker and the local pipeline restore that model's
total from the priced usage already recorded in the run's directories
(``restore_spend_once`` / ``restore_spend``): each finished ``episode.json``,
and for a partial the larger of its crash ``checkpoint.json`` and its per-query
``queries.jsonl``. Spend that left no record (a discarded attempt, a request in
flight when the process died) is not restored, so still pair each cap with a
provider-side budget limit.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from interface.agents.billing import BillingExhausted


class SpendCapReached(BillingExhausted):
    """Raised before a request once the model's spend cap is reached. A billing
    error scoped to the model: the episode ends resumable and the coordinator
    parks that model's groups (interface.agents.billing)."""

    provider_name = "api"

    def __init__(self, detail: str, *, model: Optional[str] = None):
        super().__init__(self.provider_name, detail, model=model)


class ClaudeSpendCapReached(SpendCapReached):
    provider_name = "claude"


class KimiSpendCapReached(SpendCapReached):
    provider_name = "kimi"


class SpendLedger:
    """Process-wide priced-usage totals, one per model name."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._usd: Dict[str, float] = {}
        self._restored: set[str] = set()

    def add(self, model: str, usd: float) -> float:
        with self._lock:
            self._usd[model] = self._usd.get(model, 0.0) + usd
            return self._usd[model]

    def total(self, model: str) -> float:
        with self._lock:
            return self._usd.get(model, 0.0)

    def restore(self, model: str, usd: float) -> float:
        """Raise ``model``'s total to ``usd`` (spend recorded on disk); never lowers it."""
        with self._lock:
            self._usd[model] = max(self._usd.get(model, 0.0), float(usd))
            return self._usd[model]

    def restore_once(self, model: str, recorded_usd: Callable[[], float]) -> float:
        """The first call for ``model`` in this process restores its total from
        ``recorded_usd()``; later calls only read it. The lock is held while it
        reads the disk, so no request passes the cap check before the restore."""
        with self._lock:
            if model not in self._restored:
                usd = float(recorded_usd())
                self._restored.add(model)
                self._usd[model] = max(self._usd.get(model, 0.0), usd)
            return self._usd.get(model, 0.0)

    def reset(self) -> None:
        """Forget every total, as a new process would (tests)."""
        with self._lock:
            self._usd.clear()
            self._restored.clear()


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
            f"(running total ${spent:.2f}); refusing request.",
            model=model,
        )


# --------------------------------------------------------------------------- #
# Restoring a total from what a run recorded on disk
# --------------------------------------------------------------------------- #
def price_recorded_usage(model_cfg: Mapping[str, Any], usage: Optional[Mapping[str, Any]], *, batch: bool) -> float:
    """USD for one recorded reply's usage, priced the way the model's agent priced
    it when it spent (same tables, cache TTL and batch discount). A model with no
    price on record counts nothing; its agent refuses a cap at construction."""
    if not usage:
        return 0.0
    provider = str(model_cfg.get("provider") or "").lower()
    model = str(model_cfg.get("model") or "")
    usage = dict(usage)
    try:
        if provider == "claude":
            from interface.agents.claude import CLAUDE_PRICES, ClaudeAnthropicConfig, claude_usage_cost_usd

            prices = CLAUDE_PRICES.get(model)
            if prices is None:
                return 0.0
            fields = ClaudeAnthropicConfig.__dataclass_fields__
            key = "batch_cache_ttl" if batch else "cache_ttl"
            ttl = model_cfg.get(key) or fields[key].default
            return claude_usage_cost_usd(usage, prices, cache_ttl=ttl, batch=batch)
        if provider == "kimi":
            from interface.agents.kimi_k26 import KIMI_PRICES, kimi_usage_cost_usd

            prices = KIMI_PRICES.get(model)
            if prices is None:
                return 0.0
            # Batch requests are priced without a cache TTL (KimiK26Agent.generate_batch).
            ttl = None if batch else model_cfg.get("prompt_cache_ttl")
            return kimi_usage_cost_usd(usage, prices, cache_ttl=ttl)
        if provider == "openai":
            from interface.agents.openai_agent import price_for, usage_cost_usd

            return usage_cost_usd(usage, price_for(model, model_cfg.get("service_tier")))
    except (ValueError, TypeError):
        return 0.0
    return 0.0


def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _transcript_usages(record: Optional[dict]) -> list:
    transcript = (record or {}).get("transcript") or []
    return [rec.get("usage") for rec in transcript if isinstance(rec, dict) and rec.get("kind") == "query"]


def _ledger_usages(path: Path) -> list:
    usages = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return usages
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            usages.append(entry.get("usage"))
    return usages


def recorded_spend_usd(run_dir: str | Path, model_cfg: Mapping[str, Any], *, batch: bool) -> float:
    """The priced usage one run directory records: every query of its finished
    ``episode.json`` (a resumed episode carries the queries from before its kill),
    else, for a partial, the larger of its crash ``checkpoint.json`` (applied
    queries) and its ``queries.jsonl`` ledger (every reply). ``run_inputs.json``'s
    ``pricing_tier`` says whether the run was billed at batch rates; ``batch`` is
    the default for a partial that has no run_inputs yet."""
    run_dir = Path(run_dir)
    inputs = _read_json(run_dir / "run_inputs.json")
    if inputs is not None:
        batch = inputs.get("pricing_tier") == "batch"

    def priced(usages) -> float:
        return sum(price_recorded_usage(model_cfg, u, batch=batch) for u in usages)

    episode = _read_json(run_dir / "episode.json")
    if episode is not None:
        return priced(_transcript_usages(episode))
    return max(
        priced(_transcript_usages(_read_json(run_dir / "checkpoint.json"))),
        priced(_ledger_usages(run_dir / "queries.jsonl")),
    )


def restore_spend(model_cfg: Mapping[str, Any], run_dirs: Iterable[str | Path], *, batch: bool) -> float:
    """Raise the model's running total to what ``run_dirs`` record; returns the total."""
    total = sum(recorded_spend_usd(d, model_cfg, batch=batch) for d in run_dirs)
    return SPEND.restore(str(model_cfg.get("model") or ""), total)


def restore_spend_once(
    model_cfg: Mapping[str, Any], run_dirs: Callable[[], Iterable[str | Path]], *, batch: bool
) -> float:
    """``restore_spend`` the first time this process asks for the model; a no-op after."""
    return SPEND.restore_once(
        str(model_cfg.get("model") or ""),
        lambda: sum(recorded_spend_usd(d, model_cfg, batch=batch) for d in run_dirs()),
    )
