"""Where a fleet run stands, for the run dashboard: episodes by state, API spend, time left.

With ``coordinator-serve --cloud-logging`` the event writer (``scripts/run_events.py``)
calls a ``RunProgress`` every 30 seconds and writes what it returns to Cloud
Monitoring as three custom metrics (``deploy/MONITORING.md``):

- ``multinet/episodes`` {run_id, model_id, state}: the planned episodes by state.
  The states split the plan, so they add up to the planned count:
  ``solved`` / ``unsolved`` (verified), ``running`` (handed out or uploading),
  ``waiting`` (not started, or due a retry or a resume), ``parked`` (waiting, but
  its provider is parked on a billing limit) and ``gave_up`` (out of attempts, or
  a failure that is not retried).
- ``multinet/api_cost_usd`` {run_id, model_id, kind}: ``spent`` so far and
  ``projected`` for the whole plan, for models with a price on record.
- ``multinet/time_left_minutes`` {run_id}: the episodes left, at the recent pace.

``RunProgress`` only reads the coordinator's plan and state files (both written
atomically), so it never takes the coordinator's lock. Spend comes from
``episode_cost``, which the coordinator runs on each upload.
"""

from __future__ import annotations

import datetime
import json
import math
import time
from pathlib import Path
from typing import Any, Optional

from interface.agents.openai_agent import OPENAI_PRICES, usage_cost_usd

METRIC_PREFIX = "custom.googleapis.com/multinet/"
STATES = ("solved", "unsolved", "running", "waiting", "parked", "gave_up")
PACE_WINDOW_SECONDS = 30 * 60

# USD per 1M tokens: (input, cached input, output), standard tier. OpenAI models
# use OPENAI_PRICES, the table their spend cap uses. Cache writes are priced at
# 1.25x input (Anthropic's 5-minute rate) by usage_cost_usd.
# Claude: https://platform.claude.com/docs/en/about-claude/pricing (fetched 2026-11-04).
# Kimi: the rates in docs/batch-api-lockstep-runner-design.md (July 2026). No cached
# rate is on record, so cached input is priced as input (an overestimate).
API_PRICES: dict[str, tuple[float, float, float]] = {
    "claude-opus-5-5": (4.00, 0.20, 20.00),
    "claude-opus-4-8": (5.00, 0.50, 25.00),
    "claude-sonnet-4-6": (3.00, 0.30, 15.00),
    "kimi-k2.6": (0.95, 0.95, 4.00),
}
# Batch APIs (lockstep workers record pricing_tier "batch"): share of the standard
# price. OpenAI's Batch rate is its Flex rate.
BATCH_SHARE = {"claude": 0.5, "kimi": 0.6}


def unit_prices(unit: dict[str, Any], pricing_tier: Optional[str] = None) -> Optional[tuple[float, float, float]]:
    """Per-1M-token (input, cached input, output) USD for a plan unit, or None when
    no price is on record (self-hosted models among them)."""
    config = unit.get("model_config") or {}
    model = str(config.get("model") or unit.get("model_id") or "")
    batch = pricing_tier == "batch"
    if unit.get("provider") == "openai":
        tiers = OPENAI_PRICES.get(model)
        if tiers is None:
            return None
        return tiers["flex" if batch or config.get("service_tier") == "flex" else "default"]
    prices = API_PRICES.get(model)
    if prices is None:
        return None
    share = BATCH_SHARE.get(str(unit.get("provider")), 1.0) if batch else 1.0
    return (prices[0] * share, prices[1] * share, prices[2] * share)


def episode_cost(unit: dict[str, Any], episode: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Tokens and API cost (USD, None when unpriced) of an uploaded episode, summed
    over its queries' recorded usage. A resumed episode's transcript carries the
    queries from before the kill, so this is the unit's spend so far."""
    try:
        tier = json.loads((Path(run_dir) / "run_inputs.json").read_text(encoding="utf-8")).get("pricing_tier")
    except (OSError, ValueError):
        tier = None
    prices = unit_prices(unit, tier)
    tokens, usd = 0, 0.0
    for record in episode.get("transcript") or []:
        usage = record.get("usage") if record.get("kind") == "query" else None
        if not usage:
            continue
        tokens += int(usage.get("total_tokens") or int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0)))
        if prices:
            cached = max(int(usage.get("cached_input_tokens", 0)), int(usage.get("cache_read_input_tokens", 0)))
            usd += usage_cost_usd({**usage, "cached_input_tokens": cached}, prices)
    return {"tokens": tokens, "usd": round(usd, 8) if prices else None}


def _epoch(iso: Any) -> Optional[float]:
    try:
        then = datetime.datetime.strptime(str(iso), "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None
    return then.replace(tzinfo=datetime.timezone.utc).timestamp()


def unit_progress_state(unit_state: dict[str, Any], *, parked: bool, max_attempts: int) -> str:
    """Which of STATES a unit is in, by the rules the coordinator hands units out by."""
    status = unit_state.get("status")
    if status == "verified":
        return "solved" if unit_state.get("end_reason") == "success" else "unsolved"
    if status in ("assigned", "running", "uploaded"):
        return "running"
    if unit_state.get("terminal") or (
        status != "resumable" and int(unit_state.get("attempts", 0)) >= max_attempts
    ):
        return "gave_up"
    return "parked" if parked else "waiting"


def summarize(units: list[dict[str, Any]], state: dict[str, Any], *, max_attempts: int,
              now: float) -> dict[str, Any]:
    """Per model: episodes by state, spent and projected USD; and the run's minutes left.

    ``units`` are the plan's units (``unit_id``, ``model_id``, ``model_group``,
    ``priced``). Projected = spent + the episodes not finished x the mean cost of
    the model's verified episodes; None until one has verified."""
    parked_groups = {g for info in (state.get("parked_providers") or {}).values() for g in info.get("groups", [])}
    unit_states = state.get("units") or {}
    models: dict[str, dict[str, Any]] = {}
    finished_costs: dict[str, list[float]] = {}
    done_at: list[float] = []
    begun: list[float] = []
    for unit in units:
        model = str(unit.get("model_id"))
        unit_state = unit_states.get(unit["unit_id"]) or {"status": "pending"}
        progress = unit_progress_state(unit_state, parked=unit.get("model_group") in parked_groups,
                                       max_attempts=max_attempts)
        entry = models.setdefault(model, {"states": dict.fromkeys(STATES, 0), "spent": 0.0 if unit.get("priced") else None,
                                          "projected": None})
        entry["states"][progress] += 1
        usd = (unit_state.get("api_cost") or {}).get("usd")
        if usd is not None and entry["spent"] is not None:
            entry["spent"] += usd
            if progress in ("solved", "unsolved"):
                finished_costs.setdefault(model, []).append(usd)
        if progress in ("solved", "unsolved") and (t := _epoch(unit_state.get("verified_at"))) is not None:
            done_at.append(t)
        if (t := _epoch(unit_state.get("assigned_at"))) is not None:
            begun.append(t)

    for model, entry in models.items():
        costs = finished_costs.get(model)
        if entry["spent"] is not None and costs:
            left = entry["states"]["running"] + entry["states"]["waiting"] + entry["states"]["parked"]
            entry["projected"] = entry["spent"] + left * sum(costs) / len(costs)

    left = sum(e["states"][s] for e in models.values() for s in ("running", "waiting", "parked"))
    return {"models": models, "time_left_minutes": _minutes_left(left, done_at, begun, now)}


def _minutes_left(left: int, done_at: list[float], begun: list[float], now: float) -> Optional[int]:
    """Episodes left over the pace of the last 30 minutes (or of the whole run, when
    it is younger or nothing finished in that window). None before any finishes."""
    if left == 0:
        return 0
    if not done_at:
        return None
    start = min(begun + done_at)
    window_start = max(start, now - PACE_WINDOW_SECONDS)
    recent = [t for t in done_at if t >= window_start]
    pace = len(recent) / max(now - window_start, 60.0) if recent else len(done_at) / max(now - start, 60.0)
    return math.ceil(left / pace / 60.0)


def time_series(run_id: str, summary: dict[str, Any], end_time: str) -> list[dict[str, Any]]:
    """``summarize``'s result as Cloud Monitoring time series (without the resource,
    which the writer adds)."""
    def point(metric: str, labels: dict[str, str], value: dict[str, Any]) -> dict[str, Any]:
        return {"metric": {"type": METRIC_PREFIX + metric, "labels": {"run_id": run_id, **labels}},
                "points": [{"interval": {"endTime": end_time}, "value": value}]}

    series = []
    for model, entry in summary["models"].items():
        for state, count in entry["states"].items():
            series.append(point("episodes", {"model_id": model, "state": state}, {"int64Value": str(count)}))
        for kind in ("spent", "projected"):
            if entry[kind] is not None:
                series.append(point("api_cost_usd", {"model_id": model, "kind": kind},
                                    {"doubleValue": round(entry[kind], 6)}))
    if summary["time_left_minutes"] is not None:
        series.append(point("time_left_minutes", {}, {"int64Value": str(summary["time_left_minutes"])}))
    return series


class RunProgress:
    """The run's progress now, as Cloud Monitoring time series: ``RunEvents.watch``
    calls it every 30 seconds. Reads the plan once (it does not change while the
    coordinator serves) and the state file on each call."""

    def __init__(self, plan_file: str | Path, state_file: str | Path, *, max_attempts: int) -> None:
        self._plan_file, self._state_file = Path(plan_file), Path(state_file)
        self._max_attempts = int(max_attempts)
        self._run_id: Optional[str] = None
        self._units: list[dict[str, Any]] = []

    def __call__(self) -> list[dict[str, Any]]:
        if self._run_id is None:  # first call, on the writer thread: a bad plan never reaches serve
            plan = json.loads(self._plan_file.read_text(encoding="utf-8"))
            self._units = [
                {"unit_id": u["unit_id"], "model_id": u.get("model_id"), "model_group": u.get("model_group"),
                 "priced": unit_prices(u) is not None}
                for u in plan.get("units", [])
            ]
            self._run_id = str(plan.get("run_set_id") or "")
        state = json.loads(self._state_file.read_text(encoding="utf-8"))
        now = time.time()
        end_time = datetime.datetime.fromtimestamp(now, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return time_series(self._run_id, summarize(self._units, state, max_attempts=self._max_attempts, now=now),
                           end_time)
