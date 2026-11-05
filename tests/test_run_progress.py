"""Run progress for the dashboard (scripts/run_progress.py): how a coordinator's
state becomes episodes by state, API spend and time left. Pure: no coordinator,
no network. The coordinator lifecycle and the writer are in test_run_events.py.
"""

from __future__ import annotations

import json

import pytest

from interface.agents.openai_agent import OPENAI_PRICES
from scripts import run_progress
from scripts.run_progress import STATES, episode_cost, summarize, time_series, unit_progress_state, unit_prices

NOW = 1_800_000_000.0  # 2027-01-15T08:00:00Z


def _iso(seconds_ago: float) -> str:
    import datetime
    when = datetime.datetime.fromtimestamp(NOW - seconds_ago, datetime.timezone.utc)
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


# --- prices ----------------------------------------------------------------------

def test_openai_models_use_the_agents_own_price_table_and_tier():
    unit = {"provider": "openai", "model_config": {"model": "gpt-6-luna"}}
    assert unit_prices(unit) == OPENAI_PRICES["gpt-6-luna"]["default"]
    assert unit_prices({**unit, "model_config": {"model": "gpt-6-luna", "service_tier": "flex"}}) == \
           OPENAI_PRICES["gpt-6-luna"]["flex"]
    assert unit_prices(unit, "batch") == OPENAI_PRICES["gpt-6-luna"]["flex"]  # Batch is billed at the Flex rate


@pytest.mark.parametrize("provider, model, share", [("claude", "claude-opus-4-8", 0.5), ("kimi", "kimi-k2.6", 0.6)])
def test_batch_workers_pay_the_batch_share(provider, model, share):
    unit = {"provider": provider, "model_config": {"model": model}}
    assert unit_prices(unit, "batch") == pytest.approx(tuple(p * share for p in unit_prices(unit)))


@pytest.mark.parametrize("unit", [
    {"provider": "qwen_vllm_api", "model_config": {"model": "qwen36-27b"}},  # self-hosted: no API price
    {"provider": "openai", "model_config": {"model": "gpt-unknown"}},
    {"provider": "kimi", "model_config": {"model": "kimi-k3"}},  # no rate on record yet
])
def test_a_model_with_no_price_on_record_is_unpriced(unit):
    assert unit_prices(unit) is None


def _query(**usage):
    return {"kind": "query", "usage": usage}


def test_an_episode_costs_the_sum_of_its_queries_with_cache_reads_and_writes(tmp_path):
    unit = {"provider": "claude", "model_config": {"model": "claude-opus-4-8"}}
    # Normalized Anthropic usage: input_tokens is the whole prompt; reads and writes are part of it.
    claude = _query(input_tokens=1000, cache_read_input_tokens=600, cache_creation_input_tokens=100,
                    output_tokens=50, total_tokens=1050)
    episode = {"transcript": [{"kind": "reset"}, claude, {"kind": "step"}, claude, {"kind": "query"}]}
    per_query = (300 * 5.00 + 600 * 0.50 + 100 * 5.00 * 1.25 + 50 * 25.00) / 1e6
    assert episode_cost(unit, episode, tmp_path) == {"tokens": 2100, "usd": pytest.approx(2 * per_query)}


def test_openai_cached_tokens_reported_under_both_names_count_once(tmp_path):
    unit = {"provider": "openai", "model_config": {"model": "gpt-6-luna"}}
    episode = {"transcript": [_query(input_tokens=1000, cached_input_tokens=400, cache_read_input_tokens=400,
                                     output_tokens=100, total_tokens=1100)]}
    usd = (600 * 0.10 + 400 * 0.01 + 100 * 0.50) / 1e6
    assert episode_cost(unit, episode, tmp_path)["usd"] == pytest.approx(usd)


def test_the_run_inputs_pricing_tier_says_batch(tmp_path):
    unit = {"provider": "claude", "model_config": {"model": "claude-sonnet-4-6"}}
    episode = {"transcript": [_query(input_tokens=1000, output_tokens=100)]}
    sync = episode_cost(unit, episode, tmp_path)
    (tmp_path / "run_inputs.json").write_text(json.dumps({"pricing_tier": "batch"}))
    assert episode_cost(unit, episode, tmp_path)["usd"] == pytest.approx(sync["usd"] / 2)
    assert sync["tokens"] == 1100  # no total_tokens: input + output


def test_an_unpriced_episode_still_counts_its_tokens(tmp_path):
    unit = {"provider": "qwen_vllm_api", "model_config": {"model": "qwen36-27b"}}
    episode = {"transcript": [_query(input_tokens=10, output_tokens=5, total_tokens=15)]}
    assert episode_cost(unit, episode, tmp_path) == {"tokens": 15, "usd": None}


# --- states ----------------------------------------------------------------------

@pytest.mark.parametrize("unit_state, parked, expected", [
    ({"status": "verified", "end_reason": "success"}, False, "solved"),
    ({"status": "verified", "end_reason": "max_steps"}, False, "unsolved"),
    ({"status": "assigned"}, False, "running"),
    ({"status": "running"}, True, "running"),  # parking stops new hand-outs, not running episodes
    ({"status": "uploaded"}, False, "running"),
    ({"status": "pending", "attempts": 0}, False, "waiting"),
    ({"status": "pending", "attempts": 0}, True, "parked"),
    ({"status": "failed", "attempts": 1}, False, "waiting"),  # retried
    ({"status": "stale", "attempts": 2}, False, "waiting"),
    ({"status": "failed", "attempts": 3}, False, "gave_up"),  # out of attempts
    ({"status": "stale", "attempts": 3}, False, "gave_up"),
    ({"status": "resumable", "attempts": 3}, False, "waiting"),  # resumes burn no attempt
    ({"status": "resumable", "attempts": 1}, True, "parked"),
    ({"status": "failed", "attempts": 1, "terminal": True}, False, "gave_up"),  # a final failure
])
def test_each_unit_is_in_the_state_the_coordinator_treats_it_as(unit_state, parked, expected):
    assert unit_progress_state(unit_state, parked=parked, max_attempts=3) == expected


def _plan(*models: tuple[str, str, bool, int]) -> list[dict]:
    return [{"unit_id": f"{model}-{i}", "model_id": model, "model_group": group, "priced": priced}
            for model, group, priced, count in models for i in range(count)]


def test_the_states_split_the_plan_and_spend_is_projected_per_model():
    units = _plan(("luna", "openai-api", True, 5), ("qwen", "gpu", False, 2))
    unit_states = {
        "luna-0": {"status": "verified", "end_reason": "success", "api_cost": {"usd": 0.02}},
        "luna-1": {"status": "verified", "end_reason": "max_steps", "api_cost": {"usd": 0.04}},
        "luna-2": {"status": "resumable", "api_cost": {"usd": 0.01}},  # a killed partial already spent
        "luna-3": {"status": "running"},
        "qwen-0": {"status": "verified", "end_reason": "success", "api_cost": {"usd": None}},
        # luna-4, qwen-1: no state yet, i.e. pending
    }
    summary = summarize(units, {"units": unit_states, "parked_providers": {}}, max_attempts=3, now=NOW)
    luna, qwen = summary["models"]["luna"], summary["models"]["qwen"]
    assert luna["states"] == {"solved": 1, "unsolved": 1, "running": 1, "waiting": 2, "parked": 0, "gave_up": 0}
    assert sum(luna["states"].values()) == 5 and list(luna["states"]) == list(STATES)
    assert luna["spent"] == pytest.approx(0.07)
    assert luna["projected"] == pytest.approx(0.07 + 3 * 0.03)  # 3 not finished x mean of the 2 finished
    assert qwen["states"]["solved"] == 1 and qwen["spent"] is None and qwen["projected"] is None


def test_a_parked_provider_parks_its_groups_waiting_episodes():
    units = _plan(("luna", "openai-api", True, 2), ("opus", "claude-api", True, 1))
    state = {"units": {"luna-0": {"status": "running"}},
             "parked_providers": {"openai": {"groups": ["openai-api"]}}}
    summary = summarize(units, state, max_attempts=3, now=NOW)
    assert summary["models"]["luna"]["states"]["running"] == 1
    assert summary["models"]["luna"]["states"]["parked"] == 1
    assert summary["models"]["opus"]["states"]["waiting"] == 1


def test_no_projection_until_a_model_finishes_an_episode():
    units = _plan(("luna", "g", True, 2))
    summary = summarize(units, {"units": {"luna-0": {"status": "resumable", "api_cost": {"usd": 0.01}}}},
                        max_attempts=3, now=NOW)
    assert summary["models"]["luna"]["spent"] == pytest.approx(0.01)
    assert summary["models"]["luna"]["projected"] is None


# --- time left -------------------------------------------------------------------

def _done(seconds_ago: float) -> dict:
    return {"status": "verified", "end_reason": "success", "verified_at": _iso(seconds_ago),
            "assigned_at": _iso(seconds_ago + 60)}


def test_time_left_is_the_episodes_left_at_the_last_30_minutes_pace():
    units = _plan(("m", "g", True, 12))
    # An hour-old run: 2 done long ago, 4 in the last 30 minutes, 6 left -> 4 per 30 min -> 45 min.
    unit_states = {f"m-{i}": _done(t) for i, t in enumerate([3500, 3000, 1500, 1000, 600, 100])}
    unit_states["m-6"] = {"status": "assigned", "assigned_at": _iso(3600)}
    summary = summarize(units, {"units": unit_states}, max_attempts=3, now=NOW)
    assert summary["time_left_minutes"] == 45


def test_a_young_run_uses_its_whole_pace_and_a_stalled_one_its_overall_pace():
    units = _plan(("m", "g", True, 4))
    young = {"m-0": {**_done(60), "assigned_at": _iso(600)}}  # 1 done in 10 minutes, 3 left
    assert summarize(units, {"units": young}, max_attempts=3, now=NOW)["time_left_minutes"] == 30
    stalled = {"m-0": {**_done(3000), "assigned_at": _iso(3600)}}  # nothing for 50 minutes: 1 per hour
    assert summarize(units, {"units": stalled}, max_attempts=3, now=NOW)["time_left_minutes"] == 180


def test_time_left_is_unknown_before_anything_finishes_and_zero_when_nothing_is_left():
    units = _plan(("m", "g", True, 2))
    assert summarize(units, {"units": {}}, max_attempts=3, now=NOW)["time_left_minutes"] is None
    finished = {"m-0": _done(60), "m-1": {"status": "failed", "attempts": 3}}
    assert summarize(units, {"units": finished}, max_attempts=3, now=NOW)["time_left_minutes"] == 0


# --- what goes to Cloud Monitoring -----------------------------------------------

def test_the_time_series_carry_counts_spend_and_time_left_for_one_run():
    units = _plan(("luna", "g", True, 1), ("qwen", "h", False, 1))
    summary = summarize(units, {"units": {"luna-0": _done(60) | {"api_cost": {"usd": 0.5}}}},
                        max_attempts=3, now=NOW)
    series = time_series("r1", summary, "2027-01-15T08:00:00Z")
    by_type: dict[str, list] = {}
    for s in series:
        by_type.setdefault(s["metric"]["type"].removeprefix(run_progress.METRIC_PREFIX), []).append(s)
        assert s["metric"]["labels"]["run_id"] == "r1"
        assert s["points"] == [{"interval": {"endTime": "2027-01-15T08:00:00Z"}, "value": s["points"][0]["value"]}]
    assert len(by_type["episodes"]) == 2 * len(STATES)
    assert {s["metric"]["labels"]["state"] for s in by_type["episodes"]} == set(STATES)
    solved = [s for s in by_type["episodes"] if s["metric"]["labels"] == {"run_id": "r1", "model_id": "luna",
                                                                          "state": "solved"}]
    assert solved[0]["points"][0]["value"] == {"int64Value": "1"}  # INT64 travels as a JSON string
    assert {(s["metric"]["labels"]["model_id"], s["metric"]["labels"]["kind"]) for s in by_type["api_cost_usd"]} == \
           {("luna", "spent"), ("luna", "projected")}  # the unpriced model writes no spend
    [left] = by_type["time_left_minutes"]
    assert left["metric"]["labels"] == {"run_id": "r1"}
    assert left["points"][0]["value"] == {"int64Value": "2"}  # qwen's one episode, at 1 per 2 minutes
