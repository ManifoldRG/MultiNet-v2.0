"""Project the R1-frontier (GPT-6 Astra, 50-maze panel) cost from calibration episodes.

Reads every ``episode.json`` under ``--artifacts-root`` (any layout: local run or
distributed worker roots), groups by model directory, and reports per model:
episodes, queries/episode, per-query input/output/reasoning tokens, truncations,
actual spend at the tier it ran on. Then projects the Astra panel:

    panel_cost = (mean tokens per episode) x --panel-mazes x Astra price

under three queries/episode scenarios (calibration-measured, R1 Opus, R1 Kimi)
because q/ep is the least transferable quantity from a 5-maze smoke. Per-query
tokens come from the Astra probe when one is present in the root, otherwise
from the proxy model (Terra) — proxy numbers are a guess about a different
model and are labelled as such.

Usage:
    python -m scripts.estimate_frontier_cost --artifacts-root <root> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from interface.agents.openai_agent import OPENAI_PRICES, usage_cost_usd

ASTRA = "gpt-6-astra"
# R1-20260717 actuals (Multinet-v2-results/r1-20260717, 50 episodes each, 64k cap):
# mean queries/episode = total queries / 50.
R1_QUERIES_PER_EPISODE = {"opus-4.8 (R1)": 1876 / 50, "kimi-k2.6 (R1)": 3237 / 50}


def _pct(xs: List[int], p: float) -> int:
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))] if s else 0


def collect(root: Path) -> Dict[str, Dict[str, Any]]:
    """Per-model aggregates keyed by the model directory name."""
    by_model: Dict[str, Dict[str, Any]] = {}
    for path in sorted(root.rglob("episode.json")):
        # .../runs/<task>/<backend>/<model>/seed_<n>/<variant>/episode.json
        model = path.parents[2].name
        ep = json.loads(path.read_text(encoding="utf-8"))
        agg = by_model.setdefault(
            model,
            {"episodes": 0, "successes": 0, "queries": [], "in": [], "out": [], "reasoning": [],
             "cached": [], "truncated": 0, "end_reasons": {}, "tasks": []},
        )
        agg["episodes"] += 1
        agg["successes"] += int(bool(ep.get("success")))
        end = str(ep.get("end_reason"))
        agg["end_reasons"][end] = agg["end_reasons"].get(end, 0) + 1
        agg["tasks"].append(path.parents[4].name)
        n = 0
        for turn in ep.get("transcript") or []:
            if turn.get("kind") != "query":
                continue
            n += 1
            usage = turn.get("usage") or {}
            agg["in"].append(int(usage.get("input_tokens", 0)))
            agg["out"].append(int(usage.get("output_tokens", 0)))
            agg["reasoning"].append(int(usage.get("reasoning_tokens", 0)))
            agg["cached"].append(int(usage.get("cached_input_tokens", 0)))
            if turn.get("stop_reason") in ("length", "max_tokens"):
                agg["truncated"] += 1
        agg["queries"].append(n)
    return by_model


def summarize(model: str, agg: Dict[str, Any]) -> Dict[str, Any]:
    q = sum(agg["queries"])
    eps = agg["episodes"]
    totals = {"input_tokens": sum(agg["in"]), "cached_input_tokens": sum(agg["cached"]),
              "output_tokens": sum(agg["out"])}
    spend = {}
    if model in OPENAI_PRICES:
        spend = {tier: round(usage_cost_usd(totals, prices), 2)
                 for tier, prices in OPENAI_PRICES[model].items()}
    return {
        "model": model,
        "episodes": eps,
        "successes": agg["successes"],
        "end_reasons": agg["end_reasons"],
        "queries": q,
        "queries_per_episode_mean": round(q / eps, 1) if eps else 0,
        "queries_per_episode_max": max(agg["queries"], default=0),
        "per_query_input_mean": round(totals["input_tokens"] / q) if q else 0,
        "per_query_output_mean": round(totals["output_tokens"] / q) if q else 0,
        "per_query_output_p50": statistics.median(agg["out"]) if agg["out"] else 0,
        "per_query_output_p90": _pct(agg["out"], 0.9),
        "per_query_output_p99": _pct(agg["out"], 0.99),
        "per_query_output_max": max(agg["out"], default=0),
        "per_query_reasoning_mean": round(sum(agg["reasoning"]) / q) if q else 0,
        "cached_input_share": round(totals["cached_input_tokens"] / totals["input_tokens"], 3)
        if totals["input_tokens"] else 0,
        "truncated_queries": agg["truncated"],
        "totals": totals,
        "actual_spend_usd_by_tier": spend,
    }


def project(profile: Dict[str, Any], panel_mazes: int) -> Dict[str, Any]:
    """Astra panel cost under each queries/episode scenario, at both tiers."""
    q = profile["queries"]
    per_q = {k: v / q for k, v in profile["totals"].items()} if q else {}
    scenarios = {f"calibration ({profile['model']})": profile["queries_per_episode_mean"],
                 **{k: round(v, 1) for k, v in R1_QUERIES_PER_EPISODE.items()}}
    out: Dict[str, Any] = {"token_profile_from": profile["model"],
                           "is_astra_measured": profile["model"] == ASTRA, "scenarios": {}}
    for name, qpe in scenarios.items():
        n_queries = qpe * panel_mazes
        usage = {k: v * n_queries for k, v in per_q.items()}
        out["scenarios"][name] = {
            "queries_per_episode": qpe,
            "panel_queries": round(n_queries),
            "panel_output_tokens_M": round(usage.get("output_tokens", 0) / 1e6, 2),
            "usd_flex": round(usage_cost_usd(usage, OPENAI_PRICES[ASTRA]["flex"]), 0),
            "usd_standard": round(usage_cost_usd(usage, OPENAI_PRICES[ASTRA]["default"]), 0),
        }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--artifacts-root", required=True, type=Path)
    ap.add_argument("--panel-mazes", type=int, default=50)
    ap.add_argument("--proxy-model", default="gpt-5.6-terra",
                    help="Model dir whose token profile stands in for Astra when no Astra probe exists.")
    ap.add_argument("--json", type=Path, help="Also write the full report here.")
    args = ap.parse_args(argv)

    by_model = collect(args.artifacts_root)
    if not by_model:
        print(f"no episode.json under {args.artifacts_root}", file=sys.stderr)
        return 1
    summaries = {m: summarize(m, a) for m, a in by_model.items()}
    source = ASTRA if ASTRA in summaries else args.proxy_model
    if source not in summaries:
        print(f"neither {ASTRA} nor proxy {args.proxy_model} found; have {sorted(summaries)}",
              file=sys.stderr)
        return 1
    report = {"models": summaries, "projection": project(summaries[source], args.panel_mazes)}

    for s in summaries.values():
        print(f"\n== {s['model']}: {s['episodes']} eps, {s['successes']} success, {s['queries']} queries "
              f"({s['queries_per_episode_mean']}/ep, max {s['queries_per_episode_max']}) end={s['end_reasons']}")
        print(f"   per-query: in {s['per_query_input_mean']}  out mean {s['per_query_output_mean']} "
              f"p50 {s['per_query_output_p50']} p90 {s['per_query_output_p90']} "
              f"p99 {s['per_query_output_p99']} max {s['per_query_output_max']}  "
              f"reasoning mean {s['per_query_reasoning_mean']}  truncated {s['truncated_queries']}  "
              f"cached share {s['cached_input_share']}")
        print(f"   actual spend by tier: {s['actual_spend_usd_by_tier']}")
    proj = report["projection"]
    label = "MEASURED on Astra" if proj["is_astra_measured"] else f"PROXY ({source}) — not Astra"
    print(f"\n== Astra {args.panel_mazes}-maze panel projection, token profile {label}")
    for name, sc in proj["scenarios"].items():
        print(f"   {name:28s} q/ep {sc['queries_per_episode']:6}  queries {sc['panel_queries']:6}  "
              f"out {sc['panel_output_tokens_M']:7}M  flex ${sc['usd_flex']:>7}  standard ${sc['usd_standard']:>7}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
