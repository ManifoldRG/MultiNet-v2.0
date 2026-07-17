"""R1 batch-API validation-smoke driver (task A10) — an operator tool.

Runs the 5-maze smoke manifest through the Claude/Kimi **Batch API** lockstep
runner (``interface.batch_runner.LockstepBatchRunner``) and, for the same mazes,
a tiny **unbatched control** through the plain serial runner (sync API), then
writes ``smoke_report.json`` with the price/behaviour delta and an updated R1
budget projection.

THIS SCRIPT SPENDS REAL MONEY. Two guards:
  * it refuses to run unless ``SMOKE_BUDGET_ACK=1`` is set in the environment
    (``--dry-run`` is exempt — it builds + prints the plan and calls no agent);
  * it hard-stops submitting new batch rounds once the running estimated spend
    (tokens-so-far x per-MTok prices) exceeds ``--max-usd`` (default $25).

The episode-building machinery is reused verbatim from the coordinator path
(``prepare_job`` -> per-unit ``_prepare_unit_run`` + ``build_episode_runner`` +
``EpisodeStepper``), the same seam ``tests/test_lockstep_worker.py`` exercises,
so the smoke measures the exact runner R1 will use.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

# July-2026 per-MTok USD prices (input, output). Batch = discounted tier, sync =
# on-demand tier. Source: docs/batch-api-lockstep-runner-design.md (Anthropic
# Message Batches 50% off; Moonshot Batch 40% off). Keyed by provider.
PRICES: dict[str, dict[str, tuple[float, float]]] = {
    "claude": {"batch": (2.50, 12.50), "sync": (5.00, 25.00)},
    "kimi": {"batch": (0.57, 2.40), "sync": (0.95, 4.00)},
}

# Prior central R1 (1-seed, 50-maze) budget from the design spec, for context in
# the projection block. Batch pricing shifted the sync estimate down ~half.
R1_MAZE_COUNT = 50

_DEFAULT_MANIFEST = "gridworld/fixtures/manifest.r1_smoke_batch.json"
_DEFAULT_RUN_CONFIG = "gridworld/fixtures/run_config.r1_smoke_batch.json"
_REPO_ROOT = Path(__file__).resolve().parents[1]

AgentFactory = Callable[[str, dict], tuple[Any, str]]


class _BudgetStop(Exception):
    """Raised before a batch round when the running spend estimate exceeds the cap."""


# --------------------------------------------------------------------------- #
# Spend + metering
# --------------------------------------------------------------------------- #
def _tier_cost(provider: str, tier: str, in_tok: int, out_tok: int) -> float:
    pin, pout = PRICES[provider][tier]
    return in_tok / 1_000_000 * pin + out_tok / 1_000_000 * pout


class _Spend:
    """Global running spend across every model + tier this driver has issued."""

    def __init__(self) -> None:
        self.usd = 0.0

    def add(self, provider: str, tier: str, in_tok: int, out_tok: int) -> float:
        cost = _tier_cost(provider, tier, in_tok, out_tok)
        self.usd += cost
        return cost


class _MeteringBatchAgent:
    """Wraps a batch agent to time each round, tally tokens, and gate on budget.

    Before every ``generate_batch`` it checks the *global* running spend and
    raises ``_BudgetStop`` if the cap is already exceeded — that is the
    "hard-stop new rounds" guard. After the round it records latency and folds
    the round's usage into both this model's totals and the shared ``_Spend``.
    """

    def __init__(
        self,
        inner: Any,
        *,
        provider: str,
        spend: _Spend,
        max_usd: float,
    ) -> None:
        self._inner = inner
        self._provider = provider
        self._spend = spend
        self._max_usd = max_usd
        self.input_tokens = 0
        self.output_tokens = 0
        self.round_latencies_s: list[float] = []

    def generate_batch(self, batch: list[list[dict]]) -> list[Any]:
        if self._spend.usd > self._max_usd:
            raise _BudgetStop(
                f"estimated spend ${self._spend.usd:.2f} exceeds --max-usd "
                f"${self._max_usd:.2f}; not submitting a new round"
            )
        started = time.perf_counter()
        replies = self._inner.generate_batch(batch)
        self.round_latencies_s.append(time.perf_counter() - started)
        round_in = round_out = 0
        for reply in replies:
            usage = getattr(reply, "usage", None) or {}
            round_in += int(usage.get("input_tokens", 0) or 0)
            round_out += int(usage.get("output_tokens", 0) or 0)
        self.input_tokens += round_in
        self.output_tokens += round_out
        # Fold ONLY this round's usage into the shared running spend (adding the
        # cumulative totals every round would multiply-count the budget).
        self._spend.add(self._provider, "batch", round_in, round_out)
        return replies


def _percentiles(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"median": None, "p90": None, "max": None}
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "p90": ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))],
        "max": ordered[-1],
    }


# --------------------------------------------------------------------------- #
# Plan + stepper construction (mirrors the lockstep worker's _build_unit)
# --------------------------------------------------------------------------- #
def _build_plan(
    *,
    run_config: Path,
    manifest: Path,
    artifacts_root: Path,
    scorer_config,
    difficulty_max: float,
):
    from scripts.distributed_run_pipeline import prepare_job

    return prepare_job(
        run_config_path=run_config,
        manifest_path=manifest,
        seeds=[0],
        conditions=None,
        artifacts_root=artifacts_root,
        run_set_id="r1_smoke_batch",
        scorer_config=scorer_config,
        difficulty_max_static_score=difficulty_max,
    )


def _prep_and_stepper(store, plan, unit_raw, artifacts_root):
    """Full assign payload -> prepared run dir -> a fresh EpisodeStepper."""
    from interface.config import ExperimentConfig
    from interface.episode_step import EpisodeStepper
    from pipeline.run_stage3 import build_episode_runner
    from scorer.config import ScorerConfig
    from scripts import distributed_run_pipeline as drp

    unit = store._unit_payload(plan, unit_raw)
    row = drp.materialize_worker_inputs(unit, artifacts_root)
    prep = drp.pipeline._prepare_unit_run(
        row,
        unit["model_id"],
        model_config=unit["model_config"],
        manifest_path=artifacts_root / "distributed" / "worker_manifest.json",
        artifacts_root=artifacts_root,
        scored_static=unit["task_artifacts"]["scored_static"],
        difficulty_max=float(unit["difficulty_max_static_score"]),
        config=ScorerConfig.from_dict(unit["scorer_config"]),
        seed=int(unit["seed"]),
        prompt_variant=str(unit["prompt_variant"]),
        experiment_config=ExperimentConfig.from_dict(unit["experiment_config"]),
        conditions=unit.get("condition_set"),
    )
    prep.run_dir.mkdir(parents=True, exist_ok=True)
    runner = build_episode_runner(
        prep.source,
        prep.experiment_config,
        int(unit["seed"]),
        max_steps=prep.runtime_spec.max_steps,
    )
    stepper = EpisodeStepper(runner, maze_path=str(prep.source))
    return unit, prep, stepper


# --------------------------------------------------------------------------- #
# Per-episode metrics from an episode result transcript
# --------------------------------------------------------------------------- #
def _episode_query_stats(episode: dict) -> dict[str, Any]:
    """Queries, per-query output tokens, truncations and total usage for one run."""
    out_tokens: list[int] = []
    in_total = 0
    out_total = 0
    truncated = 0
    for record in episode.get("transcript", []):
        if record.get("kind") != "query":
            continue
        usage = record.get("usage") or {}
        out = int(usage.get("output_tokens", 0) or 0)
        in_total += int(usage.get("input_tokens", 0) or 0)
        out_total += out
        out_tokens.append(out)
        if record.get("token_truncated"):
            truncated += 1
    return {
        "query_count": int(episode.get("query_count", len(out_tokens))),
        "output_tokens_per_query": out_tokens,
        "input_tokens_total": in_total,
        "output_tokens_total": out_total,
        "token_truncated_count": truncated,
        "success": bool(episode.get("success", False)),
    }


def _aggregate(
    *,
    provider: str,
    episodes: list[dict],
    round_latencies_s: list[float],
    control_episodes: list[dict],
) -> dict[str, Any]:
    per_query_out: list[int] = []
    queries_per_episode: list[int] = []
    truncated_total = 0
    batch_in = batch_out = 0
    for ep in episodes:
        stats = _episode_query_stats(ep)
        per_query_out.extend(stats["output_tokens_per_query"])
        queries_per_episode.append(stats["query_count"])
        truncated_total += stats["token_truncated_count"]
        batch_in += stats["input_tokens_total"]
        batch_out += stats["output_tokens_total"]

    ctrl_in = ctrl_out = 0
    ctrl_queries: list[int] = []
    for ep in control_episodes:
        stats = _episode_query_stats(ep)
        ctrl_in += stats["input_tokens_total"]
        ctrl_out += stats["output_tokens_total"]
        ctrl_queries.append(stats["query_count"])

    batch_usd = _tier_cost(provider, "batch", batch_in, batch_out)
    sync_equiv_usd = _tier_cost(provider, "sync", batch_in, batch_out)
    ctrl_sync_usd = _tier_cost(provider, "sync", ctrl_in, ctrl_out)

    mean_queries = statistics.mean(queries_per_episode) if queries_per_episode else 0.0
    mean_out_per_query = statistics.mean(per_query_out) if per_query_out else 0.0
    # Project a 50-maze R1 run from the measured per-episode averages at batch
    # prices. Per-episode cost = (mean queries) x (mean output/query) at the
    # batch output price + measured mean input cost per episode.
    n_ep = max(len(episodes), 1)
    projected_batch = batch_usd / n_ep * R1_MAZE_COUNT
    projected_sync = sync_equiv_usd / n_ep * R1_MAZE_COUNT

    return {
        "episodes_run": len(episodes),
        "queries_per_episode": {
            "values": queries_per_episode,
            "mean": mean_queries,
        },
        "output_tokens_per_query": {
            "count": len(per_query_out),
            "mean": mean_out_per_query,
            **_percentiles([float(v) for v in per_query_out]),
        },
        "token_truncated_count": truncated_total,
        "round_latency_s": {
            "rounds": len(round_latencies_s),
            **{
                k: v
                for k, v in _percentiles(round_latencies_s).items()
            },
            "min": min(round_latencies_s) if round_latencies_s else None,
        },
        "cost_usd": {
            "batch": {
                "input_tokens": batch_in,
                "output_tokens": batch_out,
                "usd": batch_usd,
            },
            "sync_equivalent_same_tokens": {"usd": sync_equiv_usd},
            "control_sync": {
                "episodes": len(control_episodes),
                "input_tokens": ctrl_in,
                "output_tokens": ctrl_out,
                "usd": ctrl_sync_usd,
                "queries_per_episode": ctrl_queries,
            },
            "batch_vs_sync_savings_usd": sync_equiv_usd - batch_usd,
        },
        "projected_r1": {
            "maze_count": R1_MAZE_COUNT,
            "seeds": 1,
            "batch_usd": projected_batch,
            "sync_usd": projected_sync,
            "basis": "per-episode averages from this smoke x 50 mazes",
        },
    }


# --------------------------------------------------------------------------- #
# Running one model (batch working set + serial control)
# --------------------------------------------------------------------------- #
def _run_model(
    *,
    model_key: str,
    unit_rows: list[dict],
    store,
    plan: dict,
    artifacts_root: Path,
    max_batches: int,
    control_episodes: int,
    round_deadline_s: float,
    agent_factory: AgentFactory,
    spend: _Spend,
    max_usd: float,
) -> dict[str, Any]:
    from interface.batch_runner import LockstepBatchRunner, LockstepUnit
    from interface.episode_log import flush_episode_log
    from scripts import distributed_run_pipeline as drp

    # Runtime model params (enable_thinking, effort, temperature, timeout, ...)
    # live on the UNIT's model_config — NOT plan["models"][model_key], which
    # prepare_job reduces to a routing/topology view (model_group, worker_count,
    # hardware_profile) with the runtime params stripped. The real lockstep
    # worker builds its agent from unit["model_config"]
    # (distributed_run_pipeline.py::run_lockstep_worker), so the smoke must too;
    # otherwise the batch leg silently runs thinking-OFF (enable_thinking=False)
    # while the sync control (also unit["model_config"]) runs thinking-ON.
    # Copy so the round-deadline injection never mutates shared state. The batch
    # clients read their round deadline from the CONFIG (batch_deadline_s), not a
    # generate_batch kwarg, so --round-deadline-s only reaches the real
    # Claude/Kimi agents through this field (it stays out of the unit hash — see
    # _NON_RUNTIME_MODEL_KEYS).
    model_cfg = dict(store._unit_payload(plan, unit_rows[0])["model_config"])
    model_cfg["batch_deadline_s"] = round_deadline_s
    provider = _provider_of(model_cfg)

    # Build a stepper per maze; keep prep so we can flush episode.json.
    preps: dict[str, Any] = {}
    lockstep_units: list[LockstepUnit] = []
    for unit_raw in unit_rows:
        _unit, prep, stepper = _prep_and_stepper(store, plan, unit_raw, artifacts_root)
        preps[unit_raw["unit_id"]] = prep
        lockstep_units.append(LockstepUnit(unit_raw["unit_id"], stepper))

    inner_agent, _ = agent_factory(model_key, model_cfg)
    metered = _MeteringBatchAgent(
        inner_agent, provider=provider, spend=spend, max_usd=max_usd
    )
    runner = LockstepBatchRunner(
        metered, max_batches=max_batches, round_deadline_s=round_deadline_s
    )
    for lsu in lockstep_units:
        runner.add(lsu)

    budget_stopped = False
    try:
        results = runner.run()
    except _BudgetStop as exc:
        budget_stopped = True
        print(f"  [{model_key}] BUDGET STOP: {exc}", file=sys.stderr)
        results = dict(runner._results)  # partial results collected so far

    episodes: list[dict] = []
    for unit_id, result in results.items():
        if isinstance(result, dict) and "error" in result:
            print(f"  [{model_key}] unit {unit_id} errored: {result['error']}",
                  file=sys.stderr)
            continue
        flush_episode_log(result, preps[unit_id].run_dir)
        preps[unit_id].write_run_inputs(extras={"pricing_tier": "batch"})
        episodes.append(result)

    # Unbatched control: the first N mazes through the serial (sync) runner.
    control_results: list[dict] = []
    for unit_raw in unit_rows[:control_episodes]:
        if spend.usd > max_usd:
            print(f"  [{model_key}] budget reached; skipping control episode",
                  file=sys.stderr)
            break
        unit = store._unit_payload(plan, unit_raw)
        drp.run_assigned_unit(
            unit, artifacts_root=artifacts_root, agent_factory=agent_factory, force=True
        )
        # run_assigned_unit uses the SYNC serial runner; re-read the episode it
        # wrote and fold its real usage into the global spend at sync prices.
        prep = preps[unit_raw["unit_id"]]
        # The control run dir differs only if variant/model differ; it is the
        # same run_dir, so read episode.json back.
        episode = json.loads((prep.episode_path).read_text(encoding="utf-8"))
        stats = _episode_query_stats(episode)
        spend.add(provider, "sync", stats["input_tokens_total"], stats["output_tokens_total"])
        control_results.append(episode)

    report = _aggregate(
        provider=provider,
        episodes=episodes,
        round_latencies_s=metered.round_latencies_s,
        control_episodes=control_results,
    )
    report["model_key"] = model_key
    report["provider"] = provider
    report["budget_stopped"] = budget_stopped
    return report


def _provider_of(model_cfg: dict) -> str:
    provider = (model_cfg.get("provider") or "").lower()
    if provider not in PRICES:
        raise ValueError(
            f"No smoke price table for provider {provider!r}; known: {sorted(PRICES)}"
        )
    return provider


# --------------------------------------------------------------------------- #
# Plan printing (dry-run) + orchestration
# --------------------------------------------------------------------------- #
def _units_by_model(plan: dict, models: list[str]) -> dict[str, list[dict]]:
    by_model: dict[str, list[dict]] = {m: [] for m in models}
    for unit in plan["units"]:
        if unit["model_key"] in by_model:
            by_model[unit["model_key"]].append(unit)
    return by_model


def _print_plan(by_model: dict[str, list[dict]], *, max_batches: int, control: int,
                max_usd: float, dry_run: bool) -> None:
    header = "DRY RUN — no agent will be called" if dry_run else "LIVE RUN"
    print(f"== r1 batch smoke plan ({header}) ==")
    print(f"max_batches={max_batches} control_episodes={control} max_usd=${max_usd:.2f}")
    for model_key, units in by_model.items():
        print(f"\nmodel: {model_key}  ({len(units)} mazes)")
        for unit in units:
            print(f"  - {unit['task_id']}  seed={unit['seed']}  "
                  f"variant={unit['prompt_variant']}  unit_id={unit['unit_id']}")


def run_smoke(
    *,
    models: list[str],
    manifest: Path,
    run_config: Path,
    artifacts_root: Path,
    max_batches: int,
    control_episodes: int,
    max_usd: float,
    round_deadline_s: float,
    difficulty_max: float,
    dry_run: bool,
    agent_factory: Optional[AgentFactory],
) -> Optional[dict[str, Any]]:
    from scorer.config import load_scorer_config
    from scripts.distributed_run_pipeline import CoordinatorStore

    artifacts_root.mkdir(parents=True, exist_ok=True)
    scorer_config = load_scorer_config()
    plan = _build_plan(
        run_config=run_config,
        manifest=manifest,
        artifacts_root=artifacts_root,
        scorer_config=scorer_config,
        difficulty_max=difficulty_max,
    )
    unknown = [m for m in models if m not in plan["models"]]
    if unknown:
        raise SystemExit(
            f"--models {unknown} not in run-config {sorted(plan['models'])}"
        )
    by_model = _units_by_model(plan, models)
    _print_plan(by_model, max_batches=max_batches, control=control_episodes,
                max_usd=max_usd, dry_run=dry_run)

    if dry_run:
        # Build the first-round request for every maze to prove requests assemble
        # locally (no agent), then stop — spends nothing.
        store = CoordinatorStore(artifacts_root)
        built = 0
        for model_key, unit_rows in by_model.items():
            for unit_raw in unit_rows:
                _unit, _prep, stepper = _prep_and_stepper(
                    store, plan, unit_raw, artifacts_root
                )
                stepper.start()
                messages = stepper.next_query()
                assert messages is not None, unit_raw["task_id"]
                built += 1
        print(f"\nbuilt {built} first-round batch requests; no agent called.")
        return None

    store = CoordinatorStore(artifacts_root)
    factory = agent_factory or _default_agent_factory()
    spend = _Spend()
    per_model: dict[str, Any] = {}
    for model_key, unit_rows in by_model.items():
        print(f"\n== running {model_key} ({len(unit_rows)} mazes) ==")
        per_model[model_key] = _run_model(
            model_key=model_key,
            unit_rows=unit_rows,
            store=store,
            plan=plan,
            artifacts_root=artifacts_root,
            max_batches=max_batches,
            control_episodes=control_episodes,
            round_deadline_s=round_deadline_s,
            agent_factory=factory,
            spend=spend,
            max_usd=max_usd,
        )

    report = {
        "manifest": str(manifest),
        "run_config": str(run_config),
        "max_batches": max_batches,
        "control_episodes": control_episodes,
        "max_usd": max_usd,
        "total_estimated_spend_usd": spend.usd,
        "prices_per_mtok_usd": PRICES,
        "models": per_model,
    }
    report_path = artifacts_root / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {report_path}  (estimated total spend ${spend.usd:.2f})")
    return report


def _default_agent_factory() -> AgentFactory:
    from scripts.run_pipeline import _build_agent_from_spec

    return _build_agent_from_spec


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        default="claude_opus,kimi_k26",
        help="Comma-separated run-config model keys to smoke.",
    )
    p.add_argument("--manifest", default=_DEFAULT_MANIFEST)
    p.add_argument("--run-config", default=_DEFAULT_RUN_CONFIG)
    p.add_argument("--max-batches", type=int, default=5)
    p.add_argument("--control-episodes", type=int, default=1)
    p.add_argument("--artifacts-root", default=".runs/r1_smoke_batch")
    p.add_argument("--max-usd", type=float, default=25.0)
    p.add_argument(
        "--difficulty-max-static-score",
        type=float,
        default=2000.0,
        # Only feeds the runtime composite in run_score.json, which the smoke
        # report ignores (it measures tokens/latency/cost). Must stay >= the
        # largest static score in the set (the 14x14 smoke maze is ~1133); raise
        # it if harder mazes are added.
        help="Suite difficulty max for runtime scoring (>= largest static score; smoke ~1133).",
    )
    p.add_argument(
        "--round-deadline-s",
        type=float,
        default=7200.0,
        help="Per-round batch deadline passed to the lockstep runner.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute() and not path.exists():
        candidate = _REPO_ROOT / path_str
        if candidate.exists():
            return candidate
    return path


def main(argv: Optional[list[str]] = None, *, agent_factory: Optional[AgentFactory] = None) -> int:
    args = _parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if not args.dry_run and os.environ.get("SMOKE_BUDGET_ACK") != "1":
        print(
            "REFUSING: this run spends real money. Set SMOKE_BUDGET_ACK=1 to "
            "acknowledge, or pass --dry-run to build+print the plan for free.",
            file=sys.stderr,
        )
        return 2

    run_smoke(
        models=models,
        manifest=_resolve(args.manifest),
        run_config=_resolve(args.run_config),
        artifacts_root=Path(args.artifacts_root),
        max_batches=args.max_batches,
        control_episodes=args.control_episodes,
        max_usd=args.max_usd,
        round_deadline_s=args.round_deadline_s,
        difficulty_max=args.difficulty_max_static_score,
        dry_run=args.dry_run,
        agent_factory=agent_factory,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
