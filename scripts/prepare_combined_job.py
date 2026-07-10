"""Build one MASSIVE distributed job from several batches so the fleet stays saturated.

Each batch (a run-config + conditions + prompt_variant) is prepared independently with the
tested ``prepare_job`` into ``<root>/_parts/<run_id>``; their units are then merged into a
single plan under ``<root>`` (one job_id, LPT-ordered) and the shared task artifacts copied
in. Because each batch uses a distinct prompt_variant, the units' ``run_dir``s don't collide,
so one job can carry all of them — the coordinator then keeps every worker busy instead of
idling through each batch's long-hard-maze tail.

Usage:
  python -m scripts.prepare_combined_job --artifacts-root artifacts/<combined_run_id> \
      --run-set-id <combined_run_id> --manifest <manifest> --difficulty-max-static-score N \
      --batch 2 --batch 3 ... --batch 10        # SWEEP_TOPO selects the batch family
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from scripts import distributed_run_pipeline as dist


def merge_plans(part_plans: list[dict[str, Any]], job_id: str) -> tuple[dict, dict]:
    """Merge per-batch plan dicts into one combined (plan, state). Units are concatenated
    and LPT-ordered; static_by_task is unioned; state is fresh all-pending."""
    units: list[dict[str, Any]] = []
    static_by_task: dict[str, Any] = {}
    tasks: list[dict[str, Any]] = []
    scorer_config = None
    difficulty_max = None
    for plan in part_plans:
        units.extend(plan["units"])
        static_by_task.update(plan.get("static_by_task") or {})
        # ``tasks`` (per-(task,variant) manifest metadata rows) is REQUIRED by
        # finalize_job -> _write_aggregate; dropping it makes the combined job
        # runnable but not finalizable (KeyError: 'tasks'). Each part covers a
        # distinct prompt_variant, so the rows are distinct -> concatenate.
        tasks.extend(plan.get("tasks") or [])
        scorer_config = scorer_config or plan.get("scorer_config")
        difficulty_max = plan.get("difficulty_max_static_score") if difficulty_max is None else difficulty_max
    units = dist._order_units_lpt(units, static_by_task)
    plan = {
        "schema_version": "0.1.0",
        "job_id": job_id,
        "created_at": dist._iso(),
        "pipeline_version": part_plans[0].get("pipeline_version"),
        "scorer_config": scorer_config,
        "difficulty_max_static_score": difficulty_max,
        "static_by_task": static_by_task,
        "tasks": tasks,
        "units": units,
    }
    state = {
        "schema_version": "0.1.0",
        "job_id": job_id,
        "created_at": dist._iso(),
        "updated_at": dist._iso(),
        "workers": {},
        "units": {u["unit_id"]: {"status": "pending", "attempts": 0, "worker_id": None,
                                 "updated_at": dist._iso()} for u in units},
    }
    return plan, state


def build(specs: list[dict[str, Any]], *, artifacts_root: Path, run_set_id: str,
          seeds: list[int], difficulty_max: float | None) -> dict:
    artifacts_root = Path(artifacts_root)
    part_plans = []
    tasks_root = artifacts_root / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        part_root = artifacts_root / "_parts" / spec["run_id"]
        plan = dist.prepare_job(
            run_config_path=spec["run_config"], manifest_path=spec["manifest"],
            seeds=seeds, conditions=spec["conditions"], prompt_variant=spec["prompt_variant"],
            artifacts_root=part_root, run_set_id=spec["run_id"],
            difficulty_max_static_score=difficulty_max,
        )
        part_plans.append(plan)
        # shared task artifacts (same 15 mazes) -> copy once into the combined root.
        # tasks/ holds per-task dirs AND a _suite.json file, so handle both.
        for td in (part_root / "tasks").glob("*"):
            dest = tasks_root / td.name
            if dest.exists():
                continue
            if td.is_dir():
                shutil.copytree(td, dest)
            else:
                shutil.copy2(td, dest)
    digest = dist.stable_hash({"run_set_id": run_set_id, "run_ids": [s["run_id"] for s in specs]})
    job_id = f"job_massive_{digest[:12]}"
    plan, state = merge_plans(part_plans, job_id)
    dist._distributed_root(artifacts_root).mkdir(parents=True, exist_ok=True)
    dist._write_json_atomic(dist.plan_path(artifacts_root), plan)
    dist._write_json_atomic(dist.state_path(artifacts_root), state)
    dist._uploads_root(artifacts_root).mkdir(parents=True, exist_ok=True)
    return plan


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-root", required=True)
    p.add_argument("--run-set-id", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--difficulty-max-static-score", type=float, default=None)
    p.add_argument("--seeds", type=int, nargs="*", default=[0])
    p.add_argument("--batch", type=int, action="append", required=True, dest="batches")
    args = p.parse_args()

    from scripts.sweep_state import BATCHES
    specs = []
    for n in args.batches:
        b = BATCHES[n]
        specs.append({"run_config": b["run_config"], "manifest": args.manifest,
                      "conditions": b.get("conditions"), "prompt_variant": b.get("prompt_variant"),
                      "run_id": b["run_id"]})
    plan = build(specs, artifacts_root=Path(args.artifacts_root), run_set_id=args.run_set_id,
                 seeds=args.seeds, difficulty_max=args.difficulty_max_static_score)
    print(f"Prepared MASSIVE job {plan['job_id']}: {len(plan['units'])} units from "
          f"{len(specs)} batches -> {dist.plan_path(args.artifacts_root)}")


if __name__ == "__main__":
    main()
