"""Tracking + ETA state for the sequential supervised sweep. Pure logic; the
`/loop` and `sweep_run.sh` read/write the JSON this module manages."""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any

# SWEEP_TOPO selects the run-config family:
#   api     -> Kimi+Claude only (0 GPU workers, no A100 hunt)
#   qwen    -> Qwen-only (3 A100 workers, no API cost) for the Qwen throughput pass
#   kimictx -> Kimi-only 3-worker fleet; the one-off Context-window summary
#              comparison (last3 vs text_summary vs text_summary_and_last3),
#              thinking OFF. A distinct 3-batch family (see below), NOT the
#              10-config sweep.
#   else    -> full Qwen+Kimi+Claude
_TOPO = os.environ.get("SWEEP_TOPO", "").lower()
_API_ONLY = _TOPO == "api"
_QWEN_ONLY = _TOPO == "qwen"
_KIMI_CTX = _TOPO == "kimictx"
if _API_ONLY:
    _CFG = "gridworld/fixtures/run_config.{}_claude_kimi.json"
    _SMOKE_CFG = "gridworld/fixtures/run_config.smoke_kimi_claude.json"
elif _QWEN_ONLY:
    _CFG = "gridworld/fixtures/run_config.{}_qwen.json"
    _SMOKE_CFG = "gridworld/fixtures/run_config.smoke_qwen36.json"
elif _KIMI_CTX:
    _CFG = "gridworld/fixtures/run_config.conditional_context_window_kimi.json"
    _SMOKE_CFG = "gridworld/fixtures/run_config.smoke_kimi3.json"
else:
    _CFG = "gridworld/fixtures/run_config.{}_claude_kimi_qwen.json"
    _SMOKE_CFG = "gridworld/fixtures/run_config.smoke_qwen36_kimi_claude.json"
_MANIFEST = "gridworld/fixtures/manifest.conditional_eval.json"

# n, name, run_config, manifest, conditions, prompt_variant, artifacts_root, run_id, weight
#
# SWEEP_TOPO=kimictx: the one-off Kimi-only Context-window comparison. Batch 0 is
# the 3-Kimi-worker smoke; batches 1-3 are the three context_window prompt
# variants over the 15-maze conditional set (run via `run-massive 1 2 3`).
_KIMI_CTX_BATCHES: list[dict[str, Any]] = [
    {"n": 0, "name": "smoke",
     "run_config": _SMOKE_CFG,
     "manifest": "gridworld/fixtures/manifest.smoke_eval.json",
     "conditions": None, "prompt_variant": None,
     "artifacts_root": "artifacts/smoke", "run_id": "smoke", "weight": 0.1},
    {"n": 1, "name": "ctx_last3", "run_config": _CFG,
     "manifest": _MANIFEST, "conditions": "Context window", "prompt_variant": "last3",
     "artifacts_root": "artifacts/cond/ctx_last3", "run_id": "cond_ctx_last3", "weight": 1.0},
    {"n": 2, "name": "ctx_text_summary", "run_config": _CFG,
     "manifest": _MANIFEST, "conditions": "Context window", "prompt_variant": "text_summary",
     "artifacts_root": "artifacts/cond/ctx_text_summary", "run_id": "cond_ctx_text_summary", "weight": 1.0},
    {"n": 3, "name": "ctx_text_summary_and_last3", "run_config": _CFG,
     "manifest": _MANIFEST, "conditions": "Context window", "prompt_variant": "text_summary_and_last3",
     "artifacts_root": "artifacts/cond/ctx_text_summary_and_last3",
     "run_id": "cond_ctx_text_summary_and_last3", "weight": 1.0},
]

_FULL_BATCHES: list[dict[str, Any]] = [
    {"n": 0, "name": "smoke",
     "run_config": _SMOKE_CFG,
     "manifest": "gridworld/fixtures/manifest.smoke_eval.json",
     "conditions": None, "prompt_variant": None,
     "artifacts_root": "artifacts/smoke", "run_id": "smoke", "weight": 0.1},
    {"n": 1, "name": "prompt", "run_config": _CFG.format("conditional_prompt"),
     "manifest": _MANIFEST, "conditions": "Prompt", "prompt_variant": None,
     "artifacts_root": "artifacts/cond/prompt", "run_id": "cond_prompt", "weight": 3.0},
    {"n": 2, "name": "obs_image_only", "run_config": _CFG.format("conditional_observation_format"),
     "manifest": _MANIFEST, "conditions": "Observation format", "prompt_variant": "image_only",
     "artifacts_root": "artifacts/cond/obs_image_only", "run_id": "cond_obs_image_only", "weight": 1.0},
    {"n": 3, "name": "ctx_current", "run_config": _CFG.format("conditional_context_window"),
     "manifest": _MANIFEST, "conditions": "Context window", "prompt_variant": "current",
     "artifacts_root": "artifacts/cond/ctx_current", "run_id": "cond_ctx_current", "weight": 1.0},
    {"n": 4, "name": "ctx_text_summary", "run_config": _CFG.format("conditional_context_window"),
     "manifest": _MANIFEST, "conditions": "Context window", "prompt_variant": "text_summary",
     "artifacts_root": "artifacts/cond/ctx_text_summary", "run_id": "cond_ctx_text_summary", "weight": 1.0},
    {"n": 5, "name": "act_cardinal", "run_config": _CFG.format("conditional_action_space"),
     "manifest": _MANIFEST, "conditions": "Action space", "prompt_variant": "cardinal",
     "artifacts_root": "artifacts/cond/act_cardinal", "run_id": "cond_act_cardinal", "weight": 1.0},
    {"n": 6, "name": "qry_subgoal", "run_config": _CFG.format("conditional_querying_strategy"),
     "manifest": _MANIFEST, "conditions": "Querying strategy", "prompt_variant": "subgoal",
     "artifacts_root": "artifacts/cond/qry_subgoal", "run_id": "cond_qry_subgoal", "weight": 1.0},
    {"n": 7, "name": "qry_full_trajectory", "run_config": _CFG.format("conditional_querying_strategy"),
     "manifest": _MANIFEST, "conditions": "Querying strategy", "prompt_variant": "full_trajectory",
     "artifacts_root": "artifacts/cond/qry_full_trajectory", "run_id": "cond_qry_full_trajectory", "weight": 1.0},
    {"n": 8, "name": "icl_zero_shot", "run_config": _CFG.format("conditional_in_context_learning"),
     "manifest": _MANIFEST, "conditions": "In-context learning", "prompt_variant": "zero_shot",
     "artifacts_root": "artifacts/cond/icl_zero_shot", "run_id": "cond_icl_zero_shot", "weight": 1.0},
    {"n": 9, "name": "hist_multiturn", "run_config": _CFG.format("conditional_history_mechanism"),
     "manifest": _MANIFEST, "conditions": "History mechanism", "prompt_variant": "multiturn",
     "artifacts_root": "artifacts/cond/hist_multiturn", "run_id": "cond_hist_multiturn", "weight": 1.3},
    {"n": 10, "name": "baseline_thinking", "run_config": _CFG.format("conditional_baseline_thinking"),
     "manifest": _MANIFEST, "conditions": "Prompt", "prompt_variant": "standard",
     "artifacts_root": "artifacts/cond/baseline_thinking", "run_id": "cond_baseline_thinking", "weight": 3.0},
]

BATCHES: list[dict[str, Any]] = _KIMI_CTX_BATCHES if _KIMI_CTX else _FULL_BATCHES

_DEFAULT_ANCHOR_HOURS = 3.0  # coarse; calibrated live after batch 1


def _hours(start: str, end: str) -> float:
    fmt = lambda s: datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    return (fmt(end) - fmt(start)).total_seconds() / 3600.0


def init_state(sweep_id: str, created_at: str) -> dict[str, Any]:
    batches = []
    for b in BATCHES:
        batches.append({**b, "status": "pending", "started_at": None, "ended_at": None,
                        "eta_hours": round(b["weight"] * _DEFAULT_ANCHOR_HOURS, 2),
                        "egress": None, "summaries": {}})
    return {"sweep_id": sweep_id, "created_at": created_at,
            "current_batch": 0, "phase": "provisioning", "batches": batches}


def _batch(state: dict, n: int) -> dict:
    return next(b for b in state["batches"] if b["n"] == n)


def update_batch(state: dict, n: int, **fields: Any) -> dict:
    _batch(state, n).update(fields)
    return state


def recompute_etas(state: dict, anchor_hours_per_weight: float | None = None) -> dict:
    done = [b for b in state["batches"]
            if b["status"] == "complete" and b["started_at"] and b["ended_at"] and b["weight"] > 0]
    if done:
        anchor = sum(_hours(b["started_at"], b["ended_at"]) / b["weight"] for b in done) / len(done)
    else:
        anchor = anchor_hours_per_weight if anchor_hours_per_weight is not None else _DEFAULT_ANCHOR_HOURS
    for b in state["batches"]:
        if b["status"] not in ("complete",):
            b["eta_hours"] = round(b["weight"] * anchor, 2)
    return state


def load_state(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def save_state(path: str | Path, state: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(state, indent=2) + "\n")


def render_table(state: dict) -> str:
    rows = ["| # | run_id | status | eta_h | egress | summaries |",
            "|---|---|---|---:|---|---|"]
    for b in state["batches"]:
        rows.append(f"| {b['n']} | {b['run_id']} | {b['status']} | {b['eta_hours']} | "
                    f"{b['egress'] or '-'} | {len(b['summaries'])}/3 |")
    return "\n".join(rows)
