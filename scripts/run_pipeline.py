"""Bare-bones run-pipeline orchestrator for MultiNet v2.0 (tests 1-3).

Sequential, inspectable Stage 1->5 driver. No DAG runner. Runs the chosen
experiment over the manifest, writing the ``artifacts/`` tree:

    artifacts/
      tasks/<task_id>/{canonical_paths.json, scored_static.json}
      tasks/_suite.json
      runs/<task_id>/<backend>/<agent>/seed_<seed>/<condition>/{episode.json, run_score.json}
      episode_runs.jsonl
      reports/<run_set_id>/{scoring_calibration_summary,complexity_distance_summary,mechanism_ordering_pairs}.json

Stage 3 uses the ``interface/`` runner (Stack A) with a live-model agent; the CLI
builds the agent from ``--agent``. Programmatic callers can pass any agent
callable to :func:`run_pipeline` (e.g. a stub for testing).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from interface.config import ExperimentConfig
from prompting_experiments import iter_condition_configs
from scorer import compute_runtime_score, load_scorer_config, score_task_file
from scorer.config import ScorerConfig

from pipeline import episode_metrics, reports
from pipeline.run_stage3 import run_episode

Agent = Callable[[list[dict]], str]
_REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def load_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = data["tasks"] if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError("Manifest must be a list of task rows or {'tasks': [...]}.")
    return rows


def _resolve_source(row: dict[str, Any], manifest_path: Path) -> Path:
    source = Path(row["source"])
    if source.is_absolute() and source.exists():
        return source
    for base in (manifest_path.parent, _REPO_ROOT):
        candidate = (base / source).resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Task source not found for {row.get('task_id')}: {row['source']}")


def _condition_configs(conditions: Optional[str]) -> list[tuple[str, ExperimentConfig]]:
    if not conditions:
        return [("default", ExperimentConfig())]
    return list(iter_condition_configs(conditions, ExperimentConfig()))


# --------------------------------------------------------------------------- #
# Stage 2 — static solve & score
# --------------------------------------------------------------------------- #
def score_tasks(
    rows: list[dict[str, Any]],
    manifest_path: Path,
    artifacts_root: Path,
    config: ScorerConfig,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run Stage 2 over every task; return ``task_id -> scored_static dict``."""
    static_by_task: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = row["task_id"]
        source = _resolve_source(row, manifest_path)
        out_dir = artifacts_root / "tasks" / task_id
        scored_path = out_dir / "scored_static.json"
        if scored_path.exists() and not force:
            static_by_task[task_id] = json.loads(scored_path.read_text(encoding="utf-8"))
            continue
        _, static_score = score_task_file(source, output_dir=out_dir, config=config)
        static_by_task[task_id] = static_score.to_dict()
    return static_by_task


def _suite_max(static_by_task: dict[str, dict[str, Any]]) -> float:
    scores = [float(s.get("static_score", 0.0)) for s in static_by_task.values()]
    return max(scores) if scores else 1.0


# --------------------------------------------------------------------------- #
# Stages 3-4 — runs + runtime score
# --------------------------------------------------------------------------- #
def _run_dir(artifacts_root: Path, task_id: str, agent_name: str, seed: int, condition: str) -> Path:
    return artifacts_root / "runs" / task_id / "minigrid" / agent_name / f"seed_{seed}" / condition


def run_pipeline(
    *,
    manifest_path: str | Path,
    experiment: str,
    agent: Agent,
    agent_name: str,
    seeds: Iterable[int] = (0,),
    conditions: Optional[str] = None,
    artifacts_root: str | Path = "artifacts",
    run_set_id: str = "default",
    scorer_config: Optional[ScorerConfig] = None,
    force: bool = False,
) -> dict[str, Any]:
    """Execute Stage 1->5 for one experiment and return the report payloads."""
    manifest_path = Path(manifest_path)
    artifacts_root = Path(artifacts_root)
    config = scorer_config or load_scorer_config()

    all_rows = load_manifest(manifest_path)
    rows = all_rows if experiment == "all" else [r for r in all_rows if r.get("experiment") == experiment]
    if not rows:
        raise ValueError(f"No manifest rows for experiment={experiment!r}.")

    # Stage 2 (score the full suite so difficulty_max is stable across experiments).
    static_by_task = score_tasks(all_rows, manifest_path, artifacts_root, config, force=force)
    difficulty_max = _suite_max(static_by_task)
    suite_path = artifacts_root / "tasks" / "_suite.json"
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(
        json.dumps(
            {
                "difficulty_max_static_score": difficulty_max,
                "tasks": {t: s.get("static_score") for t, s in static_by_task.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    condition_configs = _condition_configs(conditions)

    run_rows: list[dict[str, Any]] = []
    composites: dict[tuple, float] = {}

    for row in rows:
        task_id = row["task_id"]
        source = _resolve_source(row, manifest_path)
        canonical_path = artifacts_root / "tasks" / task_id / "canonical_paths.json"
        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
        scored_static = static_by_task[task_id]

        for seed in seeds:
            for condition, cfg in condition_configs:
                run_dir = _run_dir(artifacts_root, task_id, agent_name, seed, condition)
                run_score_path = run_dir / "run_score.json"
                episode_path = run_dir / "episode.json"

                # Bind the manifest condition (overrides the prompt-variant name when
                # the manifest already encodes an experimental condition).
                manifest_row = dict(row)
                manifest_row.setdefault("condition", condition)

                if run_score_path.exists() and episode_path.exists() and not force:
                    episode = json.loads(episode_path.read_text(encoding="utf-8"))
                    run_score = json.loads(run_score_path.read_text(encoding="utf-8"))
                else:
                    episode = run_episode(source, cfg, agent, seed, run_dir)
                    enriched = episode_metrics.enrich_run_for_scoring(
                        episode, manifest_row, agent_or_model=agent_name, seed=seed
                    )
                    score = compute_runtime_score(
                        enriched,
                        static_score=scored_static,
                        canonical_paths=canonical,
                        config=config,
                        difficulty_max_static_score=difficulty_max,
                    )
                    run_score = score.to_dict()
                    run_score_path.write_text(json.dumps(run_score, indent=2), encoding="utf-8")

                raw_ref = str(episode_path.relative_to(artifacts_root))
                run_row = episode_metrics.build_run_row(
                    episode,
                    canonical,
                    manifest_row,
                    agent_or_model=agent_name,
                    seed=seed,
                    raw_output_ref=raw_ref,
                )
                run_rows.append(run_row)
                composites[(task_id, agent_name, seed, manifest_row["condition"])] = run_score.get("composite")

    # Stage 3 artifact: episode_runs.jsonl (one row per run).
    jsonl_path = artifacts_root / "episode_runs.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for run_row in run_rows:
            handle.write(json.dumps(run_row) + "\n")

    # Stage 5: reports.
    report_dir = artifacts_root / "reports" / run_set_id
    report_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "scoring_calibration_summary": reports.scoring_calibration_summary(
            run_rows, composites, static_by_task
        ),
        "complexity_distance_summary": reports.complexity_distance_summary(run_rows),
        "mechanism_ordering_pairs": reports.mechanism_ordering_pairs(run_rows, all_rows),
    }
    for name, payload in payloads.items():
        (report_dir / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payloads


# --------------------------------------------------------------------------- #
# Agent construction (CLI only)
# --------------------------------------------------------------------------- #
def _build_agent(agent: str) -> tuple[Agent, str]:
    if agent == "claude":
        from interface.agents import ClaudeAnthropicAgent

        instance = ClaudeAnthropicAgent()
        return instance, instance.config.model
    if agent == "qwen":
        from interface.agents import Qwen35VLAgent

        instance = Qwen35VLAgent()
        return instance, instance.config.model
    raise ValueError(f"Unknown agent {agent!r} (expected 'claude' or 'qwen').")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="MultiNet v2.0 bare-bones run pipeline (tests 1-3).")
    parser.add_argument("--experiment", choices=["test1", "test2", "test3", "all"], default="all")
    parser.add_argument(
        "--manifest", default=str(_REPO_ROOT / "gridworld" / "fixtures" / "manifest.json")
    )
    parser.add_argument("--agent", choices=["claude", "qwen"], default="claude")
    parser.add_argument("--conditions", default=None, help="Prompt condition-set name (optional).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--artifacts-root", default=str(_REPO_ROOT / "artifacts"))
    parser.add_argument("--run-set-id", default="default")
    parser.add_argument("--force", action="store_true", help="Recompute existing artifacts.")
    args = parser.parse_args(argv)

    agent, agent_name = _build_agent(args.agent)
    payloads = run_pipeline(
        manifest_path=args.manifest,
        experiment=args.experiment,
        agent=agent,
        agent_name=agent_name,
        seeds=args.seeds,
        conditions=args.conditions,
        artifacts_root=args.artifacts_root,
        run_set_id=args.run_set_id,
        force=args.force,
    )
    summary = payloads["scoring_calibration_summary"]
    print(
        f"Pipeline complete: {summary['run_count']} runs over {summary['task_count']} tasks "
        f"-> {args.artifacts_root}/reports/{args.run_set_id}/"
    )


if __name__ == "__main__":
    main()
