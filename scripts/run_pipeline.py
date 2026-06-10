"""Bare-bones run-pipeline orchestrator for MultiNet v2.0 (tests 1-3).

Sequential, inspectable Stage 1->5 driver. No DAG runner. Writes the
``artifacts/`` tree:

    artifacts/
      tasks/<task_id>/{canonical_paths.json, scored_static.json}
      tasks/_suite.json
      runs/<task_id>/<backend>/<model>/seed_<seed>/<condition>/{episode.json, run_inputs.json, run_score.json}
      episode_runs.jsonl
      reports/<run_set_id>/{scoring_calibration_summary,complexity_distance_summary,mechanism_ordering_pairs}.json

Selection is data-driven via a **run-config** that maps each model to the task
files it should run (plus its provider/params); the **manifest** is a separate
task *catalog* that supplies per-task scoring metadata (experiment, condition,
expected_mechanisms, test-2 route cells). Stage 3 uses the ``interface/`` runner
(Stack A) with a live-model agent. Programmatic callers can inject any agent
callable, e.g. a stub for testing.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from prompting_experiments import CONDITION_SETS, iter_condition_configs
from scorer import compute_runtime_score, load_scorer_config, score_task_file
from scorer.config import SCORER_VERSION, ScorerConfig
from scorer.io import stable_hash, task_spec_from_payload

from pipeline import episode_metrics, reports

# Bump when Stage-3 run production changes in a way that invalidates cached episodes.
PIPELINE_VERSION = "0.1.0"

Agent = Callable[[list[dict]], str]
# A factory used by tests to supply stub agents: (model_name, model_cfg) -> (agent, label).
AgentFactory = Callable[[str, dict[str, Any]], "tuple[Agent, str]"]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MANIFEST = _REPO_ROOT / "gridworld" / "fixtures" / "manifest.json"
_EXPERIMENT_KEYWORDS = {"test1", "test2", "test3", "all"}


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "model"


# --------------------------------------------------------------------------- #
# Manifest catalog + task resolution
# --------------------------------------------------------------------------- #
def load_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = data["tasks"] if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError("Manifest must be a list of task rows or {'tasks': [...]}.")
    return rows


def _resolve_path(source: str, manifest_path: Path) -> Optional[Path]:
    candidate = Path(source)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    for base in (Path.cwd(), manifest_path.parent, _REPO_ROOT):
        resolved = (base / source).resolve()
        if resolved.exists():
            return resolved
    return None


def _resolve_source(row: dict[str, Any], manifest_path: Path) -> Path:
    resolved = _resolve_path(row["source"], manifest_path)
    if resolved is None:
        raise FileNotFoundError(f"Task source not found for {row.get('task_id')}: {row['source']}")
    return resolved


def _synth_row(path: Path) -> dict[str, Any]:
    """A plain task file with no catalog entry runs as a test-1 nav task."""
    return {
        "task_id": path.stem,
        "experiment": "test1",
        "condition": "default",
        "variant": path.stem,
        "source": str(path),
        "expected_mechanisms": [],
        "notes": "Synthesized (not in manifest catalog).",
    }


def resolve_task_rows(
    entries: Iterable[str],
    catalog: list[dict[str, Any]],
    manifest_path: Path,
) -> list[dict[str, Any]]:
    """Resolve run-config task entries to manifest-style rows (metadata attached).

    Each entry may be an experiment keyword (``test1``/``test2``/``test3``/``all``),
    a catalog ``task_id``, or a path to a task ``.json``. Paths are matched against
    the catalog (by resolved path) so test-2/test-3 metadata is preserved; an
    unmatched path is synthesized as a plain test-1 task. Duplicate task_ids are
    de-duplicated, keeping first occurrence.
    """
    by_id = {r["task_id"]: r for r in catalog}
    by_path: dict[Path, list[dict[str, Any]]] = {}
    for r in catalog:
        resolved = _resolve_path(r["source"], manifest_path)
        if resolved is not None:
            by_path.setdefault(resolved, []).append(r)

    resolved_rows: list[dict[str, Any]] = []
    for entry in entries:
        if entry in _EXPERIMENT_KEYWORDS:
            matches = catalog if entry == "all" else [r for r in catalog if r.get("experiment") == entry]
            if not matches:
                raise ValueError(f"No catalog tasks for experiment {entry!r}.")
            resolved_rows.extend(matches)
            continue
        if entry in by_id:
            resolved_rows.append(by_id[entry])
            continue
        path = _resolve_path(entry, manifest_path)
        if path is not None:
            matches = by_path.get(path)
            resolved_rows.append(matches[0] if matches else _synth_row(path))
            continue
        raise ValueError(
            f"Cannot resolve task entry {entry!r} (not an experiment keyword, catalog task_id, or file path)."
        )

    deduped: dict[str, dict[str, Any]] = {}
    for row in resolved_rows:
        deduped.setdefault(row["task_id"], row)
    return list(deduped.values())


def _condition_configs(conditions: Optional[str]) -> list[tuple[str, ExperimentConfig]]:
    from interface.config import ExperimentConfig

    if not conditions:
        return [("default", ExperimentConfig())]
    if conditions not in CONDITION_SETS:
        raise ValueError(
            f"Unknown --conditions {conditions!r}; available: {sorted(CONDITION_SETS)}."
        )
    return list(iter_condition_configs(conditions, ExperimentConfig()))


# --------------------------------------------------------------------------- #
# Content-hash invalidation
# --------------------------------------------------------------------------- #
def _expected_static_hash(spec, config: ScorerConfig) -> str:
    """Mirror scorer.static's scored_static inputs_hash recipe (task + config)."""
    return stable_hash(
        {"task": spec.to_dict(), "config": config.to_dict(), "scorer_version": SCORER_VERSION}
    )


def _expected_run_hash(spec, model_name: str, seed: int, backend: str) -> str:
    """Hash the inputs that determine a Stage-3 episode.

    Excludes scorer config (that invalidates run_score, not the model call) and,
    pre-v1, the prompt/ExperimentConfig (prompts are not yet versioned while we
    iterate; the prompt variant still separates runs via the <condition> dir).
    TODO(release): fold in backend_version + adapter/model code version so code
    changes invalidate cached episodes at v1.
    """
    return stable_hash(
        {
            "task": spec.to_dict(),
            "model_id": model_name,
            "seed": seed,
            "backend": backend,
            "pipeline_version": PIPELINE_VERSION,
        }
    )


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
    """Run Stage 2 over every task; return ``task_id -> scored_static dict``.

    Hash-aware: a cached ``scored_static.json`` is reused only when its
    ``inputs_hash`` matches the hash recomputed from the current task spec and
    scorer config; otherwise the task bundle (canonical_paths + scored_static)
    is regenerated. ``force`` always regenerates.
    """
    static_by_task: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = row["task_id"]
        source = _resolve_source(row, manifest_path)
        out_dir = artifacts_root / "tasks" / task_id
        scored_path = out_dir / "scored_static.json"
        canonical_path = out_dir / "canonical_paths.json"
        # Stage 3 reads canonical_paths.json unconditionally, so both halves of
        # the task bundle must be present to honor the cache.
        if scored_path.exists() and canonical_path.exists() and not force:
            cached = json.loads(scored_path.read_text(encoding="utf-8"))
            spec = task_spec_from_payload(json.loads(Path(source).read_text(encoding="utf-8")))
            if cached.get("inputs_hash") == _expected_static_hash(spec, config):
                static_by_task[task_id] = cached
                continue
        _, static_score = score_task_file(source, output_dir=out_dir, config=config)
        static_by_task[task_id] = static_score.to_dict()
    return static_by_task


def _score_suite(
    rows: list[dict[str, Any]],
    manifest_path: Path,
    artifacts_root: Path,
    config: ScorerConfig,
    force: bool,
) -> tuple[dict[str, dict[str, Any]], float]:
    static_by_task = score_tasks(rows, manifest_path, artifacts_root, config, force=force)
    scores = [float(s.get("static_score", 0.0)) for s in static_by_task.values()]
    difficulty_max = max(scores) if scores else 1.0
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
    return static_by_task, difficulty_max


# --------------------------------------------------------------------------- #
# Stages 3-4 — runs + runtime score (per model)
# --------------------------------------------------------------------------- #
def _run_dir(artifacts_root: Path, task_id: str, model: str, seed: int, condition: str) -> Path:
    return artifacts_root / "runs" / task_id / "minigrid" / model / f"seed_{seed}" / condition


def _run_one_model(
    rows: list[dict[str, Any]],
    agent: Agent,
    model_name: str,
    *,
    manifest_path: Path,
    artifacts_root: Path,
    static_by_task: dict[str, dict[str, Any]],
    difficulty_max: float,
    config: ScorerConfig,
    seeds: Iterable[int],
    conditions: Optional[str],
    force: bool,
) -> tuple[list[dict[str, Any]], dict[tuple, Optional[float]]]:
    from pipeline.run_stage3 import run_episode

    condition_configs = _condition_configs(conditions)
    run_rows: list[dict[str, Any]] = []
    composites: dict[tuple, Optional[float]] = {}

    for row in rows:
        task_id = row["task_id"]
        scored_static = static_by_task[task_id]
        # Tasks Stage 2 marks unbeatable are ineligible: skip the expensive
        # Stage 3/4 work (model/API calls + scoring) entirely. The reports
        # surface them via scoring_calibration_summary's ineligible_tasks.
        if not scored_static.get("is_beatable", True):
            continue
        source = _resolve_source(row, manifest_path)
        spec = task_spec_from_payload(json.loads(Path(source).read_text(encoding="utf-8")))
        canonical = json.loads(
            (artifacts_root / "tasks" / task_id / "canonical_paths.json").read_text(encoding="utf-8")
        )

        for seed in seeds:
            for variant, cfg in condition_configs:
                run_dir = _run_dir(artifacts_root, task_id, model_name, seed, variant)
                episode_path = run_dir / "episode.json"
                sidecar_path = run_dir / "run_inputs.json"
                run_score_path = run_dir / "run_score.json"

                # ``condition`` is the task-intrinsic axis (test-3 mechanism
                # order, carried by the manifest); ``variant`` is the orthogonal
                # prompt axis from --conditions. Keep them separate so prompt
                # variants do not collapse onto the manifest condition.
                manifest_row = dict(row)

                # Stage 3 (expensive: model calls) is hash-cached. Reuse a cached
                # episode only when its stamped run-inputs hash still matches.
                expected_hash = _expected_run_hash(spec, model_name, seed, "minigrid")
                reuse = (
                    not force
                    and episode_path.exists()
                    and sidecar_path.exists()
                    and json.loads(sidecar_path.read_text(encoding="utf-8")).get("inputs_hash")
                    == expected_hash
                )
                if reuse:
                    episode = json.loads(episode_path.read_text(encoding="utf-8"))
                else:
                    episode = run_episode(source, cfg, agent, seed, run_dir)
                    sidecar_path.write_text(
                        json.dumps(
                            {
                                "inputs_hash": expected_hash,
                                "producer_version": PIPELINE_VERSION,
                                "task_id": task_id,
                                "model_id": model_name,
                                "seed": seed,
                                "backend": "minigrid",
                                "condition": variant,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

                # Derive the test-2/test-3 signals once and share them between the
                # scorer-facing dict and the jsonl row (each call would otherwise
                # re-walk the whole transcript).
                metrics = episode_metrics.build_metrics(episode, canonical, manifest_row)

                # Stage 4 is cheap + deterministic: always (re)score from the
                # episode so scorer-config / static / canonical changes propagate.
                enriched = episode_metrics.enrich_run_for_scoring(
                    episode, manifest_row, agent_or_model=model_name, seed=seed, metrics=metrics
                )
                run_score = compute_runtime_score(
                    enriched,
                    static_score=scored_static,
                    canonical_paths=canonical,
                    config=config,
                    difficulty_max_static_score=difficulty_max,
                ).to_dict()
                run_score_path.write_text(json.dumps(run_score, indent=2), encoding="utf-8")

                run_rows.append(
                    episode_metrics.build_run_row(
                        episode,
                        canonical,
                        manifest_row,
                        agent_or_model=model_name,
                        seed=seed,
                        raw_output_ref=str(episode_path.relative_to(artifacts_root)),
                        metrics=metrics,
                        prompt_variant=variant,
                    )
                )
                composites[
                    (task_id, model_name, seed, manifest_row.get("condition"), variant)
                ] = run_score.get("composite")

    return run_rows, composites


def _write_aggregate(
    run_rows: list[dict[str, Any]],
    composites: dict[tuple, Optional[float]],
    static_by_task: dict[str, dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    artifacts_root: Path,
    run_set_id: str,
) -> dict[str, Any]:
    jsonl_path = artifacts_root / "episode_runs.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for run_row in run_rows:
            handle.write(json.dumps(run_row) + "\n")

    report_dir = artifacts_root / "reports" / run_set_id
    report_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "scoring_calibration_summary": reports.scoring_calibration_summary(
            run_rows, composites, static_by_task
        ),
        "complexity_distance_summary": reports.complexity_distance_summary(run_rows),
        "mechanism_ordering_pairs": reports.mechanism_ordering_pairs(run_rows, metadata_rows),
    }
    for name, payload in payloads.items():
        (report_dir / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Per-model reports: machine-readable, one file per model, kept separate
    # from the scorer-calibration ("tuning") artifacts above.
    models_dir = report_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_reports: dict[str, Any] = {}
    for model_id in sorted({str(r.get("agent_or_model")) for r in run_rows}):
        report = reports.model_report(run_rows, composites, model_id, run_set_id)
        (models_dir / f"{_sanitize(model_id)}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        model_reports[model_id] = report
    payloads["model_reports"] = model_reports
    return payloads


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
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
    """Single-model convenience entry: run one experiment with one agent."""
    manifest_path = Path(manifest_path)
    artifacts_root = Path(artifacts_root)
    config = scorer_config or load_scorer_config()

    catalog = load_manifest(manifest_path)
    rows = resolve_task_rows([experiment], catalog, manifest_path)
    static_by_task, difficulty_max = _score_suite(rows, manifest_path, artifacts_root, config, force)
    run_rows, composites = _run_one_model(
        rows,
        agent,
        _sanitize(agent_name),
        manifest_path=manifest_path,
        artifacts_root=artifacts_root,
        static_by_task=static_by_task,
        difficulty_max=difficulty_max,
        config=config,
        seeds=seeds,
        conditions=conditions,
        force=force,
    )
    return _write_aggregate(run_rows, composites, static_by_task, rows, artifacts_root, run_set_id)


def run_from_config(
    *,
    run_config_path: str | Path,
    manifest_path: str | Path = _DEFAULT_MANIFEST,
    seeds: Iterable[int] = (0,),
    conditions: Optional[str] = None,
    artifacts_root: str | Path = "artifacts",
    run_set_id: str = "default",
    scorer_config: Optional[ScorerConfig] = None,
    force: bool = False,
    agent_factory: Optional[AgentFactory] = None,
) -> dict[str, Any]:
    """Run-config entry: each model runs its own task selection (model -> task files)."""
    manifest_path = Path(manifest_path)
    artifacts_root = Path(artifacts_root)
    config = scorer_config or load_scorer_config()
    factory = agent_factory or _build_agent_from_spec

    run_config = load_run_config(run_config_path)
    catalog = load_manifest(manifest_path)

    # Resolve each model's task rows + build its agent.
    plans: list[tuple[str, Agent, list[dict[str, Any]]]] = []
    union: dict[str, dict[str, Any]] = {}
    for name, model_cfg in run_config["models"].items():
        entries = model_cfg.get("tasks") or model_cfg.get("runs") or []
        if not entries:
            raise ValueError(f"Model {name!r} lists no tasks/runs.")
        rows = resolve_task_rows(entries, catalog, manifest_path)
        agent, label = factory(name, model_cfg)
        plans.append((_sanitize(label), agent, rows))
        for r in rows:
            union.setdefault(r["task_id"], r)

    union_rows = list(union.values())
    static_by_task, difficulty_max = _score_suite(union_rows, manifest_path, artifacts_root, config, force)

    all_run_rows: list[dict[str, Any]] = []
    composites: dict[tuple, Optional[float]] = {}
    for model_name, agent, rows in plans:
        rr, comp = _run_one_model(
            rows,
            agent,
            model_name,
            manifest_path=manifest_path,
            artifacts_root=artifacts_root,
            static_by_task=static_by_task,
            difficulty_max=difficulty_max,
            config=config,
            seeds=seeds,
            conditions=conditions,
            force=force,
        )
        all_run_rows.extend(rr)
        composites.update(comp)

    return _write_aggregate(all_run_rows, composites, static_by_task, union_rows, artifacts_root, run_set_id)


# --------------------------------------------------------------------------- #
# Run-config + agent construction
# --------------------------------------------------------------------------- #
def load_run_config(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "models" not in data or not isinstance(data["models"], dict):
        raise ValueError("Run-config must be an object with a 'models' mapping.")
    return data


def _build_agent_from_spec(name: str, model_cfg: dict[str, Any]) -> tuple[Agent, str]:
    """Construct a live agent from a run-config model entry."""
    provider = (model_cfg.get("provider") or "").lower()
    model = model_cfg.get("model")
    temperature = float(model_cfg.get("temperature", 0.0))
    max_tokens = model_cfg.get("max_tokens")

    if provider == "claude":
        from interface.agents import ClaudeAnthropicAgent, ClaudeAnthropicConfig

        cfg = ClaudeAnthropicConfig(temperature=temperature)
        if model:
            cfg.model = model
        if max_tokens:
            cfg.max_tokens = int(max_tokens)
        return ClaudeAnthropicAgent(config=cfg), model or cfg.model
    if provider == "qwen":
        from interface.agents import Qwen35VLAgent, Qwen35VLConfig

        cfg = Qwen35VLConfig(temperature=temperature)
        if model:
            cfg.model = model
        if max_tokens:
            cfg.max_new_tokens = int(max_tokens)
        for key in (
            "device_map",
            "local_files_only",
            "trust_remote_code",
            "torch_dtype",
            "load_in_4bit",
            "attn_implementation",
            "max_memory",
        ):
            if key in model_cfg:
                setattr(cfg, key, model_cfg[key])
        return Qwen35VLAgent(config=cfg), model or cfg.model
    raise ValueError(f"Model {name!r}: unknown provider {provider!r} (expected 'claude' or 'qwen').")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="MultiNet v2.0 bare-bones run pipeline (tests 1-3).")
    parser.add_argument("--run-config", help="JSON run-config mapping models to task files (preferred).")
    parser.add_argument("--manifest", default=str(_DEFAULT_MANIFEST), help="Task catalog (metadata).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--conditions", default=None, help="Prompt condition-set name (optional).")
    parser.add_argument("--artifacts-root", default=str(_REPO_ROOT / "artifacts"))
    parser.add_argument("--run-set-id", default="default")
    parser.add_argument("--force", action="store_true", help="Recompute existing artifacts.")
    # Single-model fallback (when --run-config is not supplied):
    parser.add_argument("--experiment", choices=["test1", "test2", "test3", "all"], default="all")
    parser.add_argument("--agent", choices=["claude", "qwen"], help="Single-model provider.")
    args = parser.parse_args(argv)

    if args.run_config:
        payloads = run_from_config(
            run_config_path=args.run_config,
            manifest_path=args.manifest,
            seeds=args.seeds,
            conditions=args.conditions,
            artifacts_root=args.artifacts_root,
            run_set_id=args.run_set_id,
            force=args.force,
        )
    else:
        if not args.agent:
            parser.error("provide --run-config, or --agent for a single-model run.")
        agent, label = _build_agent_from_spec(args.agent, {"provider": args.agent})
        payloads = run_pipeline(
            manifest_path=args.manifest,
            experiment=args.experiment,
            agent=agent,
            agent_name=label,
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
