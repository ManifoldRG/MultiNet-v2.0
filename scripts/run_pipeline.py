"""Run-pipeline orchestrator for MultiNet v2.0: manifest + run-config driven evaluation runs.

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
import dataclasses
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from prompting_experiments import CONDITION_SETS, iter_condition_configs
from scorer import compute_runtime_score, load_scorer_config, score_task_file
from scorer.config import SCORER_VERSION, ScorerConfig
from scorer.io import stable_hash, task_spec_from_payload

from pipeline import episode_metrics, reports

# Bump when Stage-3 run production changes in a way that invalidates cached episodes.
PIPELINE_VERSION = "0.1.2"
RUNTIME_MAX_STEPS_OPTIMAL_MULTIPLIER = 3

Agent = Callable[[list[dict]], str]
# A factory used by tests to supply stub agents: (model_name, model_cfg) -> (agent, label).
AgentFactory = Callable[[str, dict[str, Any]], "tuple[Agent, str]"]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MANIFEST = _REPO_ROOT / "gridworld" / "fixtures" / "manifest.json"
_EXPERIMENT_KEYWORDS = {"test1", "test2", "test3", "r1", "all"}


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "model"


def _load_json_object_if_valid(path: Path) -> Optional[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        tmp_path.replace(path)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


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

    Each entry may be an experiment keyword (``test1``/``test2``/``test3``/``r1``/``all``),
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


def condition_variant_names(conditions: Optional[str]) -> list[str]:
    if not conditions:
        return ["default"]
    if conditions not in CONDITION_SETS:
        raise ValueError(
            f"Unknown --conditions {conditions!r}; available: {sorted(CONDITION_SETS)}."
        )
    return [
        variant.name
        for variant in CONDITION_SETS[conditions].variants.values()
        if variant.implemented
    ]


def _condition_configs(
    conditions: Optional[str],
    prompt_variant: Optional[str] = None,
    base_overrides: Optional[dict] = None,
) -> list[tuple[str, ExperimentConfig]]:
    from interface.config import ExperimentConfig

    base = ExperimentConfig(**(base_overrides or {}))
    if not conditions:
        if prompt_variant not in (None, "default"):
            raise ValueError("The default condition set only supports prompt_variant='default'.")
        return [("default", base)]
    if conditions not in CONDITION_SETS:
        raise ValueError(
            f"Unknown --conditions {conditions!r}; available: {sorted(CONDITION_SETS)}."
        )
    pairs = list(iter_condition_configs(conditions, base))
    if prompt_variant is not None:
        pairs = [(name, cfg) for name, cfg in pairs if name == prompt_variant]
        if not pairs:
            raise ValueError(
                f"Unknown prompt variant {prompt_variant!r} for --conditions {conditions!r}."
            )
    return pairs


# --------------------------------------------------------------------------- #
# Content-hash invalidation
# --------------------------------------------------------------------------- #
def _expected_static_hash(spec, config: ScorerConfig) -> str:
    """Mirror scorer.static's scored_static inputs_hash recipe (task + config)."""
    return stable_hash(
        {"task": spec.to_dict(), "config": config.to_dict(), "scorer_version": SCORER_VERSION}
    )


# Keys that do not change what the model produces, so they must be excluded from
# the episode cache key: editing them otherwise re-pays for every cached episode.
# Covers orchestration/scheduling knobs and transport/model-loading knobs.
# NOTE: keep output-affecting knobs IN the hash (temperature, max_tokens,
# enable_thinking, torch_dtype, load_in_4bit, attn_implementation, model, dtype,
# quantization).
_NON_RUNTIME_MODEL_KEYS = {
    # orchestration / scheduling
    "tasks",
    "runs",
    "group",
    "worker_count",
    "hardware_profile",
    "worker_tags",
    "max_in_flight",
    # transport / model-loading (do not affect outputs)
    "timeout",
    "max_attempts",
    # batch-API round deadline + cancel grace: wall-clock scheduling knobs on the
    # Claude/Kimi batch clients; they bound how long a round waits, never what the
    # model emits, so they must stay OUT of the episode/unit hash.
    "batch_deadline_s",
    "batch_cancel_grace_s",
    "batch_poll_interval_s",
    "device_map",
    "local_files_only",
    "max_memory",
    "trust_remote_code",
    "max_model_len",
    "gpu_memory_utilization",
    "tensor_parallel_size",
    "enforce_eager",
    "max_num_seqs",
    "enable_prefix_caching",
    "download_dir",
    "use_tqdm",
    "engine_kwargs",
}


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    # Canonicalize numbers so cosmetic spellings of the same value (e.g. 0 vs 0.0,
    # 128 vs 128.0) do not produce different cache keys. bool is handled by the
    # primitive branch below (it is not a float).
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Do NOT silently str() an unknown type into the cache key: a stringified
    # object (e.g. a repr with a memory address) can collide or vary across
    # machines, poisoning the cross-machine episode cache hash. Fail loudly so
    # the recipe is extended deliberately for any new type that must be hashed.
    raise TypeError(
        f"_jsonable cannot canonicalize {type(value).__name__} for the cache hash; "
        "add an explicit branch (or a to_dict) instead of stringifying it."
    )


def _runtime_model_config(model_config: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not model_config:
        return {}
    return {
        str(key): _jsonable(value)
        for key, value in sorted(model_config.items(), key=lambda item: str(item[0]))
        if key not in _NON_RUNTIME_MODEL_KEYS
    }


def _experiment_config_payload(experiment_config: Any | None) -> dict[str, Any]:
    if experiment_config is None:
        return {}
    payload = _jsonable(experiment_config)
    return payload if isinstance(payload, dict) else {"value": payload}


def _expected_run_hash(
    spec,
    model_name: str,
    seed: int,
    backend: str,
    *,
    condition_set: Optional[str] = None,
    prompt_variant: str = "default",
    experiment_config: Any | None = None,
    model_config: Optional[dict[str, Any]] = None,
) -> str:
    """Hash the inputs that determine a Stage-3 episode.

    Excludes scorer config: that invalidates run_score, not the model call.
    TODO(post-release): fold in backend_version + adapter/model code version so code
    changes invalidate cached episodes at v1.
    """
    return stable_hash(
        {
            "task": spec.to_dict(),
            "model_id": model_name,
            "model_config": _runtime_model_config(model_config),
            "seed": seed,
            "backend": backend,
            "condition_set": condition_set,
            "prompt_variant": prompt_variant,
            "experiment_config": _experiment_config_payload(experiment_config),
            "pipeline_version": PIPELINE_VERSION,
        }
    )


def _phase_episode_provenance(
    phase: Optional[dict[str, Any]], model_config: Optional[dict[str, Any]]
) -> dict[str, Any]:
    """Additive phase/caps provenance for episode.json + the run row (Task B3).

    Returns ``{}`` when the run-config declares no top-level ``phase`` block, so
    non-two-tier runs are unchanged and ``build_run_row`` falls back to
    ``pass=1``. When a phase is declared, records the two-tier ``pass``, the
    ``phase_label``, and the output caps (``max_tokens`` / ``max_model_len``) the
    unit ran under, taken from the model config.

    This is *provenance*, not a generation input: it NEVER enters
    ``_expected_run_hash`` / the distributed unit hashes (the cap change already
    differentiates the phases in the hash), so re-labeling a phase never churns
    a cached episode or an already-paid unit.
    """
    if not phase:
        return {}
    provenance: dict[str, Any] = {"pass": int(phase.get("pass", 1))}
    label = phase.get("label")
    if label is not None:
        provenance["phase_label"] = str(label)
    model_config = model_config or {}
    for cap_key in ("max_tokens", "max_model_len"):
        if model_config.get(cap_key) is not None:
            provenance[cap_key] = model_config[cap_key]
    return provenance


def _canonical_optimal_steps(canonical_paths: dict[str, Any]) -> Optional[int]:
    bfs = canonical_paths.get("bfs")
    if isinstance(bfs, dict) and bfs.get("optimal_steps") is not None:
        return int(bfs["optimal_steps"])
    if canonical_paths.get("optimal_steps") is not None:
        return int(canonical_paths["optimal_steps"])
    return None


def _runtime_capped_spec(spec, canonical_paths: dict[str, Any]):
    """Clamp runtime max_steps to 3x solver optimal without changing task files."""
    optimal_steps = _canonical_optimal_steps(canonical_paths)
    if optimal_steps is None or optimal_steps <= 0:
        return spec
    cap = max(1, optimal_steps * RUNTIME_MAX_STEPS_OPTIMAL_MULTIPLIER)
    if spec.max_steps <= cap:
        return spec
    return dataclasses.replace(spec, max_steps=cap)


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
    difficulty_max_static_score: Optional[float] = None,
) -> tuple[dict[str, dict[str, Any]], float]:
    static_by_task = score_tasks(rows, manifest_path, artifacts_root, config, force=force)
    scores = [float(s.get("static_score", 0.0)) for s in static_by_task.values()]
    observed_max = max(scores) if scores else 0.0
    difficulty_max = (
        float(difficulty_max_static_score)
        if difficulty_max_static_score is not None
        else config.difficulty_max_static_score
    )
    if difficulty_max is None:
        raise ValueError(
            "Pipeline requires difficulty_max_static_score from --difficulty-max-static-score "
            "or scorer config."
        )
    if difficulty_max <= 0:
        raise ValueError("difficulty_max_static_score must be greater than zero.")
    if observed_max > difficulty_max:
        raise ValueError(
            "difficulty_max_static_score must be at least the largest selected task "
            f"static score ({observed_max})."
        )
    suite_path = artifacts_root / "tasks" / "_suite.json"
    _write_json_atomic(
        suite_path,
        {
            "difficulty_max_static_score": difficulty_max,
            "observed_max_static_score": observed_max,
            "tasks": {t: s.get("static_score") for t, s in static_by_task.items()},
        },
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
    model_config: Optional[dict[str, Any]] = None,
    manifest_path: Path,
    artifacts_root: Path,
    static_by_task: dict[str, dict[str, Any]],
    difficulty_max: float,
    config: ScorerConfig,
    seeds: Iterable[int],
    conditions: Optional[str],
    prompt_variant: Optional[str] = None,
    base_overrides: Optional[dict] = None,
    phase: Optional[dict[str, Any]] = None,
    force: bool,
) -> tuple[list[dict[str, Any]], dict[tuple, Optional[float]]]:
    condition_configs = _condition_configs(
        conditions, prompt_variant=prompt_variant, base_overrides=base_overrides
    )
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
        for seed in seeds:
            for variant, cfg in condition_configs:
                result = _run_one_unit(
                    row,
                    agent,
                    model_name,
                    manifest_path=manifest_path,
                    artifacts_root=artifacts_root,
                    static_by_task=static_by_task,
                    difficulty_max=difficulty_max,
                    config=config,
                    seed=seed,
                    prompt_variant=variant,
                    experiment_config=cfg,
                    conditions=conditions,
                    base_overrides=base_overrides,
                    model_config=model_config,
                    phase=phase,
                    force=force,
                )
                if result is None:
                    continue
                run_row, composite = result
                run_rows.append(run_row)
                composites[
                    (task_id, model_name, seed, row.get("condition"), variant)
                ] = composite

    return run_rows, composites


@dataclasses.dataclass
class _PreparedUnitRun:
    """Everything needed to run + score ONE task/model/seed/prompt-variant unit,
    without having driven the model yet.

    ``_prepare_unit_run`` factors this out of ``_run_one_unit`` so the distributed
    lockstep-batch worker (which drives the episode through an ``EpisodeStepper``
    rather than ``run_episode``) writes byte-identical ``run_inputs.json`` /
    ``run_score.json`` through the SAME code — in particular the same
    ``inputs_hash`` that upload verification requires.
    """

    run_dir: Path
    source: Path
    runtime_spec: Any
    experiment_config: Any
    expected_hash: str
    episode_path: Path
    sidecar_path: Path
    # write_run_inputs(extras=None): write run_inputs.json. ``extras`` is additive
    # provenance ONLY (e.g. pricing_tier) and never enters ``inputs_hash``.
    write_run_inputs: Callable[..., None]
    # score_episode(episode) -> (run_row, composite): write run_score.json + row.
    score_episode: Callable[[dict[str, Any]], tuple[dict[str, Any], Optional[float]]]


def _prepare_unit_run(
    row: dict[str, Any],
    model_name: str,
    *,
    model_config: Optional[dict[str, Any]] = None,
    manifest_path: Path,
    artifacts_root: Path,
    scored_static: dict[str, Any],
    difficulty_max: float,
    config: ScorerConfig,
    seed: int,
    prompt_variant: str,
    experiment_config: Any,
    conditions: Optional[str] = None,
) -> _PreparedUnitRun:
    """Resolve the run dir, inputs hash, and the run_inputs/run_score writers for
    one unit. Pure setup — no model call, no scoring happens here.

    ``experiment_config`` must be already resolved (the callers that could pass
    ``None`` do so before this point). Verbatim-moved from ``_run_one_unit`` so
    the on-disk artifacts and hashes are unchanged.
    """
    task_id = row["task_id"]
    source = _resolve_source(row, manifest_path)
    spec = task_spec_from_payload(json.loads(Path(source).read_text(encoding="utf-8")))
    canonical = json.loads(
        (artifacts_root / "tasks" / task_id / "canonical_paths.json").read_text(encoding="utf-8")
    )
    runtime_spec = _runtime_capped_spec(spec, canonical)
    run_dir = _run_dir(artifacts_root, task_id, model_name, seed, prompt_variant)
    episode_path = run_dir / "episode.json"
    sidecar_path = run_dir / "run_inputs.json"
    run_score_path = run_dir / "run_score.json"

    # ``condition`` is the task-intrinsic axis (test-3 mechanism order, carried
    # by the manifest); ``prompt_variant`` is the orthogonal prompt axis from
    # --conditions.
    manifest_row = dict(row)

    expected_hash = _expected_run_hash(
        runtime_spec,
        model_name,
        seed,
        "minigrid",
        condition_set=conditions,
        prompt_variant=prompt_variant,
        experiment_config=experiment_config,
        model_config=model_config,
    )

    def write_run_inputs(extras: Optional[dict[str, Any]] = None) -> None:
        payload = {
            "inputs_hash": expected_hash,
            "producer_version": PIPELINE_VERSION,
            "task_id": task_id,
            "model_id": model_name,
            "model_config": _jsonable(model_config or {}),
            "runtime_model_config": _runtime_model_config(model_config),
            "seed": seed,
            "backend": "minigrid",
            "condition": prompt_variant,
            "condition_set": conditions,
            "prompt_variant": prompt_variant,
            "experiment_config": _experiment_config_payload(experiment_config),
            "runtime_max_steps_cap": {
                "multiplier": RUNTIME_MAX_STEPS_OPTIMAL_MULTIPLIER,
                "optimal_steps": _canonical_optimal_steps(canonical),
                "original_max_steps": spec.max_steps,
                "effective_max_steps": runtime_spec.max_steps,
            },
        }
        if extras:
            # Additive provenance only. ``inputs_hash`` is computed above from
            # ``_expected_run_hash`` (which never sees this sidecar payload), so
            # these keys are recorded but excluded from the hashed material.
            payload.update(extras)
        _write_json_atomic(sidecar_path, payload)

    def score_episode(
        episode: dict[str, Any],
    ) -> tuple[dict[str, Any], Optional[float]]:
        metrics = episode_metrics.build_metrics(episode, canonical, manifest_row)
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

        run_row = episode_metrics.build_run_row(
            episode,
            canonical,
            manifest_row,
            agent_or_model=model_name,
            seed=seed,
            raw_output_ref=str(episode_path.relative_to(artifacts_root)),
            metrics=metrics,
            prompt_variant=prompt_variant,
        )
        return run_row, run_score.get("composite")

    return _PreparedUnitRun(
        run_dir=run_dir,
        source=Path(source),
        runtime_spec=runtime_spec,
        experiment_config=experiment_config,
        expected_hash=expected_hash,
        episode_path=episode_path,
        sidecar_path=sidecar_path,
        write_run_inputs=write_run_inputs,
        score_episode=score_episode,
    )


def _run_one_unit(
    row: dict[str, Any],
    agent: Agent,
    model_name: str,
    *,
    model_config: Optional[dict[str, Any]] = None,
    manifest_path: Path,
    artifacts_root: Path,
    static_by_task: dict[str, dict[str, Any]],
    difficulty_max: float,
    config: ScorerConfig,
    seed: int,
    prompt_variant: str,
    experiment_config: Any | None = None,
    conditions: Optional[str] = None,
    base_overrides: Optional[dict] = None,
    phase: Optional[dict[str, Any]] = None,
    force: bool = False,
) -> Optional[tuple[dict[str, Any], Optional[float]]]:
    """Run Stage 3/4 for exactly one task/model/seed/prompt variant."""
    from pipeline.run_stage3 import run_episode

    task_id = row["task_id"]
    scored_static = static_by_task[task_id]
    if not scored_static.get("is_beatable", True):
        return None

    if experiment_config is None:
        configs = _condition_configs(
            conditions, prompt_variant=prompt_variant, base_overrides=base_overrides
        )
        if len(configs) != 1:
            raise ValueError(f"Expected one config for prompt variant {prompt_variant!r}.")
        _, experiment_config = configs[0]

    prep = _prepare_unit_run(
        row,
        model_name,
        model_config=model_config,
        manifest_path=manifest_path,
        artifacts_root=artifacts_root,
        scored_static=scored_static,
        difficulty_max=difficulty_max,
        config=config,
        seed=seed,
        prompt_variant=prompt_variant,
        experiment_config=experiment_config,
        conditions=conditions,
    )

    episode = None
    if not force and prep.episode_path.exists() and prep.sidecar_path.exists():
        sidecar = _load_json_object_if_valid(prep.sidecar_path)
        if sidecar is not None and sidecar.get("inputs_hash") == prep.expected_hash:
            episode = _load_json_object_if_valid(prep.episode_path)

    if episode is None:
        episode = run_episode(
            prep.source,
            prep.experiment_config,
            agent,
            seed,
            prep.run_dir,
            max_steps=prep.runtime_spec.max_steps,
            provenance=_phase_episode_provenance(phase, model_config),
        )
        # ``phase`` rides in the run_inputs extras (additive; written AFTER the
        # inputs_hash is computed, so it is recorded but excluded from the hash).
        prep.write_run_inputs(extras={"phase": phase} if phase else None)

    return prep.score_episode(episode)


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
    prompt_variant: Optional[str] = None,
    artifacts_root: str | Path = "artifacts",
    run_set_id: str = "default",
    scorer_config: Optional[ScorerConfig] = None,
    difficulty_max_static_score: Optional[float] = None,
    force: bool = False,
) -> dict[str, Any]:
    """Single-model convenience entry: run one experiment with one agent."""
    manifest_path = Path(manifest_path)
    artifacts_root = Path(artifacts_root)
    config = scorer_config or load_scorer_config()

    catalog = load_manifest(manifest_path)
    rows = resolve_task_rows([experiment], catalog, manifest_path)
    static_by_task, difficulty_max = _score_suite(
        rows,
        manifest_path,
        artifacts_root,
        config,
        force,
        difficulty_max_static_score=difficulty_max_static_score,
    )
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
        prompt_variant=prompt_variant,
        force=force,
    )
    return _write_aggregate(run_rows, composites, static_by_task, rows, artifacts_root, run_set_id)


def run_from_config(
    *,
    run_config_path: str | Path,
    manifest_path: str | Path = _DEFAULT_MANIFEST,
    seeds: Iterable[int] = (0,),
    conditions: Optional[str] = None,
    prompt_variant: Optional[str] = None,
    artifacts_root: str | Path = "artifacts",
    run_set_id: str = "default",
    scorer_config: Optional[ScorerConfig] = None,
    difficulty_max_static_score: Optional[float] = None,
    force: bool = False,
    agent_factory: Optional[AgentFactory] = None,
) -> dict[str, Any]:
    """Run-config entry: each model runs its own task selection (model -> task files)."""
    manifest_path = Path(manifest_path)
    artifacts_root = Path(artifacts_root)
    config = scorer_config or load_scorer_config()
    factory = agent_factory or _build_agent_from_spec

    run_config = load_run_config(run_config_path)
    check_run_config_expectations(run_config, manifest_path, conditions)
    catalog = load_manifest(manifest_path)

    from interface.config import ExperimentConfig

    exp_overlay = run_config.get("experiment_config") or {}
    if exp_overlay:
        # Fail fast on unknown keys / invalid values before any paid model call.
        ExperimentConfig.from_dict({**ExperimentConfig().to_dict(), **exp_overlay})

    # Optional top-level two-tier phase block (Task B3): {"pass": int,
    # "label": str}. Pure provenance — stamped onto artifacts, excluded from
    # every input hash (see ``_phase_episode_provenance`` and the distributed
    # ``job_digest`` phase-strip).
    phase = run_config.get("phase")

    # Resolve each model's task rows + build its agent.
    plans: list[tuple[str, Agent, dict[str, Any], list[dict[str, Any]]]] = []
    union: dict[str, dict[str, Any]] = {}
    for name, model_cfg in run_config["models"].items():
        entries = model_cfg.get("tasks") or model_cfg.get("runs") or []
        if not entries:
            raise ValueError(f"Model {name!r} lists no tasks/runs.")
        rows = resolve_task_rows(entries, catalog, manifest_path)
        agent, label = factory(name, model_cfg)
        plans.append((_sanitize(label), agent, dict(model_cfg), rows))
        for r in rows:
            union.setdefault(r["task_id"], r)

    union_rows = list(union.values())
    static_by_task, difficulty_max = _score_suite(
        union_rows,
        manifest_path,
        artifacts_root,
        config,
        force,
        difficulty_max_static_score=difficulty_max_static_score,
    )

    all_run_rows: list[dict[str, Any]] = []
    composites: dict[tuple, Optional[float]] = {}
    for model_name, agent, model_cfg, rows in plans:
        rr, comp = _run_one_model(
            rows,
            agent,
            model_name,
            model_config=model_cfg,
            manifest_path=manifest_path,
            artifacts_root=artifacts_root,
            static_by_task=static_by_task,
            difficulty_max=difficulty_max,
            config=config,
            seeds=seeds,
            conditions=conditions,
            prompt_variant=prompt_variant,
            base_overrides=exp_overlay,
            phase=phase,
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


def check_run_config_expectations(
    run_config: dict[str, Any],
    manifest_path: str | Path,
    conditions: Optional[str],
) -> None:
    """Guard against a mispaired launch: if a run-config declares ``manifest``
    and/or ``conditions``, the CLI values must match, so you cannot silently
    benchmark the wrong suite or the wrong prompt axis across paid models."""
    expected_manifest = run_config.get("manifest")
    if expected_manifest is not None:
        # The declared path is repo-root-relative; resolve it against the repo
        # root (not cwd) so the guard holds regardless of where it is launched.
        expected_path = Path(expected_manifest)
        if not expected_path.is_absolute():
            expected_path = _REPO_ROOT / expected_path
        if expected_path.resolve() != Path(manifest_path).resolve():
            raise ValueError(
                f"Run-config expects --manifest {expected_manifest!r} but got {str(manifest_path)!r}. "
                "Pass the matching --manifest (or fix the run-config)."
            )
    if "conditions" in run_config and run_config["conditions"] != conditions:
        raise ValueError(
            f"Run-config expects --conditions {run_config['conditions']!r} but got {conditions!r}. "
            "Pass the matching --conditions."
        )
    _check_equal_token_caps(run_config)


def _check_equal_token_caps(run_config: dict[str, Any]) -> None:
    """Cross-model runs must use one max_tokens for every model.

    Unequal budgets confounded baseline_thinking (FINDINGS §4): with thinking
    on, a model can spend its whole budget before the answer line, so caps of
    4k/8k/16k made the ranking measure budget, not reasoning. A config that
    intentionally runs asymmetric budgets must say so with
    ``"allow_unequal_max_tokens": true``.
    """
    models = run_config.get("models") or {}
    if len(models) < 2 or run_config.get("allow_unequal_max_tokens") is True:
        return
    caps = {name: cfg.get("max_tokens") for name, cfg in models.items()}
    if len(set(caps.values())) > 1:
        raise ValueError(
            f"Cross-model run declares unequal max_tokens {caps}. Unequal budgets make the "
            "comparison measure budget, not ability (models truncate at different depths). "
            'Use one cap for every model, or set "allow_unequal_max_tokens": true to put '
            "the asymmetry on record."
        )


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
        if "timeout" in model_cfg:
            cfg.timeout = float(model_cfg["timeout"])
        if "max_attempts" in model_cfg:
            cfg.max_attempts = int(model_cfg["max_attempts"])
        if "enable_thinking" in model_cfg:
            cfg.enable_thinking = bool(model_cfg["enable_thinking"])
        if "effort" in model_cfg:
            cfg.effort = str(model_cfg["effort"])
        if "batch_deadline_s" in model_cfg:
            cfg.batch_deadline_s = float(model_cfg["batch_deadline_s"])
        if "batch_cancel_grace_s" in model_cfg:
            cfg.batch_cancel_grace_s = float(model_cfg["batch_cancel_grace_s"])
        return ClaudeAnthropicAgent(config=cfg), model or cfg.model
    if provider == "kimi":
        from interface.agents import KimiK26Agent, KimiK26Config

        cfg = KimiK26Config(temperature=temperature)
        if model:
            cfg.model = model
        if max_tokens:
            cfg.max_tokens = int(max_tokens)
        if "timeout" in model_cfg:
            cfg.timeout = float(model_cfg["timeout"])
        if "max_attempts" in model_cfg:
            cfg.max_attempts = int(model_cfg["max_attempts"])
        if "enable_thinking" in model_cfg:
            cfg.enable_thinking = bool(model_cfg["enable_thinking"])
        if "batch_deadline_s" in model_cfg:
            cfg.batch_deadline_s = float(model_cfg["batch_deadline_s"])
        if "batch_cancel_grace_s" in model_cfg:
            cfg.batch_cancel_grace_s = float(model_cfg["batch_cancel_grace_s"])
        return KimiK26Agent(config=cfg), model or cfg.model
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
            "enable_thinking",
        ):
            if key in model_cfg:
                setattr(cfg, key, model_cfg[key])
        return Qwen35VLAgent(config=cfg), model or cfg.model
    if provider in {"qwen_vllm", "vllm"}:
        from interface.agents import QwenVLLMAgent, QwenVLLMConfig

        cfg = QwenVLLMConfig(temperature=temperature)
        if model:
            cfg.model = model
        if max_tokens:
            cfg.max_tokens = int(max_tokens)
        for key in (
            "max_model_len",
            "gpu_memory_utilization",
            "tensor_parallel_size",
            "dtype",
            "quantization",
            "trust_remote_code",
            "enforce_eager",
            "max_num_seqs",
            "enable_prefix_caching",
            "enable_thinking",
            "seed",
            "download_dir",
            "local_files_only",
            "use_tqdm",
            "engine_kwargs",
            "sampling_kwargs",
        ):
            if key in model_cfg:
                setattr(cfg, key, model_cfg[key])
        return QwenVLLMAgent(config=cfg), model or cfg.model
    if provider in {"qwen_vllm_api", "qwen_openai", "openai_compatible"}:
        from interface.agents import QwenVLLMAPIAgent, QwenVLLMAPIConfig

        cfg = QwenVLLMAPIConfig(temperature=temperature)
        if model:
            cfg.model = model
        if max_tokens:
            cfg.max_tokens = int(max_tokens)
        for key in (
            "base_url",
            "api_key",
            "timeout",
            "enable_thinking",
            "extra_body",
            "max_attempts",
        ):
            if key in model_cfg:
                setattr(cfg, key, model_cfg[key])
        return QwenVLLMAPIAgent(config=cfg), model or cfg.model
    raise ValueError(
        f"Model {name!r}: unknown provider {provider!r} "
        "(expected 'claude', 'kimi', 'qwen', or 'qwen_vllm')."
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="MultiNet v2.0 run pipeline: manifest + run-config driven evaluation runs.")
    parser.add_argument("--run-config", help="JSON run-config mapping models to task files (preferred).")
    parser.add_argument("--manifest", default=str(_DEFAULT_MANIFEST), help="Task catalog (metadata).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--conditions", default=None, help="Prompt condition-set name (optional).")
    parser.add_argument(
        "--prompt-variant",
        default=None,
        help="Run only one variant from --conditions, for example text_only/image_text/image_only.",
    )
    parser.add_argument("--artifacts-root", default=str(_REPO_ROOT / "artifacts"))
    parser.add_argument("--run-set-id", default="default")
    parser.add_argument(
        "--difficulty-max-static-score",
        type=float,
        default=None,
        help="Stable task-suite static-score maximum for runtime normalization.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute existing artifacts.")
    parser.add_argument(
        "--distributed-role",
        choices=[
            "coordinator-prepare",
            "coordinator-serve",
            "worker",
            "lockstep-worker",
            "coordinator-run-api-client",
            "coordinator-finalize",
        ],
        help="Run one distributed pipeline role instead of the single-process pipeline.",
    )
    parser.add_argument("--job-id", help="Optional durable distributed job id for coordinator-prepare.")
    parser.add_argument(
        "--storage-config",
        help="JSON storage config (gsutil bucket) for coordinator-serve/finalize bucket mirroring.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Coordinator bind host for coordinator-serve.")
    parser.add_argument("--port", type=int, default=8765, help="Coordinator bind port for coordinator-serve.")
    parser.add_argument("--coordinator-url", help="Coordinator base URL for worker mode.")
    parser.add_argument("--stale-after-seconds", type=float, default=300.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=10.0)
    parser.add_argument("--heartbeat-interval-seconds", type=float, default=30.0)
    parser.add_argument("--worker-state", help="Path to a worker-local retry/resume state file.")
    parser.add_argument("--model-group", help="Worker model group capability.")
    parser.add_argument("--hardware-profile", help="Worker hardware profile capability.")
    parser.add_argument("--worker-tag", action="append", help="Worker tag capability; may be repeated.")
    parser.add_argument("--local-model-cache", action="append", help="Locally cached model id; may be repeated.")
    parser.add_argument("--max-units", type=int, default=1, help="Maximum local API-client units to run (0 or less = drain all pending).")
    parser.add_argument("--client-artifacts-root", help="Local artifact root for coordinator-run-api-client.")
    parser.add_argument("--once", action="store_true", help="Worker mode: process at most one assignment.")
    parser.add_argument("--worker-concurrency", type=int, default=1,
                        help="Worker mode: run up to N units in parallel (served-vLLM agents only).")
    parser.add_argument(
        "--allow-partial-finalize",
        action="store_true",
        help="Finalize reports from received units even when some work is missing.",
    )
    # Single-model fallback (when --run-config is not supplied):
    parser.add_argument("--experiment", choices=["test1", "test2", "test3", "r1", "all"], default="all")
    parser.add_argument("--agent", choices=["claude", "kimi", "qwen"], help="Single-model provider.")
    args = parser.parse_args(argv)

    if args.distributed_role:
        from scripts.distributed_run_pipeline import dispatch_distributed_role

        dispatch_distributed_role(args)
        return

    if args.run_config:
        payloads = run_from_config(
            run_config_path=args.run_config,
            manifest_path=args.manifest,
            seeds=args.seeds,
            conditions=args.conditions,
            prompt_variant=args.prompt_variant,
            artifacts_root=args.artifacts_root,
            run_set_id=args.run_set_id,
            difficulty_max_static_score=args.difficulty_max_static_score,
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
            prompt_variant=args.prompt_variant,
            artifacts_root=args.artifacts_root,
            run_set_id=args.run_set_id,
            difficulty_max_static_score=args.difficulty_max_static_score,
            force=args.force,
        )

    summary = payloads["scoring_calibration_summary"]
    print(
        f"Pipeline complete: {summary['run_count']} runs over {summary['task_count']} tasks "
        f"-> {args.artifacts_root}/reports/{args.run_set_id}/"
    )


if __name__ == "__main__":
    main()
