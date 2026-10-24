#!/usr/bin/env python
"""Resume an infra-killed episode from its on-disk archive, under as-run code.

The R1 episode r1_M6_14x14_dense_kr_sg_kb_1 / kimi-k2.6 / seed_0 was killed at
env step 143 by a Moonshot network outage (terminal ``parse_failed`` produced
by empty replies, not by the model). R1-frontier GPT-6 Astra episodes were
killed mid-run by OpenAI billing exhaustion (terminal
``agent_error:RuntimeError: OpenAI API quota exhausted ...``). Both archived
``episode.json`` transcripts have exactly the record shape that
``interface/episode_checkpoint.py`` persists in a query-boundary checkpoint, so
this driver:

1. inserts ``--repo-root`` (a git worktree of the as-run SHA) at ``sys.path[0]``
   BEFORE importing any harness module, so ``interface``/``gridworld`` come
   from the old code;
2. reconstructs the ``TaskSpecification`` from the ``task_spec`` EMBEDDED in
   ``episode.json`` (no manifests/submodules touched);
3. builds a synthetic checkpoint payload from the archive (query counter and
   trailing parse-failure count rebuilt from the query records; the terminal
   infra-kill status — ``parse_failed`` or any ``agent_error:`` — cleared) and
   hands it to the old code's ``resume_stepper``, which replays the recorded
   actions through a fresh backend, verifying position_after/state_after per
   step and rebuilding stall-watchdog state with the old code's
   ``_progress_signature``;
4. ``--mode replay-verify``: prints a verification report and exits 0 only if
   the replayed final state matches the archive's last recorded state;
5. ``--mode continue``: keeps running the live query loop (the Kimi, OpenAI or
   Claude agent rebuilt from ``run_inputs.json``'s model config verbatim, OpenAI
   and Claude taking ``enable_prompt_cache`` from the experiment when the entry
   leaves it unset; OpenAI and Claude also need ``--spend-cap-usd`` because their
   spend ledgers are per process) and flushes full episode artifacts to
   ``--out-dir`` via the old code's ``flush_episode_log``.

A second source, ``--unit-id`` with ``--job-plan`` and ``--worker-root``, resumes a
unit that is still in flight on a lockstep (batch API) worker from its live
query-boundary ``checkpoint.json``. Each query is sent on its own through the
agent's standard ``generate``, so the episode no longer waits on batch rounds.
The unit is rebuilt through the same setup path the lockstep worker uses and must
reproduce the plan's ``inputs_hash``. Continue mode checkpoints every query under
``--out-dir``; an interrupted run resumes from there with ``--checkpoint``. A
finished run writes the same artifacts the worker would (episode.json,
run_inputs.json, run_score.json) plus the upload archive, and with
``--coordinator-url`` uploads it. Stop the lockstep worker before resuming its
units: both would otherwise drive the same episodes.

Sources are never written to; the worktree is never modified.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
from pathlib import Path

ARCHIVED_END_REASONS_TO_CLEAR = {"parse_failed"}
# The runner ends an episode with "agent_error:<Type>: ..." only for transport or
# billing failures (interface/runner.py) — an infra outcome, never a model one.
INFRA_KILL_PREFIXES = ("agent_error:",)
# The stepper's in-progress default (see EpisodeStepper.start()).
IN_PROGRESS_END_REASON = "max_steps"
# Mirrors interface/episode_step._INFRA_STOP_REASONS (a test keeps them equal):
# replies the provider failed to deliver, which never spend the parse-retry budget.
INFRA_STOP_REASONS = frozenset(
    {"sync_error", "batch_expired", "batch_errored", "batch_canceled", "engine_overloaded"}
)
# Providers whose resumes must pass their own spend cap: the agent's ledger is per
# process, so the archived per-run cap would not bound a resume.
_PER_PROCESS_SPEND_PROVIDERS = ("openai", "claude")
# Providers that take enable_prompt_cache from the experiment when the entry leaves
# it unset (run_pipeline._agent_model_config).
_PROMPT_CACHE_PROVIDERS = ("openai", "claude")


# --------------------------------------------------------------------------
# Pure helpers (no harness imports; unit-testable from the main checkout).
# --------------------------------------------------------------------------

def trailing_parse_failures(query_records: list[dict]) -> int:
    """parse_failures resets to 0 on any successfully parsed reply and an infra
    reply (see INFRA_STOP_REASONS) leaves it unchanged, so the live counter
    equals the number of TRAILING parse_ok=False query records that were not
    infra replies."""
    n = 0
    for rec in reversed(query_records):
        if rec.get("parse_ok"):
            break
        if rec.get("stop_reason") in INFRA_STOP_REASONS:
            continue
        n += 1
    return n


def is_infra_kill(end_reason: object) -> bool:
    """True for archived end reasons this driver may clear and resume."""
    return end_reason in ARCHIVED_END_REASONS_TO_CLEAR or str(end_reason).startswith(
        INFRA_KILL_PREFIXES
    )


def build_checkpoint_payload(episode: dict) -> dict:
    """A synthetic query-boundary checkpoint equivalent to what the as-run
    process would have saved just before the infra failure killed it, with the
    terminal status cleared so the episode continues as if the lost query were
    re-issued."""
    transcript = episode["transcript"]
    query_records = [r for r in transcript if r.get("kind") == "query"]
    step_records = [r for r in transcript if r.get("kind") == "step"]
    end_reason = episode.get("end_reason")
    if not is_infra_kill(end_reason):
        raise SystemExit(
            f"archive end_reason={end_reason!r} is not an infra-kill this driver "
            f"knows how to clear (expected one of {sorted(ARCHIVED_END_REASONS_TO_CLEAR)} "
            f"or a {INFRA_KILL_PREFIXES} prefix); refusing to resume a genuinely "
            "finished episode"
        )
    parse_failures = trailing_parse_failures(query_records)
    return {
        "transcript": transcript,
        "seed": episode["task_spec"]["seed"],
        "query_count": len(query_records),
        "parse_failures": parse_failures,
        "consecutive_failures": (
            step_records[-1]["consecutive_failures_after"] if step_records else 0
        ),
        "finished": False,
        "end_reason": IN_PROGRESS_END_REASON,
    }


def rehydrate_agent_messages(transcript: list[dict], archive_dir: Path) -> int:
    """Archived query records store images as {"type": "image", "file": ...}
    references into the archive's queries/ dir. The old flush path only
    understands live {"type": "image_url", ...data URI...} blocks and would
    silently drop the archived references on re-flush. Convert them back so a
    continue-mode flush writes complete artifacts. A PNG missing from the
    archive (the outage-era pull lost queries 134-146's images) becomes an
    explicit text placeholder rather than a silent drop. Reads only; mutates
    the in-memory transcript copy. Returns the count of missing images."""
    missing = 0
    for rec in transcript:
        if rec.get("kind") != "query":
            continue
        qdir = archive_dir / "queries" / f"query_{rec['query_index']:03d}"
        for msg in rec.get("agent_messages", []):
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "image" and "file" in blk:
                    png_path = qdir / blk["file"]
                    if png_path.is_file():
                        b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
                        blk.clear()
                        blk["type"] = "image_url"
                        blk["image_url"] = {"url": "data:image/png;base64," + b64}
                    else:
                        missing += 1
                        name = blk["file"]
                        blk.clear()
                        blk["type"] = "text"
                        blk["text"] = f"[image {name} missing from source archive]"
    return missing


def drive_continue_loop(stepper, agent, *, max_new_queries: int, log=print, reset_usage=None,
                        checkpoint=None):
    """Run live query rounds until the stepper finishes, the cap is hit, or the
    agent dies. Returns ``(new_queries, stopped_early)``; a non-None
    ``stopped_early`` becomes a ``resume_interrupted:`` end_reason.

    A transport failure (socket timeout past the retry budget, non-retryable
    HTTP status) is caught rather than propagated: crashing here would discard
    every paid query of a multi-hour resume, since artifacts are only flushed
    after this loop returns.

    ``checkpoint(stepper)``, when given, runs before each query is sent: that is
    the query boundary ``save_checkpoint`` accepts, so a killed process loses at
    most the query in flight.

    Duck-typed on the as-run code's stepper/agent so it stays importable (and
    testable) from the main checkout.
    """
    new_queries = 0
    stopped_early: str | None = None
    while (messages := stepper.next_query()) is not None:
        if getattr(agent, "exhausted", False):
            stopped_early = "scripted_agent_exhausted"
        elif new_queries >= max_new_queries:
            stopped_early = f"max_new_queries({max_new_queries})"
        if stopped_early:
            # next_query already reserved this query's index; roll it back so
            # the artifact's query_count equals the applied-query count (same
            # convention resume_stepper uses for an in-flight round).
            stepper.query_count -= 1
            break
        if checkpoint is not None:
            checkpoint(stepper)
        if reset_usage is not None:
            reset_usage(agent)
        try:
            reply = agent.generate(messages)
            stepper.apply_reply(reply)
        except Exception as exc:  # noqa: BLE001 — never lose paid work
            stopped_early = f"agent_error:{type(exc).__name__}: {exc}"[:200]
            log(f"  ABORTING on agent error (artifacts still flushed): {exc}")
            stepper.query_count -= 1
            break
        new_queries += 1
        log(f"  new query {new_queries}: env_step={stepper.state.step_count} "
            f"parsed={stepper.action_queue or 'PARSE-FAIL'}")
    return new_queries, stopped_early


def _render_block(payload: dict) -> dict | None:
    render = payload.get("render")
    if render is None or (isinstance(render, dict) and render.get("backend", "minigrid") == "minigrid"):
        return None
    return render


def recorded_render(episode: dict, run_inputs: dict) -> dict | None:
    """The 3D render block an archive recorded, or None for a 2D archive.

    run_inputs.json and episode.json must record the same block, and a 3D
    backend label needs one: otherwise the driver cannot tell how the frames
    were drawn, so it refuses rather than continue on a guessed backend."""
    from_inputs, from_episode = _render_block(run_inputs), _render_block(episode)
    if from_inputs != from_episode:
        raise SystemExit(
            f"refusing to resume: run_inputs.json render={from_inputs!r} and "
            f"episode.json render={from_episode!r} disagree"
        )
    label = run_inputs.get("backend", "minigrid")
    if from_inputs is None and label != "minigrid":
        raise SystemExit(
            f"refusing to resume: run_inputs.json backend {label!r} names a 3D render, "
            "but no render block records how its frames were drawn"
        )
    return from_inputs


def scripted_action_from_spec(spec: str) -> str:
    prefix = "scripted:"
    if not spec.startswith(prefix) or not spec[len(prefix):]:
        raise SystemExit(f"--agent must look like scripted:<ACTION>, got {spec!r}")
    return spec[len(prefix):]


def _capped_fields(config_cls, provider: str) -> set[str]:
    """The config fields the agent accepts. An as-run worktree whose agent has no
    spend_cap_usd field would drop the cap silently, so refuse instead."""
    allowed = set(config_cls.__dataclass_fields__)
    if "spend_cap_usd" not in allowed:
        raise SystemExit(
            f"the as-run {provider} agent has no spend_cap_usd field, so this resume "
            "could not be capped; use a worktree whose agent enforces one"
        )
    return allowed


def build_live_agent(model_cfg: dict, *, timeout: float | None = None,
                     spend_cap_usd: float | None = None, log=print):
    """Rebuild the live agent from the archived run_inputs model config.

    Imports are deferred so that, under main(), they resolve against the as-run
    worktree. Only fields the agent's config dataclass defines pass through;
    fleet routing keys (group, max_in_flight, tasks, ...) are dropped.
    """
    cfg = dict(model_cfg)
    if timeout is not None:
        log(f"  timeout override: {cfg.get('timeout')} -> {timeout}s")
        cfg["timeout"] = timeout
    provider = str(cfg.get("provider") or "kimi").lower()
    if provider in _PER_PROCESS_SPEND_PROVIDERS:
        if spend_cap_usd is None:
            raise SystemExit(
                f"--spend-cap-usd is required to resume a {provider} episode: the agent's "
                "spend ledger is per process and starts at $0 here, so the archived "
                f"per-run cap (${cfg.get('spend_cap_usd')}) would not bound this resume"
            )
        cfg["spend_cap_usd"] = float(spend_cap_usd)
    if provider == "openai":
        from interface.agents.openai_agent import OpenAIAgent, OpenAIConfig

        allowed = _capped_fields(OpenAIConfig, provider)
        return OpenAIAgent(config=OpenAIConfig(**{k: v for k, v in cfg.items() if k in allowed}))
    if provider == "claude":
        # Queries go through the agent's standard (sync) generate, never the
        # Message Batches path, whatever worker_role the run config named.
        from interface.agents.claude import ClaudeAnthropicAgent, ClaudeAnthropicConfig

        cfg["cache_ttl"] = "1h"
        allowed = _capped_fields(ClaudeAnthropicConfig, provider)
        return ClaudeAnthropicAgent(
            config=ClaudeAnthropicConfig(**{k: v for k, v in cfg.items() if k in allowed})
        )
    if provider == "kimi":
        from interface.agents.kimi_k26 import KimiK26Agent, KimiK26Config

        allowed = set(KimiK26Config.__dataclass_fields__)
        return KimiK26Agent(config=KimiK26Config(**{k: v for k, v in cfg.items() if k in allowed}))
    raise SystemExit(f"resume does not support provider {provider!r}")


def with_prompt_cache_default(model_cfg: dict, experiment_config: dict) -> dict:
    """The pipeline derives Claude's and OpenAI's enable_prompt_cache from the
    experiment's prompt_cache when the entry leaves it unset
    (run_pipeline._agent_model_config); inlined because this module imports
    from the as-run worktree."""
    cfg = dict(model_cfg)
    if str(cfg.get("provider") or "").lower() in _PROMPT_CACHE_PROVIDERS:
        cfg.setdefault("enable_prompt_cache", bool(experiment_config.get("prompt_cache", False)))
    return cfg


def unit_payload(plan: dict, unit_id: str, task_artifacts_root: Path) -> dict:
    """The payload the coordinator's assign hands a worker for ``unit_id``
    (distributed_run_pipeline.CoordinatorStore._unit_payload), rebuilt from the
    job plan and the task artifacts under ``<task_artifacts_root>/tasks/<task_id>/``
    (a worker's artifacts root holds the copies it materialized on assign)."""
    matches = [u for u in plan["units"] if u["unit_id"] == unit_id]
    if not matches:
        raise SystemExit(f"unit {unit_id!r} is not in the job plan")
    unit = dict(matches[0])
    task_dir = Path(task_artifacts_root) / "tasks" / unit["task_id"]
    try:
        canonical_paths = json.loads((task_dir / "canonical_paths.json").read_text(encoding="utf-8"))
        scored_static = json.loads((task_dir / "scored_static.json").read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"task artifacts for {unit['task_id']} missing under {task_dir}: {exc}")
    unit["scorer_config"] = plan["scorer_config"]
    unit["difficulty_max_static_score"] = plan["difficulty_max_static_score"]
    unit["task_artifacts"] = {
        "canonical_paths": canonical_paths,
        "scored_static": scored_static,
        "task_payload": unit["task_payload"],
    }
    return unit


# --------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------

def _install_repo_root(repo_root: Path) -> None:
    """Make the as-run worktree the ONLY source of harness modules."""
    already = {"interface", "gridworld", "prompting_experiments", "scorer", "pipeline", "scripts"}
    loaded = [m for m in sys.modules if m.split(".")[0] in already]
    if loaded:
        raise SystemExit(f"harness modules imported before path setup: {loaded}")
    resolved = str(repo_root.resolve())
    # Drop any entry that would shadow the worktree with the live checkout
    # (script dir, cwd, '' etc. all resolve under some other repo root).
    sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()) != resolved]
    sys.path.insert(0, resolved)


class _ScriptedAgent:
    """Offline stand-in for the live agent: emits one fixed action in valid
    FINAL_OUTPUT form for ``limit`` queries, then reports itself exhausted."""

    def __init__(self, action: str, reply_cls, limit: int = 2) -> None:
        self.action = action
        self.limit = limit
        self.calls = 0
        self._reply_cls = reply_cls

    @property
    def exhausted(self) -> bool:
        return self.calls >= self.limit

    def generate(self, messages):
        self.calls += 1
        return self._reply_cls(
            text=f"FINAL_OUTPUT: {self.action}",
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            stop_reason="scripted",
        )


def _main_unit(args) -> int:
    """Resume an in-flight lockstep unit from its query-boundary checkpoint.

    The setup and the finalize steps follow run_lockstep_worker's _build_unit and
    _finalize_unit, inlined because this module imports from the as-run worktree.
    """
    if args.job_plan is None or args.worker_root is None:
        raise SystemExit("--unit-id needs --job-plan and --worker-root")
    if bool(args.coordinator_url) != bool(args.worker_id):
        raise SystemExit("--coordinator-url and --worker-id go together")
    plan = json.loads(args.job_plan.read_text(encoding="utf-8"))
    worker_root: Path = args.worker_root.resolve()
    out_root: Path = args.out_dir.resolve()
    unit = unit_payload(plan, args.unit_id, worker_root)
    checkpoint = (args.checkpoint or worker_root / unit["run_dir"] / "checkpoint.json").resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"no checkpoint at {checkpoint}")
    if out_root == worker_root or worker_root in out_root.parents:
        raise SystemExit("--out-dir must not point into --worker-root")
    if not (args.repo_root / "interface" / "episode_checkpoint.py").is_file():
        raise SystemExit(f"{args.repo_root} does not look like a harness worktree")

    _install_repo_root(args.repo_root)

    # Old-code imports — only AFTER the worktree owns sys.path[0].
    import scripts.distributed_run_pipeline as drp
    from interface.config import ExperimentConfig
    from interface.episode_checkpoint import discard_checkpoint, resume_stepper, save_checkpoint
    from interface.episode_log import flush_episode_log
    from interface.runner import _reset_agent_usage
    from pipeline.run_stage3 import build_episode_runner

    render = drp._unit_render(unit)
    row = drp.materialize_worker_inputs(unit, out_root)
    prep = drp.pipeline._prepare_unit_run(
        row,
        unit["model_id"],
        model_config=unit["model_config"],
        manifest_path=out_root / drp.DISTRIBUTED_DIR / "worker_manifest.json",
        artifacts_root=out_root,
        scored_static=unit["task_artifacts"]["scored_static"],
        difficulty_max=float(unit["difficulty_max_static_score"]),
        config=drp.ScorerConfig.from_dict(unit["scorer_config"]),
        seed=int(unit["seed"]),
        prompt_variant=str(unit["prompt_variant"]),
        experiment_config=ExperimentConfig.from_dict(unit["experiment_config"]),
        conditions=unit.get("condition_set"),
        render=render,
    )
    # The hash the coordinator checks an upload's run_inputs.json against.
    if prep.expected_hash != unit["episode_inputs_hash"]:
        raise SystemExit(
            f"refusing to resume: this worktree computes inputs_hash {prep.expected_hash} "
            f"for {unit['unit_id']}, but the plan recorded {unit['episode_inputs_hash']}"
        )
    print(f"backend: {render.label}")
    runner = build_episode_runner(
        prep.source, prep.experiment_config, int(unit["seed"]),
        max_steps=prep.runtime_spec.max_steps, render=render,
    )
    try:
        stepper = resume_stepper(checkpoint, runner=runner)
    except ValueError as exc:
        print("REPLAY DIVERGED — the worktree SHA does not reproduce the "
              "as-run env semantics:")
        print(f"  {exc}")
        return 1
    stepper.maze_path = str(prep.source)
    resumed_at_query = stepper.query_count

    report = {
        "unit_id": unit["unit_id"],
        "task_id": unit["task_id"],
        "checkpoint": str(checkpoint),
        "inputs_hash": prep.expected_hash,
        "steps_replayed": stepper.step_index,
        "env_step_count": stepper.state.step_count,
        "max_steps": prep.runtime_spec.max_steps,
        "queries_counted": stepper.query_count,
        "parse_failures_pending": stepper.parse_failures,
        "stall_count": stepper._stall_watchdog.stall_count if stepper._stall_watchdog else 0,
        "finished": stepper.finished,
    }
    print("=== replay verification ===")
    for key, value in report.items():
        print(f"  {key}: {value}")
    if args.mode == "replay-verify":
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / f"{unit['unit_id']}.replay_verification.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print("replay-verify: PASS")
        return 0

    # ---- continue mode ------------------------------------------------------
    if args.agent:
        from interface.agents.reply import Reply
        agent = _ScriptedAgent(scripted_action_from_spec(args.agent), Reply)
    else:
        agent = build_live_agent(
            with_prompt_cache_default(unit["model_config"], unit["experiment_config"]),
            timeout=args.timeout,
            spend_cap_usd=args.spend_cap_usd,
        )
    prep.run_dir.mkdir(parents=True, exist_ok=True)
    resumable = prep.run_dir / "checkpoint.json"
    new_queries, stopped_early = drive_continue_loop(
        stepper,
        agent,
        max_new_queries=args.max_new_queries,
        reset_usage=_reset_agent_usage,
        checkpoint=lambda s: save_checkpoint(resumable, s),
    )
    if stopped_early:
        save_checkpoint(resumable, stepper)
        print("=== continue interrupted ===")
        print(f"  reason: {stopped_early}")
        print(f"  queries: {stepper.query_count} (+{new_queries} new)")
        print(f"  resume with: --checkpoint {resumable}")
        return 3

    result = stepper.result()
    if render.backend != "minigrid":  # as run_episode stamps it
        result["render"] = render.provenance(prep.runtime_spec)
    flush_episode_log(result, prep.run_dir)
    # Provenance only (never enters inputs_hash): queries before resumed_from_query
    # went through the batch API at batch prices.
    prep.write_run_inputs(extras={"pricing_tier": "standard", "resumed_from_query": resumed_at_query})
    prep.score_episode(json.loads(prep.episode_path.read_text(encoding="utf-8")))
    discard_checkpoint(resumable)
    archive = drp.package_run_archive(unit, artifacts_root=out_root)
    print("=== continue outcome ===")
    print(f"  end_reason: {result['end_reason']}")
    print(f"  success: {result['success']}")
    print(f"  steps_used: {result['steps_used']}")
    print(f"  query_count: {result['query_count']} (+{new_queries} new)")
    print(f"  archive: {archive}")
    if args.coordinator_url:
        drp._upload_archive(
            drp.CoordinatorClient(args.coordinator_url), args.worker_id, unit["unit_id"], archive
        )
        print(f"  uploaded to {args.coordinator_url} as {args.worker_id}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--archive-dir", type=Path,
                    help="the archived seed_<n>/<variant> episode dir (read-only)")
    ap.add_argument("--unit-id", help="resume this in-flight lockstep unit from its checkpoint "
                                      "instead of an archive (needs --job-plan, --worker-root)")
    ap.add_argument("--job-plan", type=Path, help="the coordinator's distributed/job_plan.json")
    ap.add_argument("--worker-root", type=Path,
                    help="the lockstep worker's --artifacts-root (read-only): its tasks/ "
                         "copies and, by default, the unit's checkpoint")
    ap.add_argument("--checkpoint", type=Path,
                    help="checkpoint to resume the unit from (default: the worker's; pass "
                         "the one under --out-dir to continue an interrupted resume)")
    ap.add_argument("--coordinator-url", help="upload the finished unit's archive here")
    ap.add_argument("--worker-id", help="worker id to upload under (the unit's assignee)")
    ap.add_argument("--repo-root", required=True, type=Path,
                    help="worktree of the as-run SHA; inserted at sys.path[0]")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="fresh dir for resumed-episode artifacts (the artifacts root, "
                         "for a unit)")
    ap.add_argument("--mode", required=True, choices=["replay-verify", "continue"])
    ap.add_argument("--max-new-queries", type=int, default=200,
                    help="safety cap on NEW model queries in continue mode")
    ap.add_argument("--timeout", type=float, default=None,
                    help="override the archived per-call socket timeout (seconds). "
                         "Client patience only — no effect on sampling or on "
                         "comparability with the as-run episode. The archived 600s "
                         "predates the run_config default for 64k thinking runs "
                         "(the as-run process had the KIMI_TIMEOUT_OVERRIDE hack).")
    ap.add_argument("--spend-cap-usd", type=float, default=None,
                    help="hard spend cap for THIS process; required for OpenAI and Claude "
                         "agents. Their ledger is per process and starts at $0, so pass this "
                         "resume's share of the remaining budget, not the archived cap.")
    ap.add_argument("--agent", default=None,
                    help="override: scripted:<ACTION> emits that action for 2 "
                         "queries then stops (no API key / spend)")
    args = ap.parse_args()

    if (args.archive_dir is None) == (args.unit_id is None):
        raise SystemExit("pass exactly one source: --archive-dir or --unit-id")
    if args.unit_id is not None:
        return _main_unit(args)
    if bool(args.coordinator_url) != bool(args.worker_id):
        raise SystemExit("--coordinator-url and --worker-id go together")
    if args.coordinator_url or args.checkpoint:
        raise SystemExit("--checkpoint and --coordinator-url apply to --unit-id resumes only")

    archive_dir: Path = args.archive_dir.resolve()
    out_dir: Path = args.out_dir.resolve()
    if not (archive_dir / "episode.json").is_file():
        raise SystemExit(f"no episode.json under {archive_dir}")
    if not (args.repo_root / "interface" / "episode_checkpoint.py").is_file():
        raise SystemExit(f"{args.repo_root} does not look like a harness worktree")
    if out_dir == archive_dir or archive_dir in out_dir.parents:
        raise SystemExit("--out-dir must not point into the archive")

    _install_repo_root(args.repo_root)

    # Old-code imports — only AFTER the worktree owns sys.path[0].
    from gridworld.backends.minigrid_backend import MiniGridBackend
    from gridworld.task_spec import TaskSpecification
    from interface.config import ExperimentConfig
    from interface.episode_checkpoint import resume_stepper
    from interface.episode_log import _json_safe, flush_episode_log, state_snapshot
    from interface.runner import build_runner

    episode = json.loads((archive_dir / "episode.json").read_text(encoding="utf-8"))
    run_inputs = json.loads((archive_dir / "run_inputs.json").read_text(encoding="utf-8"))
    render = recorded_render(episode, run_inputs)

    # Everything derives from the archive: experiment config (incl. stall K and
    # parse-retry caps) from run_inputs, the task (incl. the runtime-capped
    # max_steps) from the embedded task_spec.
    config = ExperimentConfig.from_dict(run_inputs["experiment_config"])
    task_spec = TaskSpecification.from_dict(episode["task_spec"])
    if render is None:
        backend, label = MiniGridBackend(render_mode="rgb_array"), "minigrid"
    else:
        # Rebuild the recorded 3D backend, only if this worktree draws the same
        # frames (render version, mujoco, settings, frame size).
        from gridworld.backends import get_backend
        from gridworld.render_settings import RenderSettings

        keys = ("backend", "camera", "resolution", "start_map")
        settings = RenderSettings.from_run_config({"render": {k: render[k] for k in keys if k in render}})
        current = settings.provenance(task_spec)
        if current != render or run_inputs.get("backend") != settings.label:
            raise SystemExit(
                f"refusing to resume: the archive's frames were drawn with render={render!r} "
                f"(backend {run_inputs.get('backend')!r}), but this worktree draws "
                f"render={current!r} (backend {settings.label!r})"
            )
        backend, label = get_backend(settings.backend, **settings.backend_kwargs(task_spec)), settings.label
    print(f"backend: {label}")
    backend.configure(task_spec)
    runner = build_runner(config, backend, task_spec)

    payload = build_checkpoint_payload(episode)
    with tempfile.TemporaryDirectory(prefix="resume_ckpt_") as tmp:
        ckpt_path = Path(tmp) / "checkpoint.json"
        ckpt_path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            # resume_stepper replays every recorded action through the fresh
            # backend, verifying position_after AND full state_after per step
            # (positions, event outcomes, env_step_count, collected keys,
            # doors, ...), and rebuilds stall state via _progress_signature.
            stepper = resume_stepper(ckpt_path, runner=runner)
        except ValueError as exc:
            print("REPLAY DIVERGED — the worktree SHA does not reproduce the "
                  "as-run env semantics:")
            print(f"  {exc}")
            return 1

    step_records = [r for r in episode["transcript"] if r.get("kind") == "step"]
    replayed_final = _json_safe(state_snapshot(stepper.state))
    archived_final = episode["final_state"]
    final_matches = replayed_final == archived_final
    mismatched_keys = [
        k for k in set(archived_final) | set(replayed_final)
        if archived_final.get(k) != replayed_final.get(k)
    ]

    report = {
        "steps_replayed": stepper.step_index,
        "archived_steps_used": episode["steps_used"],
        "env_step_count": replayed_final["step_count"],
        "final_position_xy": replayed_final["agent_position"],
        "final_position_row_col": replayed_final["position_row_col"],
        "facing": replayed_final["facing"],
        "collected_keys": replayed_final["collected_keys"],
        "open_doors": replayed_final["open_doors"],
        "stall_count": stepper._stall_watchdog.stall_count if stepper._stall_watchdog else 0,
        "stall_k": stepper.stall_k,
        "queries_counted": stepper.query_count,
        "parse_failures_pending": stepper.parse_failures,
        "consecutive_failures": stepper.consecutive_failures,
        "next_feedback": stepper.last_feedback,
        "divergences": 0 if final_matches and stepper.step_index == episode["steps_used"] else len(mismatched_keys) or 1,
        "final_state_matches_archive": final_matches,
        "mismatched_final_state_keys": mismatched_keys,
    }
    print("=== replay verification ===")
    for key, value in report.items():
        print(f"  {key}: {value}")

    if args.mode == "replay-verify":
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "replay_verification.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        if final_matches and stepper.step_index == episode["steps_used"]:
            print("replay-verify: PASS")
            return 0
        print("replay-verify: FAIL (final state mismatch)")
        return 1

    # ---- continue mode ------------------------------------------------------
    if not final_matches:
        print("refusing to continue from a divergent replay")
        return 1

    from interface.runner import _reset_agent_usage

    if args.agent:
        from interface.agents.reply import Reply
        agent = _ScriptedAgent(scripted_action_from_spec(args.agent), Reply)
    else:
        model_cfg = with_prompt_cache_default(
            {**run_inputs["model_config"], **(run_inputs.get("runtime_model_config") or {})},
            run_inputs["experiment_config"],
        )
        agent = build_live_agent(model_cfg, timeout=args.timeout, spend_cap_usd=args.spend_cap_usd)

    new_queries, stopped_early = drive_continue_loop(
        stepper,
        agent,
        max_new_queries=args.max_new_queries,
        reset_usage=_reset_agent_usage,
    )

    result = stepper.result()
    if stopped_early:
        result["end_reason"] = f"resume_interrupted:{stopped_early}"
    # Re-attach archived image payloads so the old flush path re-writes the
    # replayed queries' images instead of silently dropping the references.
    missing_images = rehydrate_agent_messages(result["transcript"], archive_dir)
    if missing_images:
        print(f"  note: {missing_images} archived query image(s) were absent from "
              "the archive and recorded as text placeholders")
    episode_path = flush_episode_log(result, out_dir)

    print("=== continue outcome ===")
    print(f"  end_reason: {result['end_reason']}")
    print(f"  success: {result['success']}")
    print(f"  steps_used: {result['steps_used']} (archived at {episode['steps_used']})")
    print(f"  query_count: {result['query_count']} (+{new_queries} new)")
    print(f"  artifacts: {episode_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
