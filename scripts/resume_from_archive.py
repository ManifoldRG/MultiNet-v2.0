#!/usr/bin/env python
"""Resume an infra-killed episode from its on-disk archive, under as-run code.

The R1 episode r1_M6_14x14_dense_kr_sg_kb_1 / kimi-k2.6 / seed_0 was killed at
env step 143 by a Moonshot network outage (terminal ``parse_failed`` produced
by empty replies, not by the model). Its archived ``episode.json`` transcript
has exactly the record shape that ``interface/episode_checkpoint.py`` persists
in a query-boundary checkpoint, so this driver:

1. inserts ``--repo-root`` (a git worktree of the as-run SHA) at ``sys.path[0]``
   BEFORE importing any harness module, so ``interface``/``gridworld`` come
   from the old code;
2. reconstructs the ``TaskSpecification`` from the ``task_spec`` EMBEDDED in
   ``episode.json`` (no manifests/submodules touched);
3. builds a synthetic checkpoint payload from the archive (query counter and
   trailing parse-failure count rebuilt from the query records; the terminal
   ``parse_failed`` status cleared) and hands it to the old code's
   ``resume_stepper``, which replays the recorded actions through a fresh
   backend, verifying position_after/state_after per step and rebuilding
   stall-watchdog state with the old code's ``_progress_signature``;
4. ``--mode replay-verify``: prints a verification report and exits 0 only if
   the replayed final state matches the archive's last recorded state;
5. ``--mode continue``: keeps running the live query loop (Kimi agent, sync
   transport, model config verbatim from ``run_inputs.json``) and flushes full
   episode artifacts to ``--out-dir`` via the old code's ``flush_episode_log``.

The archive directory is never written to; the worktree is never modified.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
from pathlib import Path

ARCHIVED_END_REASONS_TO_CLEAR = {"parse_failed"}
# The stepper's in-progress default (see EpisodeStepper.start()).
IN_PROGRESS_END_REASON = "max_steps"


# --------------------------------------------------------------------------
# Pure helpers (no harness imports; unit-testable from the main checkout).
# --------------------------------------------------------------------------

def trailing_parse_failures(query_records: list[dict]) -> int:
    """parse_failures resets to 0 on any successfully parsed reply, so the
    live counter equals the number of TRAILING consecutive parse_ok=False
    query records."""
    n = 0
    for rec in reversed(query_records):
        if rec.get("parse_ok"):
            break
        n += 1
    return n


def build_checkpoint_payload(episode: dict) -> dict:
    """A synthetic query-boundary checkpoint equivalent to what the as-run
    process would have saved just before the outage killed it, with the
    terminal parse_failed status cleared so the episode continues as if the
    lost query were re-issued."""
    transcript = episode["transcript"]
    query_records = [r for r in transcript if r.get("kind") == "query"]
    step_records = [r for r in transcript if r.get("kind") == "step"]
    end_reason = episode.get("end_reason")
    if end_reason not in ARCHIVED_END_REASONS_TO_CLEAR:
        raise SystemExit(
            f"archive end_reason={end_reason!r} is not an infra-kill this driver "
            f"knows how to clear (expected one of {sorted(ARCHIVED_END_REASONS_TO_CLEAR)}); "
            "refusing to resume a genuinely finished episode"
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


def drive_continue_loop(stepper, agent, *, max_new_queries: int, log=print, reset_usage=None):
    """Run live query rounds until the stepper finishes, the cap is hit, or the
    agent dies. Returns ``(new_queries, stopped_early)``; a non-None
    ``stopped_early`` becomes a ``resume_interrupted:`` end_reason.

    A transport failure (socket timeout past the retry budget, non-retryable
    HTTP status) is caught rather than propagated: crashing here would discard
    every paid query of a multi-hour resume, since artifacts are only flushed
    after this loop returns.

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


def scripted_action_from_spec(spec: str) -> str:
    prefix = "scripted:"
    if not spec.startswith(prefix) or not spec[len(prefix):]:
        raise SystemExit(f"--agent must look like scripted:<ACTION>, got {spec!r}")
    return spec[len(prefix):]


# --------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------

def _install_repo_root(repo_root: Path) -> None:
    """Make the as-run worktree the ONLY source of harness modules."""
    already = {"interface", "gridworld", "prompting_experiments", "scorer"}
    loaded = [m for m in sys.modules if m.split(".")[0] in already]
    if loaded:
        raise SystemExit(f"harness modules imported before path setup: {loaded}")
    resolved = str(repo_root.resolve())
    # Drop any entry that would shadow the worktree with the live checkout
    # (script dir, cwd, '' etc. all resolve under some other repo root).
    sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()) != resolved]
    sys.path.insert(0, resolved)


class _ScriptedAgent:
    """Offline stand-in for the Kimi agent: emits one fixed action in valid
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--archive-dir", required=True, type=Path,
                    help="the archived seed_<n>/<variant> episode dir (read-only)")
    ap.add_argument("--repo-root", required=True, type=Path,
                    help="worktree of the as-run SHA; inserted at sys.path[0]")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="fresh dir for resumed-episode artifacts")
    ap.add_argument("--mode", required=True, choices=["replay-verify", "continue"])
    ap.add_argument("--max-new-queries", type=int, default=200,
                    help="safety cap on NEW model queries in continue mode")
    ap.add_argument("--timeout", type=float, default=None,
                    help="override the archived per-call socket timeout (seconds). "
                         "Client patience only — no effect on sampling or on "
                         "comparability with the as-run episode. The archived 600s "
                         "predates the run_config default for 64k thinking runs "
                         "(the as-run process had the KIMI_TIMEOUT_OVERRIDE hack).")
    ap.add_argument("--agent", default=None,
                    help="override: scripted:<ACTION> emits that action for 2 "
                         "queries then stops (no API key / spend)")
    args = ap.parse_args()

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

    # Everything derives from the archive: experiment config (incl. stall K and
    # parse-retry caps) from run_inputs, the task (incl. the runtime-capped
    # max_steps) from the embedded task_spec.
    config = ExperimentConfig.from_dict(run_inputs["experiment_config"])
    task_spec = TaskSpecification.from_dict(episode["task_spec"])
    backend = MiniGridBackend(render_mode="rgb_array")
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
        from interface.agents.kimi_k26 import KimiK26Agent, KimiK26Config
        model_cfg = {**run_inputs["model_config"], **run_inputs["runtime_model_config"]}
        if args.timeout is not None:
            print(f"  timeout override: {model_cfg.get('timeout')} -> {args.timeout}s")
            model_cfg["timeout"] = args.timeout
        allowed = set(KimiK26Config.__dataclass_fields__)
        agent = KimiK26Agent(
            config=KimiK26Config(**{k: v for k, v in model_cfg.items() if k in allowed})
        )

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
