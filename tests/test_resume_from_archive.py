"""Unit tests for the pure (worktree-independent) helpers in
scripts/resume_from_archive.py. The replay itself runs against an as-run
worktree and is exercised by the script's own replay-verify gate."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from scripts.resume_from_archive import (
    IN_PROGRESS_END_REASON,
    INFRA_STOP_REASONS,
    build_checkpoint_payload,
    build_live_agent,
    drive_continue_loop,
    rehydrate_agent_messages,
    scripted_action_from_spec,
    trailing_parse_failures,
    unit_payload,
    with_prompt_cache_default,
)


def _query(idx: int, parse_ok: bool, stop_reason: str | None = None) -> dict:
    rec = {"kind": "query", "query_index": idx, "parse_ok": parse_ok, "agent_messages": []}
    if stop_reason is not None:
        rec["stop_reason"] = stop_reason
    return rec


def _step(idx: int, failures_after: int = 0) -> dict:
    return {"kind": "step", "step_index": idx, "consecutive_failures_after": failures_after}


def test_trailing_parse_failures_counts_only_the_trailing_run():
    records = [_query(1, False), _query(2, True), _query(3, False), _query(4, False)]
    assert trailing_parse_failures(records) == 2
    # A successful final parse resets the live counter to zero, even with
    # earlier mid-episode outage blips (queries 13/15/68 in the R1 archive).
    assert trailing_parse_failures([_query(1, False), _query(2, True)]) == 0
    assert trailing_parse_failures([]) == 0


def test_trailing_parse_failures_skips_infra_replies():
    # The 3D Opus run: three deadline-canceled batch rounds in a row, counted as
    # parse failures by code that predated batch_canceled being infra, ended a
    # healthy episode parse_failed. Infra replies neither count nor reset.
    canceled = [_query(i, False, "batch_canceled") for i in (2, 3, 4)]
    assert trailing_parse_failures([_query(1, True), *canceled]) == 0
    assert trailing_parse_failures([_query(1, False), *canceled]) == 1
    assert trailing_parse_failures(
        [_query(1, True), _query(2, False), _query(3, False, "batch_expired"), _query(4, False)]
    ) == 2


def test_infra_stop_reasons_match_the_stepper():
    from interface.episode_step import _INFRA_STOP_REASONS

    assert INFRA_STOP_REASONS == _INFRA_STOP_REASONS


def test_build_checkpoint_payload_clears_a_batch_canceled_kill():
    episode = {
        "end_reason": "parse_failed",
        "task_spec": {"seed": 0},
        "transcript": [
            {"kind": "reset"}, _query(1, True), _step(1),
            *[_query(i, False, "batch_canceled") for i in (2, 3, 4)],
        ],
    }
    payload = build_checkpoint_payload(episode)
    assert payload["query_count"] == 4
    assert payload["parse_failures"] == 0


def test_build_checkpoint_payload_clears_the_infra_kill():
    episode = {
        "end_reason": "parse_failed",
        "task_spec": {"seed": 7},
        "transcript": [
            {"kind": "reset"},
            _query(1, True),
            _step(1, failures_after=1),
            _query(2, False),
        ],
    }
    payload = build_checkpoint_payload(episode)
    assert payload["seed"] == 7
    assert payload["query_count"] == 2
    assert payload["parse_failures"] == 1
    assert payload["consecutive_failures"] == 1
    assert payload["finished"] is False
    assert payload["end_reason"] == IN_PROGRESS_END_REASON


def test_build_checkpoint_payload_refuses_finished_episodes():
    episode = {"end_reason": "success", "task_spec": {"seed": 0}, "transcript": []}
    with pytest.raises(SystemExit):
        build_checkpoint_payload(episode)


def test_build_checkpoint_payload_clears_an_agent_error_kill():
    """R1-frontier: OpenAI billing exhaustion ended episodes mid-run. The
    runner rolls back the failed query's index, so the transcript ends at a
    clean query boundary with no trailing parse failures."""
    episode = {
        "end_reason": "agent_error:RuntimeError: OpenAI API quota exhausted "
                      "(HTTP 429 insufficient_quota): {",
        "task_spec": {"seed": 0},
        "transcript": [{"kind": "reset"}, _query(1, True), _step(1), _query(2, True), _step(2)],
    }
    payload = build_checkpoint_payload(episode)
    assert payload["query_count"] == 2
    assert payload["parse_failures"] == 0
    assert payload["finished"] is False
    assert payload["end_reason"] == IN_PROGRESS_END_REASON


@pytest.mark.parametrize(
    "end_reason",
    ["stalled", "truncated", "max_steps", "wrong_done", "resume_interrupted:agent_error:X", None],
)
def test_build_checkpoint_payload_refuses_model_outcomes(end_reason):
    with pytest.raises(SystemExit):
        build_checkpoint_payload({"end_reason": end_reason, "task_spec": {"seed": 0}, "transcript": []})


_OPENAI_RUN_CFG = {
    "provider": "openai", "model": "gpt-6-astra", "max_tokens": 64000,
    "reasoning_effort": "xhigh", "service_tier": "flex", "image_detail": "high",
    "timeout": 1800, "max_attempts": 30, "spend_cap_usd": 150,
    "group": "openai-api", "worker_count": 1, "max_in_flight": 50, "tasks": ["all"],
}


def test_build_live_agent_openai_keeps_archived_params_and_overrides_the_cap(monkeypatch):
    from interface.agents.openai_agent import OpenAIAgent

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    agent = build_live_agent(_OPENAI_RUN_CFG, spend_cap_usd=13.0, log=lambda _m: None)
    assert isinstance(agent, OpenAIAgent)
    c = agent.config
    assert (c.model, c.max_tokens, c.reasoning_effort, c.service_tier, c.image_detail) == (
        "gpt-6-astra", 64000, "xhigh", "flex", "high",
    )
    assert (c.timeout, c.max_attempts) == (1800, 30)
    # Never the archived per-run cap: this process's ledger starts at $0.
    assert c.spend_cap_usd == 13.0


def test_build_live_agent_openai_requires_a_spend_cap(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    with pytest.raises(SystemExit, match="spend-cap-usd"):
        build_live_agent(_OPENAI_RUN_CFG, spend_cap_usd=None, log=lambda _m: None)


def test_build_live_agent_timeout_override_is_client_patience_only(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    agent = build_live_agent(_OPENAI_RUN_CFG, spend_cap_usd=5, timeout=99, log=lambda _m: None)
    assert agent.config.timeout == 99
    assert agent.config.reasoning_effort == "xhigh"


def test_build_live_agent_kimi_path_unchanged(monkeypatch):
    from interface.agents.kimi_k26 import KimiK26Agent

    monkeypatch.setenv("MOONSHOT_API_KEY", "k")
    agent = build_live_agent(
        {"provider": "kimi", "model": "kimi-k2.6", "max_tokens": 64000, "temperature": 1.0,
         "enable_thinking": True, "timeout": 2400, "group": "kimi-api"},
        log=lambda _m: None,
    )
    assert isinstance(agent, KimiK26Agent)
    assert (agent.config.enable_thinking, agent.config.timeout) == (True, 2400)


def test_build_live_agent_rejects_unknown_provider():
    with pytest.raises(SystemExit, match="provider"):
        build_live_agent({"provider": "qwen_vllm_api"}, log=lambda _m: None)


_CLAUDE_RUN_CFG = {
    "provider": "claude", "model": "claude-opus-5-5", "max_tokens": 64000, "effort": "xhigh",
    "enable_thinking": True, "enable_prompt_cache": True, "timeout": 600, "spend_cap_usd": 200,
    "group": "claude-api", "worker_role": "lockstep-worker", "worker_concurrency": 15,
    "max_in_flight": 64, "tasks": ["all"],
}


def test_build_live_agent_claude_keeps_archived_params_and_overrides_the_cap(monkeypatch):
    from interface.agents.claude import ClaudeAnthropicAgent

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    agent = build_live_agent(_CLAUDE_RUN_CFG, spend_cap_usd=12.5, timeout=1200, log=lambda _m: None)
    assert isinstance(agent, ClaudeAnthropicAgent)
    c = agent.config
    assert (c.model, c.max_tokens, c.effort, c.enable_thinking, c.enable_prompt_cache) == (
        "claude-opus-5-5", 64000, "xhigh", True, True,
    )
    # The sync path's cache TTL; the batch knobs are unused by generate().
    assert c.cache_ttl == "5m"
    assert c.timeout == 1200
    assert c.spend_cap_usd == 12.5


def test_build_live_agent_claude_requires_a_spend_cap(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    with pytest.raises(SystemExit, match="spend-cap-usd"):
        build_live_agent(_CLAUDE_RUN_CFG, spend_cap_usd=None, log=lambda _m: None)


def test_build_live_agent_refuses_an_agent_without_a_cap_field(monkeypatch):
    import dataclasses

    from interface.agents import claude

    @dataclasses.dataclass
    class _Uncapped:
        model: str = "m"

    monkeypatch.setattr(claude, "ClaudeAnthropicConfig", _Uncapped)
    with pytest.raises(SystemExit, match="no spend_cap_usd"):
        build_live_agent(_CLAUDE_RUN_CFG, spend_cap_usd=5, log=lambda _m: None)


@pytest.mark.parametrize("provider", ["claude", "openai"])
def test_prompt_cache_defaults_from_the_experiment(provider):
    assert with_prompt_cache_default({"provider": provider}, {"prompt_cache": True})[
        "enable_prompt_cache"
    ] is True
    assert with_prompt_cache_default({"provider": provider}, {})["enable_prompt_cache"] is False
    # An explicit entry wins.
    assert with_prompt_cache_default(
        {"provider": provider, "enable_prompt_cache": False}, {"prompt_cache": True}
    )["enable_prompt_cache"] is False


def test_prompt_cache_default_leaves_kimi_alone():
    assert "enable_prompt_cache" not in with_prompt_cache_default({"provider": "kimi"}, {"prompt_cache": True})


def test_rehydrate_restores_present_images_and_placeholders_missing(tmp_path: Path):
    qdir = tmp_path / "queries" / "query_001"
    qdir.mkdir(parents=True)
    (qdir / "have.png").write_bytes(b"png-bytes")
    transcript = [
        {
            "kind": "query",
            "query_index": 1,
            "agent_messages": [
                {"role": "system", "content": "plain string is untouched"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "image", "file": "have.png"},
                        {"type": "image", "file": "lost.png"},
                    ],
                },
            ],
        }
    ]
    missing = rehydrate_agent_messages(transcript, tmp_path)
    assert missing == 1
    blocks = transcript[0]["agent_messages"][1]["content"]
    assert blocks[0] == {"type": "text", "text": "hello"}
    expected_b64 = base64.b64encode(b"png-bytes").decode("ascii")
    assert blocks[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + expected_b64},
    }
    assert blocks[2]["type"] == "text" and "lost.png" in blocks[2]["text"]


def test_scripted_action_spec_parsing():
    assert scripted_action_from_spec("scripted:MOVE_FORWARD") == "MOVE_FORWARD"
    with pytest.raises(SystemExit):
        scripted_action_from_spec("MOVE_FORWARD")
    with pytest.raises(SystemExit):
        scripted_action_from_spec("scripted:")


# --------------------------------------------------------------------------
# drive_continue_loop: duck-typed against the old code's stepper/agent, so it
# is testable from the main checkout with fakes.
# --------------------------------------------------------------------------

class _FakeState:
    step_count = 0


class _FakeStepper:
    """Minimal stand-in: yields `rounds` query rounds, then None."""

    def __init__(self, rounds: int) -> None:
        self.rounds = rounds
        self.query_count = 0
        self.applied: list[object] = []
        self.action_queue: list[str] = []
        self.state = _FakeState()

    def next_query(self):
        if self.query_count >= self.rounds:
            return None
        self.query_count += 1  # the real stepper reserves the index here
        return [{"role": "user", "content": "go"}]

    def apply_reply(self, reply):
        self.applied.append(reply)


class _FakeAgent:
    def __init__(self, replies=(), raises_on: int | None = None) -> None:
        self.calls = 0
        self.raises_on = raises_on

    def generate(self, messages):
        self.calls += 1
        if self.raises_on is not None and self.calls == self.raises_on:
            raise RuntimeError("Moonshot API timed out after 600s")
        return f"reply-{self.calls}"


def test_continue_loop_runs_to_exhaustion_of_the_stepper():
    stepper, agent = _FakeStepper(rounds=3), _FakeAgent()
    new_queries, stopped_early = drive_continue_loop(
        stepper, agent, max_new_queries=10, log=lambda _m: None
    )
    assert (new_queries, stopped_early) == (3, None)
    assert stepper.applied == ["reply-1", "reply-2", "reply-3"]
    assert stepper.query_count == 3


def test_continue_loop_stops_at_the_new_query_cap_and_rolls_the_index_back():
    stepper, agent = _FakeStepper(rounds=10), _FakeAgent()
    new_queries, stopped_early = drive_continue_loop(
        stepper, agent, max_new_queries=2, log=lambda _m: None
    )
    assert new_queries == 2
    assert stopped_early == "max_new_queries(2)"
    # query_count equals the APPLIED query count: the reserved-but-unused
    # index is rolled back.
    assert stepper.query_count == 2


def test_continue_loop_flushes_instead_of_crashing_on_an_agent_error():
    """A transport failure must never discard hours of paid work: the loop
    reports it as an interruption so main() still writes episode artifacts."""
    stepper, agent = _FakeStepper(rounds=10), _FakeAgent(raises_on=3)
    new_queries, stopped_early = drive_continue_loop(
        stepper, agent, max_new_queries=10, log=lambda _m: None
    )
    assert new_queries == 2  # the two that applied cleanly
    assert stopped_early.startswith("agent_error:RuntimeError: Moonshot API timed out")
    assert stepper.query_count == 2
    assert stepper.applied == ["reply-1", "reply-2"]


def test_continue_loop_stops_when_a_scripted_agent_reports_exhausted():
    class _Exhausted(_FakeAgent):
        exhausted = True

    stepper = _FakeStepper(rounds=5)
    new_queries, stopped_early = drive_continue_loop(
        stepper, _Exhausted(), max_new_queries=10, log=lambda _m: None
    )
    assert (new_queries, stopped_early) == (0, "scripted_agent_exhausted")
    assert stepper.query_count == 0


def test_continue_loop_checkpoints_before_every_query_it_sends():
    stepper, agent = _FakeStepper(rounds=10), _FakeAgent(raises_on=3)
    seen: list[tuple[int, int]] = []
    drive_continue_loop(
        stepper, agent, max_new_queries=10, log=lambda _m: None,
        checkpoint=lambda s: seen.append((s.query_count, agent.calls)),
    )
    # Taken with the query's index reserved and before it is sent, including the
    # one whose send failed; never after the loop stopped.
    assert seen == [(1, 0), (2, 1), (3, 2)]


def test_continue_loop_skips_the_checkpoint_for_a_round_it_does_not_send():
    stepper, seen = _FakeStepper(rounds=10), []
    drive_continue_loop(
        stepper, _FakeAgent(), max_new_queries=2, log=lambda _m: None,
        checkpoint=lambda s: seen.append(s.query_count),
    )
    assert seen == [1, 2]


# --------------------------------------------------------------------------
# --unit-id: resume an in-flight lockstep unit from its live checkpoint.
# --------------------------------------------------------------------------

def _plan_with_task_files(tmp_path: Path) -> tuple[dict, Path]:
    import json

    root = tmp_path / "worker"
    task_dir = root / "tasks" / "t1"
    task_dir.mkdir(parents=True)
    (task_dir / "canonical_paths.json").write_text(json.dumps({"paths": [1]}))
    (task_dir / "scored_static.json").write_text(json.dumps({"score": 2}))
    plan = {
        "scorer_config": {"s": 1},
        "difficulty_max_static_score": 99.0,
        "units": [{"unit_id": "u1", "task_id": "t1", "task_payload": {"maze": 3}}],
    }
    return plan, root


def test_unit_payload_rebuilds_the_assign_payload(tmp_path: Path):
    plan, root = _plan_with_task_files(tmp_path)
    unit = unit_payload(plan, "u1", root)
    assert unit["scorer_config"] == {"s": 1}
    assert unit["difficulty_max_static_score"] == 99.0
    assert unit["task_artifacts"] == {
        "canonical_paths": {"paths": [1]}, "scored_static": {"score": 2}, "task_payload": {"maze": 3},
    }
    assert "task_artifacts" not in plan["units"][0]  # the plan is not mutated


def test_unit_payload_refuses_unknown_units_and_missing_task_files(tmp_path: Path):
    plan, root = _plan_with_task_files(tmp_path)
    with pytest.raises(SystemExit, match="not in the job plan"):
        unit_payload(plan, "nope", root)
    with pytest.raises(SystemExit, match="task artifacts"):
        unit_payload(plan, "u1", tmp_path / "empty")


def _seed_lockstep_checkpoint(tmp_path: Path):
    """A coordinator plan plus the live checkpoint a lockstep worker leaves at
    corr_c's first query boundary, built through the worker's own setup path."""
    import json

    import scripts.distributed_run_pipeline as drp
    from interface.config import ExperimentConfig
    from interface.episode_checkpoint import save_checkpoint
    from interface.episode_step import EpisodeStepper
    from pipeline.run_stage3 import build_episode_runner
    from scorer.config import ScorerConfig
    from tests.test_lockstep_worker import _write_job

    artifacts, plan = _write_job(tmp_path)
    store = drp.CoordinatorStore(artifacts)
    unit_raw = next(u for u in plan["units"] if u["task_id"] == "corr_c")
    unit = store._unit_payload(plan, unit_raw)
    worker_root = tmp_path / "worker_root"
    row = drp.materialize_worker_inputs(unit, worker_root)
    prep = drp.pipeline._prepare_unit_run(
        row,
        unit["model_id"],
        model_config=unit["model_config"],
        manifest_path=worker_root / "distributed" / "worker_manifest.json",
        artifacts_root=worker_root,
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
        prep.source, prep.experiment_config, int(unit["seed"]), max_steps=prep.runtime_spec.max_steps,
    )
    stepper = EpisodeStepper(runner, maze_path=str(prep.source))
    stepper.start()
    stepper.next_query()  # the first query boundary (empty buffers)
    save_checkpoint(prep.run_dir / "checkpoint.json", stepper)
    plan_path = artifacts / "distributed" / "job_plan.json"
    assert json.loads(plan_path.read_text())["units"]  # prepare_job wrote the plan here
    return store, unit, plan_path, worker_root


def _resume_unit(plan_path: Path, unit_id: str, worker_root: Path, out_dir: Path, *extra: str):
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "resume_from_archive.py"), "--unit-id", unit_id,
         "--job-plan", str(plan_path), "--worker-root", str(worker_root), "--repo-root", str(repo),
         "--out-dir", str(out_dir), *extra],
        capture_output=True, text=True, timeout=300,
    )


def test_unit_resume_replays_a_live_lockstep_checkpoint(tmp_path: Path):
    _store, unit, plan_path, worker_root = _seed_lockstep_checkpoint(tmp_path)
    result = _resume_unit(plan_path, unit["unit_id"], worker_root, tmp_path / "out", "--mode", "replay-verify")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "replay-verify: PASS" in result.stdout
    assert f"inputs_hash: {unit['episode_inputs_hash']}" in result.stdout


def test_unit_resume_continues_checkpoints_and_finishes_into_a_verified_upload(tmp_path: Path):
    """corr_c needs three MOVE_FORWARDs; the scripted agent sends two per process.
    The first resume stops with a checkpoint under --out-dir, the second continues
    from it, finishes, and packages an archive the coordinator verifies."""
    import json

    store, unit, plan_path, worker_root = _seed_lockstep_checkpoint(tmp_path)
    out = tmp_path / "out"
    resumable = out / unit["run_dir"] / "checkpoint.json"
    live = worker_root / unit["run_dir"] / "checkpoint.json"
    live_before = live.read_text()

    first = _resume_unit(plan_path, unit["unit_id"], worker_root, out,
                         "--mode", "continue", "--agent", "scripted:MOVE_FORWARD")
    assert first.returncode == 3, first.stdout + first.stderr
    assert "scripted_agent_exhausted" in first.stdout
    assert json.loads(resumable.read_text())["query_count"] == 2
    assert live.read_text() == live_before  # the worker's checkpoint is never written

    second = _resume_unit(plan_path, unit["unit_id"], worker_root, out, "--mode", "continue",
                          "--agent", "scripted:MOVE_FORWARD", "--checkpoint", str(resumable))
    assert second.returncode == 0, second.stdout + second.stderr
    assert "success: True" in second.stdout
    assert not resumable.exists()
    run_dir = out / unit["run_dir"]
    run_inputs = json.loads((run_dir / "run_inputs.json").read_text())
    assert run_inputs["inputs_hash"] == unit["episode_inputs_hash"]
    assert (run_inputs["pricing_tier"], run_inputs["resumed_from_query"]) == ("standard", 2)
    assert (run_dir / "run_score.json").is_file()

    archive = out / "distributed" / f"{unit['unit_id']}.tar.gz"
    store.upload("w-resume", unit["unit_id"], archive.read_bytes())
    state = json.loads((store.artifacts_root / "distributed" / "job_state.json").read_text())
    assert state["units"][unit["unit_id"]]["status"] == "verified"


def test_unit_resume_refuses_a_worktree_that_moves_the_inputs_hash(tmp_path: Path):
    import json

    _store, unit, plan_path, worker_root = _seed_lockstep_checkpoint(tmp_path)
    plan = json.loads(plan_path.read_text())
    for u in plan["units"]:
        if u["unit_id"] == unit["unit_id"]:
            u["episode_inputs_hash"] = "0" * 64
    plan_path.write_text(json.dumps(plan))
    result = _resume_unit(plan_path, unit["unit_id"], worker_root, tmp_path / "out", "--mode", "replay-verify")
    assert result.returncode != 0
    assert "refusing to resume" in result.stderr and "inputs_hash" in result.stderr


def test_unit_resume_refuses_an_out_dir_inside_the_worker_root(tmp_path: Path):
    _store, unit, plan_path, worker_root = _seed_lockstep_checkpoint(tmp_path)
    result = _resume_unit(plan_path, unit["unit_id"], worker_root, worker_root / "out", "--mode", "replay-verify")
    assert result.returncode != 0
    assert "--out-dir must not point into --worker-root" in result.stderr


# --------------------------------------------------------------------------
# 3D archives: the driver rebuilds the backend the archive recorded, and
# refuses one this worktree cannot draw identically (other render version,
# other mujoco, or a render block the two files disagree on).
# --------------------------------------------------------------------------

_RENDER_3D = {"backend": "mujoco3d", "camera": "top_down", "resolution": "grid"}


def _parse_failed_archive(tmp_path: Path, render: dict | None = None) -> Path:
    """A real infra-killed (parse_failed) episode archive, 2D unless ``render``."""
    import itertools
    import json

    from interface.loader import default_maze_path
    from scripts.run_pipeline import run_from_config

    replies = itertools.chain(["FINAL_OUTPUT: MOVE_FORWARD"], itertools.repeat("(network outage)"))

    class _DyingAgent:
        last_usage = None

        def __call__(self, messages):
            self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
            return next(replies)

    cfg = tmp_path / "run_config.json"
    cfg.write_text(json.dumps({
        **({} if render is None else {"render": render}),
        "models": {"stub": {
            "provider": "claude", "model": "stub-model", "tasks": [str(default_maze_path("V01_empty_room.json"))],
        }},
    }))
    artifacts = tmp_path / "artifacts"
    run_from_config(
        run_config_path=cfg, manifest_path=Path(__file__).resolve().parents[1] / "gridworld/fixtures/manifest.json",
        seeds=[0], artifacts_root=artifacts, run_set_id="r", agent_factory=lambda n, c: (_DyingAgent(), c["model"]),
        difficulty_max_static_score=1000.0,
    )
    [archive] = [p.parent for p in artifacts.rglob("episode.json")]
    assert json.loads((archive / "episode.json").read_text())["end_reason"] == "parse_failed"
    return archive


def _edit_render(archive: Path, **changes) -> None:
    import json

    for name in ("episode.json", "run_inputs.json"):
        payload = json.loads((archive / name).read_text())
        payload["render"] = {**payload.get("render", {}), **changes}
        (archive / name).write_text(json.dumps(payload))


def _resume(archive: Path, out_dir: Path):
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "resume_from_archive.py"), "--archive-dir", str(archive),
         "--repo-root", str(repo), "--out-dir", str(out_dir), "--mode", "replay-verify"],
        capture_output=True, text=True, timeout=300,
    )


def test_resume_replays_a_2d_archive(tmp_path: Path):
    result = _resume(_parse_failed_archive(tmp_path), tmp_path / "out")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "replay-verify: PASS" in result.stdout
    assert "backend: minigrid" in result.stdout


def test_resume_replays_a_3d_archive_on_the_3d_backend(tmp_path: Path):
    pytest.importorskip("mujoco")
    from gridworld.render3d import RENDER3D_VERSION

    result = _resume(_parse_failed_archive(tmp_path, _RENDER_3D), tmp_path / "out")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "replay-verify: PASS" in result.stdout
    assert f"backend: mujoco3d_top_down_grid_r{RENDER3D_VERSION}" in result.stdout


@pytest.mark.parametrize("changes", [{"render3d_version": "0"}, {"mujoco_version": "0.0.0"}, {"camera": "first_person"}])
def test_resume_refuses_a_3d_archive_this_worktree_would_draw_differently(tmp_path: Path, changes):
    pytest.importorskip("mujoco")
    archive = _parse_failed_archive(tmp_path, _RENDER_3D)
    _edit_render(archive, **changes)
    result = _resume(archive, tmp_path / "out")
    assert result.returncode != 0
    assert "replay-verify: PASS" not in result.stdout
    assert "refusing to resume" in result.stderr and "render" in result.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "episode, run_inputs",
    [
        ({"render": _RENDER_3D}, {}),                        # the two files disagree
        ({}, {"render": _RENDER_3D}),
        ({"render": _RENDER_3D}, {"render": {**_RENDER_3D, "camera": "first_person"}}),
        ({}, {"backend": "mujoco3d_top_down_grid"}),          # a 3D label with no render block
    ],
)
def test_recorded_render_refuses_an_inconsistent_archive(episode, run_inputs):
    from scripts.resume_from_archive import recorded_render

    with pytest.raises(SystemExit, match="mujoco3d"):
        recorded_render(episode, run_inputs)


@pytest.mark.parametrize(
    "episode, run_inputs",
    [({}, {}), ({}, {"backend": "minigrid"}), ({"render": {"backend": "minigrid"}}, {"backend": "minigrid"})],
)
def test_recorded_render_is_none_for_2d_archives(episode, run_inputs):
    from scripts.resume_from_archive import recorded_render

    assert recorded_render(episode, run_inputs) is None


def test_recorded_render_returns_the_shared_3d_block():
    from scripts.resume_from_archive import recorded_render

    block = {**_RENDER_3D, "frame_pixels": 160}
    assert recorded_render({"render": block}, {"render": dict(block), "backend": "x"}) == block
