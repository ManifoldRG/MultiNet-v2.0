"""Unit tests for the pure (worktree-independent) helpers in
scripts/resume_from_archive.py. The replay itself runs against an as-run
worktree and is exercised by the script's own replay-verify gate."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from scripts.resume_from_archive import (
    IN_PROGRESS_END_REASON,
    build_checkpoint_payload,
    build_live_agent,
    drive_continue_loop,
    rehydrate_agent_messages,
    scripted_action_from_spec,
    trailing_parse_failures,
)


def _query(idx: int, parse_ok: bool) -> dict:
    return {"kind": "query", "query_index": idx, "parse_ok": parse_ok, "agent_messages": []}


def _step(idx: int, failures_after: int = 0) -> dict:
    return {"kind": "step", "step_index": idx, "consecutive_failures_after": failures_after}


def test_trailing_parse_failures_counts_only_the_trailing_run():
    records = [_query(1, False), _query(2, True), _query(3, False), _query(4, False)]
    assert trailing_parse_failures(records) == 2
    # A successful final parse resets the live counter to zero, even with
    # earlier mid-episode outage blips (queries 13/15/68 in the R1 archive).
    assert trailing_parse_failures([_query(1, False), _query(2, True)]) == 0
    assert trailing_parse_failures([]) == 0


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


# --------------------------------------------------------------------------
# 3D archives: the driver rebuilds a 2D MiniGrid backend, so a 3D episode
# would continue with 2D frames. It must refuse instead.
# --------------------------------------------------------------------------

_RENDER_3D = {"backend": "mujoco3d", "camera": "top_down", "resolution": "grid"}


def _parse_failed_archive(tmp_path: Path) -> Path:
    """A real infra-killed (parse_failed) 2D episode archive."""
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
    cfg.write_text(json.dumps({"models": {"stub": {
        "provider": "claude", "model": "stub-model", "tasks": [str(default_maze_path("V01_empty_room.json"))],
    }}}))
    artifacts = tmp_path / "artifacts"
    run_from_config(
        run_config_path=cfg, manifest_path=Path(__file__).resolve().parents[1] / "gridworld/fixtures/manifest.json",
        seeds=[0], artifacts_root=artifacts, run_set_id="r", agent_factory=lambda n, c: (_DyingAgent(), c["model"]),
        difficulty_max_static_score=1000.0,
    )
    [archive] = [p.parent for p in artifacts.rglob("episode.json")]
    assert json.loads((archive / "episode.json").read_text())["end_reason"] == "parse_failed"
    return archive


def _mark_3d(archive: Path) -> None:
    import json

    for name in ("episode.json", "run_inputs.json"):
        payload = json.loads((archive / name).read_text())
        payload["render"] = dict(_RENDER_3D)
        if name == "run_inputs.json":
            payload["backend"] = "mujoco3d_top_down_grid"
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


def test_resume_refuses_a_3d_archive_instead_of_continuing_in_2d(tmp_path: Path):
    archive = _parse_failed_archive(tmp_path)
    _mark_3d(archive)
    result = _resume(archive, tmp_path / "out")
    assert result.returncode != 0
    assert "replay-verify: PASS" not in result.stdout
    assert "3D" in result.stderr and "mujoco3d" in result.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "episode, run_inputs",
    [
        ({"render": _RENDER_3D}, {}),
        ({}, {"render": _RENDER_3D}),
        ({}, {"backend": "mujoco3d_chase_grid"}),
    ],
)
def test_refuse_non_2d_render_names_the_recorded_render(episode, run_inputs):
    from scripts.resume_from_archive import refuse_non_2d_render

    with pytest.raises(SystemExit, match="mujoco3d"):
        refuse_non_2d_render(episode, run_inputs)


@pytest.mark.parametrize(
    "episode, run_inputs",
    [({}, {}), ({}, {"backend": "minigrid"}), ({"render": {"backend": "minigrid"}}, {"backend": "minigrid"})],
)
def test_refuse_non_2d_render_accepts_2d_archives(episode, run_inputs):
    from scripts.resume_from_archive import refuse_non_2d_render

    assert refuse_non_2d_render(episode, run_inputs) is None
