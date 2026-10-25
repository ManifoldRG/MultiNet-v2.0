"""Unit tests for the pure (worktree-independent) helpers in
scripts/resume_from_archive.py. The replay itself runs against an as-run
worktree and is exercised by the script's own replay-verify gate."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from scripts.resume_from_archive import (
    IN_PROGRESS_END_REASON,
    SET_VALUED_STATE_FIELDS,
    build_checkpoint_payload,
    build_live_agent,
    canonical_state,
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
    ["stalled", "truncated", "max_steps", "wrong_done", "success", None],
)
def test_build_checkpoint_payload_refuses_model_outcomes(end_reason):
    with pytest.raises(SystemExit):
        build_checkpoint_payload({"end_reason": end_reason, "task_spec": {"seed": 0}, "transcript": []})


@pytest.mark.parametrize(
    "end_reason",
    [
        "resume_interrupted:max_new_queries(200)",
        "resume_interrupted:agent_error:RuntimeError: Moonshot API timed out after 600s.",
        "resume_interrupted:scripted_agent_exhausted",
    ],
)
def test_build_checkpoint_payload_clears_an_interrupted_resume(end_reason):
    """continue mode stops on its query cap or an agent error and writes
    resume_interrupted:<why>: the driver stopped, not the model, so the
    archive resumes again."""
    episode = {
        "end_reason": end_reason,
        "task_spec": {"seed": 3},
        "transcript": [{"kind": "reset"}, _query(1, True), _step(1), _query(2, True), _step(2)],
    }
    payload = build_checkpoint_payload(episode)
    assert (payload["query_count"], payload["parse_failures"]) == (2, 0)
    assert payload["finished"] is False
    assert payload["end_reason"] == IN_PROGRESS_END_REASON


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
# 3D archives: the driver rebuilds the backend the archive recorded, and
# refuses one this worktree cannot draw identically (other render version,
# other mujoco, or a render block the two files disagree on).
# --------------------------------------------------------------------------

_RENDER_3D = {"backend": "mujoco3d", "camera": "top_down", "resolution": "grid"}


def _parse_failed_archive(
    tmp_path: Path, render: dict | None = None, maze: str = "V01_empty_room.json", actions=("MOVE_FORWARD",),
) -> Path:
    """A real infra-killed (parse_failed) episode archive, 2D unless ``render``:
    ``actions`` answer the first queries, then the network is out."""
    import itertools
    import json

    from interface.loader import default_maze_path
    from scripts.run_pipeline import run_from_config

    replies = itertools.chain((f"FINAL_OUTPUT: {a}" for a in actions), itertools.repeat("(network outage)"))

    class _DyingAgent:
        last_usage = None

        def __call__(self, messages):
            self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
            return next(replies)

    cfg = tmp_path / "run_config.json"
    cfg.write_text(json.dumps({
        **({} if render is None else {"render": render}),
        "models": {"stub": {
            "provider": "claude", "model": "stub-model", "tasks": [str(default_maze_path(maze))],
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


def _resume(archive: Path, out_dir: Path, *extra: str, mode: str = "replay-verify", env: dict | None = None):
    import os
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "resume_from_archive.py"), "--archive-dir", str(archive),
         "--repo-root", str(repo), "--out-dir", str(out_dir), "--mode", mode, *extra],
        capture_output=True, text=True, timeout=300, env=None if env is None else {**os.environ, **env},
    )


def test_resume_replays_a_2d_archive(tmp_path: Path):
    result = _resume(_parse_failed_archive(tmp_path), tmp_path / "out")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "replay-verify: PASS" in result.stdout
    assert "backend: minigrid" in result.stdout


def test_canonical_state_ignores_the_order_of_set_valued_fields():
    archived = {"collected_keys": ["k_red", "k_blue"], "visible_cells": [[2, 1], [1, 1]], "agent_position": [1, 1],
                "rotator_dirs": [3, 0]}
    replayed = {"collected_keys": ["k_blue", "k_red"], "visible_cells": [[1, 1], [2, 1]], "agent_position": [1, 1],
                "rotator_dirs": [3, 0]}
    assert archived != replayed
    assert canonical_state(archived) == canonical_state(replayed)
    # Ordered fields keep their order.
    assert canonical_state({**replayed, "rotator_dirs": [0, 3]}) != canonical_state(archived)


def test_set_valued_state_fields_mirror_the_checkpoint_replay():
    from interface.episode_checkpoint import _SET_VALUED_STATE_FIELDS, _canonical_state

    assert SET_VALUED_STATE_FIELDS == _SET_VALUED_STATE_FIELDS
    state = {key: [["b"], ["a"]] for key in SET_VALUED_STATE_FIELDS}
    assert canonical_state(state) == _canonical_state(state)


def _two_key_archive(tmp_path: Path) -> tuple[Path, list[str]]:
    """A parse_failed archive killed after both of V08's keys were collected."""
    import json

    from gridworld.baselines import plan_bfs_path
    from interface.loader import default_maze_path
    from interface.parser import ACTION_ORDER
    from scorer.io import task_spec_from_payload

    plan = plan_bfs_path(task_spec_from_payload(json.loads(default_maze_path("V08_chain_kk.json").read_text())))
    archive = _parse_failed_archive(tmp_path, maze="V08_chain_kk.json",
                                    actions=[ACTION_ORDER[a] for a in plan.actions][:-1])
    keys = json.loads((archive / "episode.json").read_text())["final_state"]["collected_keys"]
    assert len(keys) == 2, "the plan must collect both keys before the kill"
    return archive, keys


@pytest.mark.parametrize("order", [sorted, lambda keys: sorted(keys, reverse=True)])
def test_resume_verifies_an_archive_whatever_order_it_listed_set_fields_in(tmp_path: Path, order):
    """Archives written before set fields were sorted list them in their
    process's hash order; the final-state check must compare them as sets."""
    import json

    archive, keys = _two_key_archive(tmp_path)
    episode = json.loads((archive / "episode.json").read_text())
    episode["final_state"]["collected_keys"] = order(keys)
    (archive / "episode.json").write_text(json.dumps(episode))
    # A fixed hash seed for the replay, so one of the two orders is not the one
    # its process would list: the check must not depend on it.
    result = _resume(archive, tmp_path / "out", env={"PYTHONHASHSEED": "0"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "replay-verify: PASS" in result.stdout


def test_an_interrupted_resume_writes_an_archive_that_resumes_again(tmp_path: Path):
    """A capped continue run (here the offline scripted agent's two queries)
    ends resume_interrupted:...; its out dir must itself be a resumable
    archive: replay-verify passes and another continue picks up from it."""
    import json

    first, second = tmp_path / "first", tmp_path / "second"
    archive = _parse_failed_archive(tmp_path)
    run = _resume(archive, first, "--agent", "scripted:TURN_LEFT", mode="continue")
    assert run.returncode == 0, run.stdout + run.stderr
    episode = json.loads((first / "episode.json").read_text())
    assert episode["end_reason"] == "resume_interrupted:scripted_agent_exhausted"
    assert (first / "run_inputs.json").read_bytes() == (archive / "run_inputs.json").read_bytes()

    verify = _resume(first, tmp_path / "verify")
    assert verify.returncode == 0, verify.stdout + verify.stderr
    assert "replay-verify: PASS" in verify.stdout

    run = _resume(first, second, "--agent", "scripted:TURN_LEFT", mode="continue")
    assert run.returncode == 0, run.stdout + run.stderr
    assert "absent from the archive" not in run.stdout  # the replayed queries kept their images
    again = json.loads((second / "episode.json").read_text())
    assert again["query_count"] == episode["query_count"] + 2
    assert again["steps_used"] == episode["steps_used"] + 2


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
