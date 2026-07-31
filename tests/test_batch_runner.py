"""Behavioural tests for ``interface.batch_runner.LockstepBatchRunner``.

The runner drives many ``EpisodeStepper`` episodes in lockstep over a batch API.
These tests pin: (1) ragged termination — episodes of different length finish at
different rounds, the working set shrinks, and no transcript is contaminated by
another episode's actions; (2) refill keeps the working set saturated without
ever exceeding ``max_batches``; (3) a batch-item stub reply lands as an ordinary
(recoverable) parse failure and does not touch sibling units; (4) the
``on_round`` heartbeat fires once per round with the active units; (5) a stepper
that raises is captured as an error without killing the batch; (6) checkpoints
are written only at the query boundary (empty buffers) and deleted on finish;
(7) an end-to-end serial-equivalence drift guard: the same scripts through the
batch runner produce byte-identical transcripts to ``ExperimentRunner.run``.
"""

from __future__ import annotations

import json

import pytest

from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.task_spec import TaskSpecification
from interface.agents.reply import Reply
from interface.batch_runner import LockstepBatchRunner, LockstepUnit
from interface.config import ExperimentConfig
from interface.episode_log import _json_safe
from interface.episode_step import EpisodeStepper
from interface.runner import build_runner


# --------------------------------------------------------------------------- #
# Test doubles + helpers
# --------------------------------------------------------------------------- #
class ScriptedBatchAgent:
    """A batch agent that routes each prompt to a per-unit scripted queue.

    ``scripts`` maps a unique marker string (a substring guaranteed to appear in
    exactly one unit's prompt — here the rendered "H by W grid" phrase) to that
    unit's ordered action tokens. Each round it emits one token per prompt from
    the matching queue (``DONE`` once a queue is exhausted). Because the mapping
    is keyed by prompt content, batch position is irrelevant — units may drop out
    in any order without misrouting a reply.
    """

    def __init__(self, scripts, *, usage=None):
        self._scripts = {k: list(v) for k, v in scripts.items()}
        self._cursor = {k: 0 for k in scripts}
        self._usage = usage
        self.batch_sizes = []

    def _key(self, messages):
        blob = json.dumps(messages, default=str)
        hits = [k for k in self._scripts if k in blob]
        if len(hits) != 1:
            raise AssertionError(f"expected exactly one marker in prompt, got {hits}")
        return hits[0]

    def generate_batch(self, batch):
        self.batch_sizes.append(len(batch))
        replies = []
        for messages in batch:
            key = self._key(messages)
            i = self._cursor[key]
            script = self._scripts[key]
            token = script[i] if i < len(script) else "DONE"
            self._cursor[key] = i + 1
            replies.append(
                Reply(text=f"FINAL_OUTPUT: {token}", usage=self._usage)
            )
        return replies


class ScriptedAgent:
    """Serial reference double: ``__call__(messages) -> str`` + usage side-channel."""

    def __init__(self, actions):
        self._actions = list(actions)
        self._i = 0
        self.last_usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}

    def __call__(self, messages):
        action = self._actions[self._i] if self._i < len(self._actions) else "DONE"
        self._i += 1
        return f"FINAL_OUTPUT: {action}"


def _spec(task_id, dims, start, goal, **extra):
    spec = {
        "task_id": task_id,
        "seed": 0,
        "difficulty_tier": 1,
        "maze": {"dimensions": list(dims), "walls": [], "start": list(start), "goal": list(goal)},
        "mechanisms": {},
        "goal": {"type": "reach_position", "target": list(goal)},
        "max_steps": 60,
    }
    spec.update(extra)
    return spec


def _make_unit(unit_id, spec_dict, *, checkpoint_path=None, **config_kwargs):
    spec = TaskSpecification.from_dict(spec_dict)
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    runner = build_runner(ExperimentConfig(**config_kwargs), backend, spec)
    return LockstepUnit(unit_id, EpisodeStepper(runner), checkpoint_path=checkpoint_path)


def _steps(result):
    return [e["action"] for e in result["transcript"] if e.get("kind") == "step"]


def _scrub(transcript):
    scrubbed = _json_safe(transcript)
    for rec in scrubbed:
        rec.pop("llm_latency_s", None)
    return scrubbed


# --------------------------------------------------------------------------- #
# 1. Ragged termination
# --------------------------------------------------------------------------- #
def test_ragged_termination():
    # Three egocentric episodes of increasing length with DISTINCT action
    # scripts and distinct "H by W grid" markers, so a misrouted reply would
    # corrupt a transcript. They finish after 2 / 5 / 8 queries.
    p = _make_unit("P", _spec("p", [5, 3], [1, 1], [3, 1]))          # "3 by 5"
    q = _make_unit("Q", _spec("q", [5, 5], [1, 1], [3, 3]))          # "5 by 5"
    r = _make_unit("R", _spec("r", [6, 7], [1, 1], [4, 5]))          # "7 by 6"

    scripts = {
        "3 by 5": ["MOVE_FORWARD", "MOVE_FORWARD"],
        "5 by 5": ["MOVE_FORWARD", "MOVE_FORWARD", "TURN_RIGHT", "MOVE_FORWARD", "MOVE_FORWARD"],
        "7 by 6": ["MOVE_FORWARD"] * 3 + ["TURN_RIGHT"] + ["MOVE_FORWARD"] * 4,
    }
    agent = ScriptedBatchAgent(scripts)

    round_sizes = []
    runner = LockstepBatchRunner(
        agent, max_batches=3, on_round=lambda active: round_sizes.append(len(active))
    )
    for u in (p, q, r):
        runner.add(u)
    results = runner.run()

    # All solved, with the expected query counts.
    assert {k: v["success"] for k, v in results.items()} == {"P": True, "Q": True, "R": True}
    assert results["P"]["query_count"] == 2
    assert results["Q"]["query_count"] == 5
    assert results["R"]["query_count"] == 8

    # Each transcript's executed actions are its OWN script (no cross-contamination).
    assert _steps(results["P"]) == scripts["3 by 5"]
    assert _steps(results["Q"]) == scripts["5 by 5"]
    assert _steps(results["R"]) == scripts["7 by 6"]

    # Working set only shrinks; later rounds are strictly smaller than the first.
    # (A trailing 0 is the silent all-finished heartbeat round.)
    assert round_sizes[0] == 3
    assert round_sizes[-1] == 0            # final silent heartbeat
    assert max(round_sizes) == 3
    assert [n for n in round_sizes if n > 0][-1] == 1   # last live round is R alone
    assert all(b <= a for a, b in zip(round_sizes, round_sizes[1:]))
    # generate_batch saw shrinking batches too.
    assert agent.batch_sizes[0] == 3
    assert agent.batch_sizes[-1] == 1


# --------------------------------------------------------------------------- #
# 2. Refill keeps the working set full
# --------------------------------------------------------------------------- #
def test_refill_keeps_working_set_full():
    # Pool of 4 straight corridors of lengths 1/2/3/4 fed via `refill`; the
    # working set holds at most 2. All four must complete.
    # (unit_id, dims, goal_x, marker); straight corridors from (1,1) east.
    specs = [
        ("A", [4, 3], 2, "3 by 4"),
        ("B", [5, 3], 3, "3 by 5"),
        ("C", [6, 3], 4, "3 by 6"),
        ("D", [7, 3], 5, "3 by 7"),
    ]
    pool = [
        _make_unit(uid, _spec(uid.lower(), dims, [1, 1], [goal_x, 1]))
        for uid, dims, goal_x, _marker in specs
    ]
    scripts = {marker: ["MOVE_FORWARD"] * (goal_x - 1) for _uid, _dims, goal_x, marker in specs}
    agent = ScriptedBatchAgent(scripts)

    pool_iter = iter(pool)

    def refill():
        return next(pool_iter, None)

    peak = []
    runner = LockstepBatchRunner(
        agent, max_batches=2, refill=refill,
        on_round=lambda active: peak.append(len(active)),
    )
    results = runner.run()

    assert set(results) == {"A", "B", "C", "D"}
    assert all(v["success"] for v in results.values())
    assert max(peak) <= 2
    assert max(agent.batch_sizes) <= 2


# --------------------------------------------------------------------------- #
# 3. Batch-item stub reply == ordinary (recoverable) parse failure
# --------------------------------------------------------------------------- #
class StubOnceAgent:
    """Returns a `batch_expired` stub for one marker on round 1, else MOVE_FORWARD."""

    def __init__(self, stub_marker):
        self._stub_marker = stub_marker
        self._round = 0

    def generate_batch(self, batch):
        self._round += 1
        replies = []
        for messages in batch:
            blob = json.dumps(messages, default=str)
            if self._round == 1 and self._stub_marker in blob:
                replies.append(Reply(text="", stop_reason="batch_expired"))
            else:
                replies.append(Reply(text="FINAL_OUTPUT: MOVE_FORWARD"))
        return replies


def test_batch_stub_reply_is_parse_failure():
    a = _make_unit("A", _spec("a", [6, 3], [1, 1], [4, 1]))   # "3 by 6"
    b = _make_unit("B", _spec("b", [7, 3], [1, 1], [4, 1]))   # "3 by 7" (stubbed once)
    agent = StubOnceAgent(stub_marker="3 by 7")

    runner = LockstepBatchRunner(agent, max_batches=2)
    for u in (a, b):
        runner.add(u)
    results = runner.run()

    # Both eventually solve; the stub was a recoverable parse failure for B only.
    assert results["A"]["success"] is True
    assert results["B"]["success"] is True

    def parse_failures(res):
        return [q for q in res["transcript"] if q.get("kind") == "query" and not q["parse_ok"]]

    a_fails = parse_failures(results["A"])
    b_fails = parse_failures(results["B"])
    assert a_fails == []  # sibling untouched
    assert len(b_fails) == 1
    assert b_fails[0]["stop_reason"] == "batch_expired"


# --------------------------------------------------------------------------- #
# 4. on_round heartbeat
# --------------------------------------------------------------------------- #
def test_on_round_called_with_active_units():
    a = _make_unit("A", _spec("a", [5, 3], [1, 1], [3, 1]))   # 2 queries
    b = _make_unit("B", _spec("b", [7, 3], [1, 1], [5, 1]))   # 4 queries
    scripts = {"3 by 5": ["MOVE_FORWARD"] * 2, "3 by 7": ["MOVE_FORWARD"] * 4}
    agent = ScriptedBatchAgent(scripts)

    calls = []

    def on_round(active):
        assert isinstance(active, list)
        assert all(isinstance(u, LockstepUnit) for u in active)
        calls.append([u.unit_id for u in active])

    runner = LockstepBatchRunner(agent, max_batches=2, on_round=on_round)
    for u in (a, b):
        runner.add(u)
    runner.run()

    # One callback per round that had active units at the top; membership
    # reflects who is still live after replies are applied (A drops out once it
    # has finished). The final all-finished round is a SILENT round (no batch
    # submitted) but still fires a heartbeat with the now-empty active set.
    assert calls[0] == ["A", "B"]
    assert ["B"] in calls               # B outlives A
    assert calls[-1] == []              # silent final round still heartbeats


def test_on_round_fires_on_silent_round():
    # A single unit that finishes: round 1 dispatches a batch, round 2 has the
    # unit active at the top but its next_query returns None (goal reached) so no
    # batch is submitted. on_round must still fire that round (empty active set).
    u = _make_unit("U", _spec("u", [4, 3], [1, 1], [2, 1]))   # 1 query then done
    agent = ScriptedBatchAgent({"3 by 4": ["MOVE_FORWARD"]})

    calls = []
    runner = LockstepBatchRunner(
        agent, max_batches=1,
        on_round=lambda active: calls.append([x.unit_id for x in active]),
    )
    runner.add(u)
    results = runner.run()

    assert results["U"]["success"] is True
    # Two rounds fired a heartbeat: the batch round, then the silent finish round.
    assert calls == [["U"], []]


# --------------------------------------------------------------------------- #
# Checkpoint-write failure is isolated to the failing unit
# --------------------------------------------------------------------------- #
def test_checkpoint_write_failure_isolated_per_unit(tmp_path, monkeypatch):
    import interface.batch_runner as br

    good_ckpt = tmp_path / "good.json"
    bad_ckpt = tmp_path / "bad.json"
    real_save = br.save_checkpoint

    def flaky_save(path, stepper):
        # Simulate a disk-full / OSError on one unit's checkpoint write only.
        if str(path) == str(bad_ckpt):
            raise OSError("disk full")
        return real_save(path, stepper)

    monkeypatch.setattr(br, "save_checkpoint", flaky_save)

    good = _make_unit("GOOD", _spec("good", [6, 3], [1, 1], [4, 1]),   # "3 by 6"
                      checkpoint_path=good_ckpt)
    bad = _make_unit("BAD", _spec("bad", [7, 3], [1, 1], [5, 1]),      # "3 by 7"
                     checkpoint_path=bad_ckpt)
    agent = ScriptedBatchAgent({
        "3 by 6": ["MOVE_FORWARD"] * 3,
        "3 by 7": ["MOVE_FORWARD"] * 4,
    })

    runner = LockstepBatchRunner(agent, max_batches=2)
    for u in (good, bad):
        runner.add(u)
    results = runner.run()

    # The unit whose checkpoint write failed is recorded as an error and dropped;
    # the batch is NOT aborted and the sibling completes normally.
    assert results["BAD"] == {"error": "disk full"}
    assert results["GOOD"]["success"] is True


# --------------------------------------------------------------------------- #
# 5. A raising stepper is isolated, not fatal
# --------------------------------------------------------------------------- #
class ExplodingStepper:
    """Duck-typed stepper that raises on next_query."""

    def start(self):
        self.transcript = []

    def next_query(self):
        raise RuntimeError("boom")


def test_unit_exception_is_captured_and_batch_continues():
    good = _make_unit("GOOD", _spec("good", [6, 3], [1, 1], [4, 1]))   # "3 by 6"
    bad = LockstepUnit("BAD", ExplodingStepper())
    agent = ScriptedBatchAgent({"3 by 6": ["MOVE_FORWARD"] * 3})

    runner = LockstepBatchRunner(agent, max_batches=2)
    for u in (good, bad):
        runner.add(u)
    results = runner.run()

    assert results["GOOD"]["success"] is True
    assert results["BAD"] == {"error": "boom"}


# --------------------------------------------------------------------------- #
# 6. Checkpoints: only at the query boundary, deleted on finish
# --------------------------------------------------------------------------- #
def test_checkpoint_only_at_query_boundary_and_deleted_on_finish(tmp_path, monkeypatch):
    import interface.batch_runner as br

    seen = {"saves": 0}
    real_save = br.save_checkpoint

    def guarded_save(path, stepper):
        # The invariant under test: at every save the stepper is at a query
        # boundary (both buffers empty). A cardinal MOVE_SOUTH expands to two
        # primitives, so a save mistakenly taken after apply_reply would find a
        # non-empty buffer here and fail this assertion.
        assert stepper.action_queue == [], stepper.action_queue
        assert stepper.primitive_buffer == [], stepper.primitive_buffer
        seen["saves"] += 1
        return real_save(path, stepper)

    monkeypatch.setattr(br, "save_checkpoint", guarded_save)

    ckpt = tmp_path / "unit.json"
    # Cardinal: MOVE_EAST, MOVE_EAST, MOVE_SOUTH, MOVE_SOUTH -> reaches (3,3).
    unit = _make_unit(
        "U", _spec("u", [5, 5], [1, 1], [3, 3]),
        checkpoint_path=ckpt, action_space="cardinal",
    )
    scripts = {"5 by 5": ["MOVE_EAST", "MOVE_EAST", "MOVE_SOUTH", "MOVE_SOUTH"]}
    agent = ScriptedBatchAgent(scripts)

    runner = LockstepBatchRunner(agent, max_batches=1)
    runner.add(unit)
    results = runner.run()

    assert results["U"]["success"] is True
    assert seen["saves"] >= 1          # the checkpoint path really was exercised
    assert not ckpt.exists()           # deleted when the unit finished


# --------------------------------------------------------------------------- #
# 7. Serial-equivalence drift guard (e2e, 3 real mini-mazes)
# --------------------------------------------------------------------------- #
def _serial_result(spec_dict, actions, **config_kwargs):
    spec = TaskSpecification.from_dict(spec_dict)
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    runner = build_runner(ExperimentConfig(**config_kwargs), backend, spec)
    return runner.run(ScriptedAgent(actions), verbose=False)


def test_serial_equivalence_e2e():
    # Same three cardinal episodes, driven serially through ExperimentRunner vs
    # in lockstep through the batch runner, must yield identical transcripts.
    m1 = _spec("cardinal_move", [5, 5], [1, 1], [1, 3])
    m2 = {  # reuse of the mechanics maze from tests/test_cardinal_runner.py
        "task_id": "cardinal_mechanics",
        "seed": 0,
        "difficulty_tier": 2,
        "maze": {"dimensions": [7, 3], "walls": [], "start": [1, 1], "goal": [5, 1]},
        "mechanisms": {
            "keys": [{"id": "k1", "position": [2, 1], "color": "red"}],
            "doors": [{"id": "d1", "position": [3, 1], "requires_key": "red",
                       "initial_state": "locked"}],
        },
        "goal": {"type": "reach_position", "target": [5, 1]},
        "max_steps": 30,
    }
    m3 = _spec("corridor", [6, 3], [1, 1], [4, 1])

    s1 = ["MOVE_SOUTH", "MOVE_SOUTH"]
    s2 = ["MOVE_EAST", "PICKUP", "INTERACT", "MOVE_EAST", "MOVE_EAST", "MOVE_EAST"]
    s3 = ["MOVE_EAST", "MOVE_EAST", "MOVE_EAST"]

    serial = {
        "M1": _serial_result(m1, s1, action_space="cardinal"),
        "M2": _serial_result(m2, s2, action_space="cardinal"),
        "M3": _serial_result(m3, s3, action_space="cardinal"),
    }
    assert all(r["success"] for r in serial.values())

    # markers: "5 by 5", "3 by 7", "3 by 6"
    scripts = {"5 by 5": s1, "3 by 7": s2, "3 by 6": s3}
    agent = ScriptedBatchAgent(scripts, usage=None)
    runner = LockstepBatchRunner(agent, max_batches=3)
    runner.add(_make_unit("M1", m1, action_space="cardinal"))
    runner.add(_make_unit("M2", m2, action_space="cardinal"))
    runner.add(_make_unit("M3", m3, action_space="cardinal"))
    batched = runner.run()

    for uid in ("M1", "M2", "M3"):
        assert batched[uid]["success"] is True
        assert _scrub(batched[uid]["transcript"]) == _scrub(serial[uid]["transcript"])


class RaiseOnRoundBatchAgent:
    """Wraps a batch agent and raises on the Nth ``generate_batch`` call, to
    model a batch client that throws (e.g. a cancel-on-ended 4xx that escaped
    its salvage, or a hard HTTP error)."""

    def __init__(self, inner, *, raise_on_call):
        self._inner = inner
        self._calls = 0
        self._raise_on_call = raise_on_call

    def generate_batch(self, batch):
        self._calls += 1
        if self._calls == self._raise_on_call:
            raise RuntimeError("batch tick blew up")
        return self._inner.generate_batch(batch)


def test_batch_exception_errors_round_but_keeps_prior_results():
    """A raised generate_batch marks that round's in-flight units as errors via
    the per-unit path; already-completed results survive and run() returns."""
    good = _make_unit("GOOD", _spec("good", [6, 3], [1, 1], [2, 1]))   # "3 by 6", 1 move
    bad = _make_unit("BAD", _spec("bad", [7, 3], [1, 1], [3, 1]))      # "3 by 7", 2 moves
    inner = ScriptedBatchAgent(
        {"3 by 6": ["MOVE_FORWARD"], "3 by 7": ["MOVE_FORWARD", "MOVE_FORWARD"]}
    )
    # Round 1: both submit (call #1, normal). Round 2: GOOD has finished and is
    # finalized before the batch; only BAD submits (call #2) -> raise.
    agent = RaiseOnRoundBatchAgent(inner, raise_on_call=2)

    runner = LockstepBatchRunner(agent, max_batches=2)
    for u in (good, bad):
        runner.add(u)
    results = runner.run()

    assert results["GOOD"]["success"] is True          # prior result survived
    assert "error" in results["BAD"]                    # round unit errored, not crashed
    assert "batch tick blew up" in results["BAD"]["error"]


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #
def test_rejects_agent_without_generate_batch():
    with pytest.raises(TypeError, match="generate_batch"):
        LockstepBatchRunner(object(), max_batches=1)


def test_deadline_kwarg_forwarded_only_when_accepted():
    from interface.batch_runner import _accepts_deadline_kwarg

    def with_deadline(batch, *, deadline_s=1.0):
        return []

    def without(batch):
        return []

    assert _accepts_deadline_kwarg(with_deadline) is True
    assert _accepts_deadline_kwarg(without) is False
