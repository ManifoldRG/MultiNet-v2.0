"""Coordinator ``lockstep-worker`` role: drive the batch-API lockstep runner as a
distributed worker.

These tests pin the wiring contract (spec §Coordinator integration / task A9):

1. assign -> LockstepBatchRunner(ScriptedBatchAgent) -> upload happy path for 3
   units with ``max_batches=2`` (a single ``run()`` drains all three via
   continuous refill through ``client.assign``).
2. Per-round heartbeats fire for every active unit each round (recorded on the
   store; progress visible in ``job_state.json``).
3. The uploaded ``run_inputs.json`` carries an ``inputs_hash`` equal to the
   unit's ``episode_inputs_hash`` (the store's upload verify accepts it), plus
   the additive ``pricing_tier: "batch"`` provenance that must NOT change it.
4. A unit whose stepper raises is failed back via ``client.fail`` without killing
   the batch — sibling units still upload.

The store is the in-process ``CoordinatorStore`` (no HTTP), matching the fixture
pattern used by ``tests/test_distributed_progress.py`` /
``tests/test_run_pipeline.py``. The batch agent is a scripted double routed by a
unique per-maze marker (same idea as ``tests/test_batch_runner.py``).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from interface.agents.reply import Reply
from scripts.distributed_run_pipeline import (
    CoordinatorStore,
    prepare_job,
    run_lockstep_worker,
    state_path,
)

_STABLE_DIFFICULTY_MAX = 1000.0


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class ScriptedBatchAgent:
    """Batch agent routed by a unique marker substring in each unit's prompt.

    ``scripts`` maps a marker (the rendered "H by W grid" phrase, unique per
    maze) to that unit's ordered action tokens. Each round it emits one token per
    prompt from the matching queue (``DONE`` once exhausted). If a marker is in
    ``explode_markers`` a non-``Reply`` object is returned for it, so the
    stepper's ``apply_reply`` raises and that unit lands as an error result —
    without touching siblings.
    """

    _USAGE = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}

    def __init__(self, scripts, *, explode_markers=()):
        self._scripts = {k: list(v) for k, v in scripts.items()}
        self._cursor = {k: 0 for k in scripts}
        self._explode = set(explode_markers)
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
            if key in self._explode:
                replies.append(object())  # apply_reply(reply).text -> AttributeError
                continue
            i = self._cursor[key]
            script = self._scripts[key]
            token = script[i] if i < len(script) else "DONE"
            self._cursor[key] = i + 1
            replies.append(Reply(text=f"FINAL_OUTPUT: {token}", usage=self._USAGE))
        return replies


class RecordingStore(CoordinatorStore):
    """In-process store that records every heartbeat (worker, unit, progress)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heartbeats: list[tuple[str, str | None, int | None]] = []

    def heartbeat(self, worker_id, unit_id=None, progress=None):
        self.heartbeats.append((worker_id, unit_id, progress))
        return super().heartbeat(worker_id, unit_id, progress)


# --------------------------------------------------------------------------- #
# Fixtures: synthesize tiny corridors + a single-group (batch) job plan
# --------------------------------------------------------------------------- #
def _corridor_task(task_id: str, w: int, goal_x: int) -> dict:
    """A straight east corridor solvable by ``goal_x - 1`` MOVE_FORWARDs."""
    return {
        "task_id": task_id,
        "version": "2.0",
        "seed": 0,
        "difficulty_tier": 1,
        "description": "corridor",
        "maze": {"dimensions": [w, 3], "walls": [], "start": [1, 1], "goal": [goal_x, 1]},
        "mechanisms": {
            "keys": [], "doors": [], "switches": [], "gates": [],
            "blocks": [], "teleporters": [], "hazards": [],
        },
        "rules": {
            "key_consumption": True, "switch_type": "toggle", "hidden_mechanisms": [],
            "observability": "full", "view_size": 7,
        },
        "goal": {"type": "reach_position", "target": [goal_x, 1], "auxiliary_conditions": []},
        "metadata": {"chain_pattern": "none", "tiling": "square", "wall_topology": "open"},
        "max_steps": 100,
    }


# (task_id, width, goal_x, marker, forwards-to-solve)
_CORRIDORS = [
    ("corr_a", 4, 2, "3 by 4", 1),
    ("corr_b", 5, 3, "3 by 5", 2),
    ("corr_c", 6, 4, "3 by 6", 3),
]


def _write_job(tmp_path: Path, group: str = "claude-batch") -> tuple[Path, dict]:
    tasks_dir = tmp_path / "tasks_src"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    sources = []
    for task_id, w, goal_x, _marker, _n in _CORRIDORS:
        src = tasks_dir / f"{task_id}.json"
        src.write_text(json.dumps(_corridor_task(task_id, w, goal_x)), encoding="utf-8")
        sources.append(str(src))
        manifest_rows.append(
            {
                "task_id": task_id,
                "experiment": "test1",
                "condition": "default",
                "variant": task_id,
                "source": str(src),
                "expected_mechanisms": [],
            }
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"tasks": manifest_rows}), encoding="utf-8")

    run_config = {
        "models": {
            "claude_batch": {
                "provider": "claude",
                "model": "stub-batch-model",
                "group": group,
                "tasks": sources,
            }
        }
    }
    cfg_path = tmp_path / "run_config.json"
    cfg_path.write_text(json.dumps(run_config), encoding="utf-8")

    artifacts = tmp_path / "artifacts"
    plan = prepare_job(
        run_config_path=cfg_path,
        manifest_path=manifest_path,
        seeds=[0],
        conditions=None,
        artifacts_root=artifacts,
        run_set_id="lockstep",
        difficulty_max_static_score=_STABLE_DIFFICULTY_MAX,
    )
    return artifacts, plan


def _scripts_for(markers=None) -> dict:
    return {marker: ["MOVE_FORWARD"] * n for _t, _w, _g, marker, n in _CORRIDORS
            if markers is None or marker in markers}


def _agent_factory(agent):
    return lambda model_key, model_config: (agent, "stub-batch-model")


# --------------------------------------------------------------------------- #
# 1 + 2 + 3. Happy path: assign -> run -> upload, per-round heartbeats, hash match
# --------------------------------------------------------------------------- #
def test_lockstep_worker_assign_run_upload_happy_path(tmp_path):
    artifacts, plan = _write_job(tmp_path)
    assert {u["model_group"] for u in plan["units"]} == {"claude-batch"}
    assert len(plan["units"]) == 3

    store = RecordingStore(artifacts)
    agent = ScriptedBatchAgent(_scripts_for())
    caps = {"model_group": "claude-batch", "worker_concurrency": 2}

    result = run_lockstep_worker(
        artifacts_root=artifacts,
        capabilities=caps,
        max_batches=2,
        client=store,
        agent_factory=_agent_factory(agent),
    )

    # All three units completed and verified through the upload/verify path.
    assert result["completed"] == 3
    state = json.loads(state_path(artifacts).read_text(encoding="utf-8"))
    statuses = Counter(u["status"] for u in state["units"].values())
    assert statuses == {"verified": 3}

    # The batch was genuinely batched (a round drove >1 unit at least once).
    assert max(agent.batch_sizes) >= 2

    # Per-round heartbeats: every unit id was heartbeated, more than once each
    # (one per round it was active), and progress advanced in job_state.json.
    hb_by_unit = Counter(uid for _w, uid, _p in store.heartbeats if uid is not None)
    unit_ids = set(state["units"])
    assert unit_ids.issubset(set(hb_by_unit))
    assert all(hb_by_unit[uid] >= 1 for uid in unit_ids)
    assert sum(hb_by_unit.values()) > len(unit_ids)  # multiple rounds
    assert all(int(u.get("progress", 0)) >= 1 for u in state["units"].values())

    # Every uploaded run_inputs.json carries inputs_hash == episode_inputs_hash
    # (store verify already accepted it) plus the batch pricing tier, and the
    # provenance key did NOT perturb the hash.
    by_id = {u["unit_id"]: u for u in plan["units"]}
    for unit_id, unit in by_id.items():
        sidecar = json.loads(
            (artifacts / unit["run_dir"] / "run_inputs.json").read_text(encoding="utf-8")
        )
        assert sidecar["inputs_hash"] == unit["episode_inputs_hash"]
        assert sidecar["pricing_tier"] == "batch"
        episode = json.loads(
            (artifacts / unit["run_dir"] / "episode.json").read_text(encoding="utf-8")
        )
        assert episode["success"] is True


# --------------------------------------------------------------------------- #
# 4. A failing unit is failed back, the batch keeps going.
# --------------------------------------------------------------------------- #
def test_lockstep_worker_failing_unit_does_not_kill_loop(tmp_path):
    artifacts, plan = _write_job(tmp_path)
    # One attempt only, so the exploding unit fails once and is not retried.
    store = CoordinatorStore(artifacts, max_unit_attempts=1)
    agent = ScriptedBatchAgent(_scripts_for(), explode_markers={"3 by 5"})
    caps = {"model_group": "claude-batch", "worker_concurrency": 2}

    result = run_lockstep_worker(
        artifacts_root=artifacts,
        capabilities=caps,
        max_batches=2,
        client=store,
        agent_factory=_agent_factory(agent),
    )

    state = json.loads(state_path(artifacts).read_text(encoding="utf-8"))
    by_task = {u["unit_id"]: u["task_id"] for u in plan["units"]}
    statuses = {by_task[uid]: st["status"] for uid, st in state["units"].items()}

    # The exploding corridor (corr_b / "3 by 5") failed; the siblings verified.
    assert statuses["corr_b"] == "failed"
    assert statuses["corr_a"] == "verified"
    assert statuses["corr_c"] == "verified"
    assert result["completed"] == 2


# --------------------------------------------------------------------------- #
# 5. Crash resume: an assigned unit with a checkpoint resumes instead of restarting.
# --------------------------------------------------------------------------- #
def test_lockstep_worker_resumes_from_checkpoint(tmp_path, monkeypatch):
    import scripts.distributed_run_pipeline as drp
    from interface import episode_checkpoint as ckpt_mod
    from interface.config import ExperimentConfig
    from interface.episode_checkpoint import save_checkpoint
    from interface.episode_step import EpisodeStepper
    from scorer.config import ScorerConfig
    from pipeline.run_stage3 import build_episode_runner

    artifacts, plan = _write_job(tmp_path)
    store = CoordinatorStore(artifacts)
    agent = ScriptedBatchAgent(_scripts_for())
    caps = {"model_group": "claude-batch", "worker_concurrency": 2}

    # Spy on resume_stepper (lazily imported by the worker from this module).
    resumed: list[str] = []
    real_resume = ckpt_mod.resume_stepper

    def spy_resume(path, *, runner):
        resumed.append(str(path))
        return real_resume(path, runner=runner)

    monkeypatch.setattr(ckpt_mod, "resume_stepper", spy_resume)

    # Pre-seed a query-boundary checkpoint at one unit's run dir, built through the
    # SAME prep path the worker uses (identical maze/seed/max_steps) so replay
    # validates.
    unit_raw = next(u for u in plan["units"] if u["task_id"] == "corr_c")
    unit = store._unit_payload(plan, unit_raw)  # full assign payload (task_artifacts, scorer_config)
    row = drp.materialize_worker_inputs(unit, artifacts)
    prep = drp.pipeline._prepare_unit_run(
        row,
        unit["model_id"],
        model_config=unit["model_config"],
        manifest_path=artifacts / "distributed" / "worker_manifest.json",
        artifacts_root=artifacts,
        scored_static=unit["task_artifacts"]["scored_static"],
        difficulty_max=float(unit["difficulty_max_static_score"]),
        config=ScorerConfig.from_dict(unit["scorer_config"]),
        seed=int(unit["seed"]),
        prompt_variant=str(unit["prompt_variant"]),
        experiment_config=ExperimentConfig.from_dict(unit["experiment_config"]),
        conditions=unit.get("condition_set"),
    )
    run_dir = prep.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    runner = build_episode_runner(
        prep.source, prep.experiment_config, int(unit["seed"]),
        max_steps=prep.runtime_spec.max_steps,
    )
    seed_stepper = EpisodeStepper(runner, maze_path=str(prep.source))
    seed_stepper.start()
    seed_stepper.next_query()  # advance to the first query boundary (empty buffers)
    save_checkpoint(run_dir / "checkpoint.json", seed_stepper)

    run_lockstep_worker(
        artifacts_root=artifacts,
        capabilities=caps,
        max_batches=2,
        client=store,
        agent_factory=_agent_factory(agent),
    )

    # The seeded checkpoint was resumed (not started fresh) for that unit, the
    # unit still verified, and the checkpoint was cleaned up on completion.
    assert str(run_dir / "checkpoint.json") in resumed
    assert not (run_dir / "checkpoint.json").exists()
    state = json.loads(state_path(artifacts).read_text(encoding="utf-8"))
    assert Counter(u["status"] for u in state["units"].values())["verified"] == 3
