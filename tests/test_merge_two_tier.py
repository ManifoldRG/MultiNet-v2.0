"""Tests for the later-pass-wins merge (scripts/merge_two_tier.py).

The two-tier Qwen rerun (docs/qwen-two-tier-rerun-design.md §Result reconciliation
& provenance) writes phase 1 (all mazes @8k) and phase 2 (flagged mazes @64k) to
SEPARATE artifacts roots. Final results = phase-1 episodes for unflagged units +
phase-2 overwrites for flagged units, merged EXPLICITLY (never rely on path
clobber), keyed by (task_id, model, seed, condition, prompt_variant).

Binding provenance invariant (B3 review): distributed workers do NOT self-stamp
``pass`` onto episode.json — the merge stamps provenance onto the episode.json
COPIES it writes into the merged root (phase-2 winners: pass=2 + max_tokens /
max_model_len + truncated_at_ceiling; phase-1 copies: pass=1 when absent), and
NEVER mutates the immutable source phase roots.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.merge_two_tier import MergeError, main, merge_phase_roots

_REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Fixture builders: synthetic phase artifacts roots
# --------------------------------------------------------------------------- #
def _query(output_tokens: int, *, stop_reason=None) -> dict:
    rec: dict = {
        "kind": "query",
        "query_index": 0,
        "assistant_reply": "FINAL_OUTPUT: forward",
        "parse_ok": True,
        "usage": {
            "input_tokens": 100,
            "output_tokens": output_tokens,
            "total_tokens": 100 + output_tokens,
        },
    }
    if stop_reason is not None:
        rec["stop_reason"] = stop_reason
    return rec


def _episode(output_tokens: int, *, stop_reason=None) -> dict:
    return {
        "success": True,
        "end_reason": "success",
        "steps_used": 3,
        "final_state": {"reward": 1.0},
        "transcript": [{"kind": "reset"}, _query(output_tokens, stop_reason=stop_reason)],
    }


def _row(task_id: str, model: str, seed: int, condition, variant: str) -> dict:
    """A build_run_row-shaped episode_runs.jsonl row (subset of real fields)."""
    ref = f"runs/{task_id}/minigrid/{model}/seed_{seed}/{variant}/episode.json"
    return {
        "task_id": task_id,
        "experiment": "r1",
        "condition": condition,
        "prompt_variant": variant,
        "backend": "minigrid",
        "agent_or_model": model,
        "seed": seed,
        "pass": 1,
        "max_tokens": None,
        "success": True,
        "end_reason": "success",
        "terminated": True,
        "truncated": False,
        "reward": 1.0,
        "steps": 3,
        "optimal_steps": 3,
        "optimality_ratio": 1.0,
        "path_choice": None,
        "mechanism_interaction_order": [],
        "failure_point": None,
        "tokens": 100 + 42,
        "raw_output_ref": ref,
    }


def _write_phase_root(root: Path, units: list[dict]) -> None:
    """Build a phase artifacts root from unit specs.

    Each unit dict: task_id, model, seed, condition, variant, output_tokens,
    max_tokens (cap in model_config), and optional max_model_len.
    Writes runs/**/{episode.json,run_inputs.json,run_score.json} plus the
    aggregate episode_runs.jsonl.
    """
    rows = []
    for u in units:
        variant = u["variant"]
        run_dir = (
            root / "runs" / u["task_id"] / "minigrid" / u["model"]
            / f"seed_{u['seed']}" / variant
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "episode.json").write_text(
            json.dumps(_episode(u["output_tokens"], stop_reason=u.get("stop_reason"))),
            encoding="utf-8",
        )
        model_config: dict = {}
        if u.get("max_tokens") is not None:
            model_config["max_tokens"] = u["max_tokens"]
        if u.get("max_model_len") is not None:
            model_config["max_model_len"] = u["max_model_len"]
        (run_dir / "run_inputs.json").write_text(
            json.dumps({
                "task_id": u["task_id"],
                "model_id": u["model"],
                "model_config": model_config,
            }),
            encoding="utf-8",
        )
        (run_dir / "run_score.json").write_text(
            json.dumps({"composite": 1.0, "task_id": u["task_id"]}),
            encoding="utf-8",
        )
        rows.append(_row(u["task_id"], u["model"], u["seed"], u["condition"], variant))
    (root / "episode_runs.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )


def _unit(task_id: str, *, model="qwen36_27b_vllm", seed=0, condition="c0",
          variant="egocentric", output_tokens=142, max_tokens=8000,
          max_model_len=None, stop_reason=None) -> dict:
    return {
        "task_id": task_id, "model": model, "seed": seed, "condition": condition,
        "variant": variant, "output_tokens": output_tokens,
        "max_tokens": max_tokens, "max_model_len": max_model_len,
        "stop_reason": stop_reason,
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _key(row: dict) -> tuple:
    return (row["task_id"], row["agent_or_model"], row["seed"],
            row.get("condition"), row["prompt_variant"])


# --------------------------------------------------------------------------- #
# Core merge behaviour
# --------------------------------------------------------------------------- #
def test_five_phase1_two_reruns_merges_to_five_rows_two_from_phase2(tmp_path):
    p1 = tmp_path / "phase1"
    p2 = tmp_path / "phase2"
    out = tmp_path / "merged"
    _write_phase_root(p1, [_unit(f"t{i}") for i in range(5)])
    # phase 2 reruns t1 and t3 at the 64k ceiling.
    _write_phase_root(p2, [
        _unit("t1", output_tokens=5000, max_tokens=64000, max_model_len=96000),
        _unit("t3", output_tokens=5000, max_tokens=64000, max_model_len=96000),
    ])

    summary = merge_phase_roots(p1, p2, out)

    rows = _read_jsonl(out / "episode_runs.jsonl")
    assert len(rows) == 5
    assert summary["total"] == 5
    assert summary["from_phase2"] == 2
    passes = {r["task_id"]: r["pass"] for r in rows}
    assert passes == {"t0": 1, "t1": 2, "t2": 1, "t3": 2, "t4": 1}
    # no duplicate keys
    keys = [_key(r) for r in rows]
    assert len(keys) == len(set(keys))


def test_phase2_winner_run_dir_is_copied_with_sidecars(tmp_path):
    p1 = tmp_path / "phase1"
    p2 = tmp_path / "phase2"
    out = tmp_path / "merged"
    _write_phase_root(p1, [_unit("t0"), _unit("t1")])
    _write_phase_root(p2, [_unit("t1", output_tokens=5000, max_tokens=64000,
                                 max_model_len=96000)])

    merge_phase_roots(p1, p2, out)

    for task in ("t0", "t1"):
        rd = out / "runs" / task / "minigrid" / "qwen36_27b_vllm" / "seed_0" / "egocentric"
        assert (rd / "episode.json").exists()
        assert (rd / "run_inputs.json").exists()
        assert (rd / "run_score.json").exists()
    # run_inputs / run_score copied verbatim from the winning phase.
    src = p2 / "runs/t1/minigrid/qwen36_27b_vllm/seed_0/egocentric/run_inputs.json"
    dst = out / "runs/t1/minigrid/qwen36_27b_vllm/seed_0/egocentric/run_inputs.json"
    assert json.loads(dst.read_text()) == json.loads(src.read_text())


# --------------------------------------------------------------------------- #
# Ceiling truncation (terminal case)
# --------------------------------------------------------------------------- #
def test_ceiling_truncation_stamped_in_row_and_episode_copy(tmp_path):
    p1 = tmp_path / "phase1"
    p2 = tmp_path / "phase2"
    out = tmp_path / "merged"
    _write_phase_root(p1, [_unit("t0"), _unit("hot")])
    # phase-2 rerun STILL hits the 64k ceiling.
    _write_phase_root(p2, [_unit("hot", output_tokens=64000, max_tokens=64000,
                                 max_model_len=96000)])

    summary = merge_phase_roots(p1, p2, out)

    rows = {r["task_id"]: r for r in _read_jsonl(out / "episode_runs.jsonl")}
    assert rows["hot"]["truncated_at_ceiling"] is True
    assert rows["t0"].get("truncated_at_ceiling") in (None, False)

    ep_copy = json.loads(
        (out / "runs/hot/minigrid/qwen36_27b_vllm/seed_0/egocentric/episode.json").read_text()
    )
    assert ep_copy["truncated_at_ceiling"] is True
    assert ep_copy["pass"] == 2
    assert ep_copy["max_tokens"] == 64000
    assert ep_copy["max_model_len"] == 96000
    assert summary["truncated_at_ceiling"] == ["hot"]


def test_phase2_under_ceiling_not_flagged(tmp_path):
    p1 = tmp_path / "phase1"
    p2 = tmp_path / "phase2"
    out = tmp_path / "merged"
    _write_phase_root(p1, [_unit("t0")])
    _write_phase_root(p2, [_unit("t0", output_tokens=5000, max_tokens=64000,
                                 max_model_len=96000)])

    summary = merge_phase_roots(p1, p2, out)

    rows = {r["task_id"]: r for r in _read_jsonl(out / "episode_runs.jsonl")}
    assert rows["t0"]["truncated_at_ceiling"] is False
    assert summary["truncated_at_ceiling"] == []


# --------------------------------------------------------------------------- #
# Provenance: source roots immutable, merged copies stamped
# --------------------------------------------------------------------------- #
def test_source_roots_unmodified_merged_copies_stamped(tmp_path):
    p1 = tmp_path / "phase1"
    p2 = tmp_path / "phase2"
    out = tmp_path / "merged"
    _write_phase_root(p1, [_unit("t0"), _unit("t1")])
    _write_phase_root(p2, [_unit("t1", output_tokens=5000, max_tokens=64000,
                                 max_model_len=96000)])

    src_p1_ep = p1 / "runs/t1/minigrid/qwen36_27b_vllm/seed_0/egocentric/episode.json"
    src_p2_ep = p2 / "runs/t1/minigrid/qwen36_27b_vllm/seed_0/egocentric/episode.json"
    before_p1 = src_p1_ep.read_text()
    before_p2 = src_p2_ep.read_text()

    merge_phase_roots(p1, p2, out)

    # SOURCE roots are the immutable ground truth: never mutated.
    assert src_p1_ep.read_text() == before_p1
    assert src_p2_ep.read_text() == before_p2
    assert "pass" not in json.loads(before_p1)
    assert "pass" not in json.loads(before_p2)

    # Merged COPIES carry provenance.
    p1_copy = json.loads(
        (out / "runs/t0/minigrid/qwen36_27b_vllm/seed_0/egocentric/episode.json").read_text()
    )
    p2_copy = json.loads(
        (out / "runs/t1/minigrid/qwen36_27b_vllm/seed_0/egocentric/episode.json").read_text()
    )
    assert p1_copy["pass"] == 1
    assert p2_copy["pass"] == 2


# --------------------------------------------------------------------------- #
# Fail-closed guards
# --------------------------------------------------------------------------- #
def test_expected_rerun_task_missing_from_phase2_is_fatal(tmp_path):
    p1 = tmp_path / "phase1"
    p2 = tmp_path / "phase2"
    out = tmp_path / "merged"
    _write_phase_root(p1, [_unit("t0"), _unit("t1"), _unit("t2")])
    # phase 2 ran only t1; t2 was flagged but is MISSING → must fail closed.
    _write_phase_root(p2, [_unit("t1", output_tokens=5000, max_tokens=64000)])
    manifest = tmp_path / "rerun_manifest.json"
    manifest.write_text(json.dumps({
        "tasks": [{"task_id": "t1"}, {"task_id": "t2"}],
    }), encoding="utf-8")

    with pytest.raises(Exception) as exc:
        merge_phase_roots(p1, p2, out, expected_rerun_manifest=manifest)
    assert "t2" in str(exc.value)
    assert "t1" not in str(exc.value)


def test_phase2_task_not_in_phase1_is_fatal(tmp_path):
    p1 = tmp_path / "phase1"
    p2 = tmp_path / "phase2"
    out = tmp_path / "merged"
    _write_phase_root(p1, [_unit("t0")])
    # "ghost" was never in phase 1 → phase 2 must be a subset.
    _write_phase_root(p2, [_unit("ghost", output_tokens=5000, max_tokens=64000)])

    with pytest.raises(Exception) as exc:
        merge_phase_roots(p1, p2, out)
    assert "ghost" in str(exc.value)


# --------------------------------------------------------------------------- #
# Fail-closed: phase-2 winner with an unresolvable ceiling cap
# --------------------------------------------------------------------------- #
def test_phase2_unresolvable_cap_no_stamp_is_fatal(tmp_path):
    # A phase-2 winner that genuinely hit 64k output tokens but whose
    # run_inputs.json has no model_config.max_tokens AND no explicit provider
    # stamp cannot be judged. Fail closed (matches the scanner's policy) rather
    # than silently recording truncated_at_ceiling=False.
    p1 = tmp_path / "phase1"
    p2 = tmp_path / "phase2"
    out = tmp_path / "merged"
    _write_phase_root(p1, [_unit("t0"), _unit("t1")])
    _write_phase_root(p2, [_unit("t1", output_tokens=64000, max_tokens=None,
                                 max_model_len=96000)])

    with pytest.raises(MergeError) as exc:
        merge_phase_roots(p1, p2, out)
    assert "t1" in str(exc.value)


def test_phase2_unresolvable_cap_allowed_records_in_summary(tmp_path):
    p1 = tmp_path / "phase1"
    p2 = tmp_path / "phase2"
    out = tmp_path / "merged"
    _write_phase_root(p1, [_unit("t0"), _unit("t1")])
    _write_phase_root(p2, [_unit("t1", output_tokens=64000, max_tokens=None,
                                 max_model_len=96000)])

    summary = merge_phase_roots(p1, p2, out, allow_unresolved_cap=True)

    assert summary["cap_unresolved"] == ["t1"]
    # Undecidable -> NOT claimed as a ceiling truncation.
    assert summary["truncated_at_ceiling"] == []
    rows = {r["task_id"]: r for r in _read_jsonl(out / "episode_runs.jsonl")}
    assert "truncated_at_ceiling" not in rows["t1"]  # unknown, not False
    assert rows["t1"]["pass"] == 2  # provenance still stamped
    ep = json.loads(
        (out / "runs/t1/minigrid/qwen36_27b_vllm/seed_0/egocentric/episode.json").read_text()
    )
    assert ep["pass"] == 2
    assert "truncated_at_ceiling" not in ep
    # Written summary matches the returned one.
    assert json.loads((out / "two_tier_merge.json").read_text()) == summary


def test_phase2_unresolvable_cap_but_stamped_still_flags(tmp_path):
    # Cap unresolvable, but an explicit stop_reason="length" decides it: this is
    # a genuine ceiling truncation, NOT a cap_unresolved case.
    p1 = tmp_path / "phase1"
    p2 = tmp_path / "phase2"
    out = tmp_path / "merged"
    _write_phase_root(p1, [_unit("t0"), _unit("t1")])
    _write_phase_root(p2, [_unit("t1", output_tokens=50, max_tokens=None,
                                 stop_reason="length", max_model_len=96000)])

    summary = merge_phase_roots(p1, p2, out)

    assert summary["truncated_at_ceiling"] == ["t1"]
    assert summary["cap_unresolved"] == []
    rows = {r["task_id"]: r for r in _read_jsonl(out / "episode_runs.jsonl")}
    assert rows["t1"]["truncated_at_ceiling"] is True


# --------------------------------------------------------------------------- #
# Multi-model production shape: qwen-only phase-2 among Claude/Kimi/Qwen
# --------------------------------------------------------------------------- #
def test_multi_model_qwen_only_rerun_passes_others_through(tmp_path):
    p1 = tmp_path / "phase1"
    p2 = tmp_path / "phase2"
    out = tmp_path / "merged"
    # phase 1: two tasks x three models (Claude/Kimi @64k, Qwen @8k).
    p1_units = []
    for task in ("t0", "t1"):
        p1_units.append(_unit(task, model="claude-opus-4-8", max_tokens=64000))
        p1_units.append(_unit(task, model="kimi-k2-6", max_tokens=64000))
        p1_units.append(_unit(task, model="qwen36_27b_vllm", max_tokens=8000))
    _write_phase_root(p1, p1_units)
    # phase 2: qwen-only rerun of a single task.
    _write_phase_root(p2, [_unit("t0", model="qwen36_27b_vllm",
                                 output_tokens=5000, max_tokens=64000,
                                 max_model_len=96000)])

    summary = merge_phase_roots(p1, p2, out)

    rows = _read_jsonl(out / "episode_runs.jsonl")
    assert len(rows) == 6  # subset guard does not fire on the qwen-only rerun
    assert summary["from_phase2"] == 1
    by_key = {(r["task_id"], r["agent_or_model"]): r["pass"] for r in rows}
    assert by_key[("t0", "qwen36_27b_vllm")] == 2  # qwen winner
    assert by_key[("t0", "claude-opus-4-8")] == 1  # passed through
    assert by_key[("t0", "kimi-k2-6")] == 1
    assert by_key[("t1", "qwen36_27b_vllm")] == 1
    # Claude/Kimi run dirs copied through.
    for model in ("claude-opus-4-8", "kimi-k2-6"):
        rd = out / "runs/t0/minigrid" / model / "seed_0/egocentric"
        assert (rd / "episode.json").exists()
        assert json.loads((rd / "episode.json").read_text())["pass"] == 1


# --------------------------------------------------------------------------- #
# Summary file + CLI
# --------------------------------------------------------------------------- #
def test_summary_written_to_out_root(tmp_path):
    p1 = tmp_path / "phase1"
    p2 = tmp_path / "phase2"
    out = tmp_path / "merged"
    _write_phase_root(p1, [_unit("t0"), _unit("t1")])
    _write_phase_root(p2, [_unit("t1", output_tokens=64000, max_tokens=64000,
                                 max_model_len=96000)])

    summary = merge_phase_roots(p1, p2, out)

    written = json.loads((out / "two_tier_merge.json").read_text())
    assert written == summary
    assert written["total"] == 2
    assert written["from_phase2"] == 1
    assert written["truncated_at_ceiling"] == ["t1"]


def test_cli_merges_and_writes_summary(tmp_path):
    p1 = tmp_path / "phase1"
    p2 = tmp_path / "phase2"
    out = tmp_path / "merged"
    _write_phase_root(p1, [_unit("t0"), _unit("t1")])
    _write_phase_root(p2, [_unit("t1", output_tokens=5000, max_tokens=64000,
                                 max_model_len=96000)])

    rc = main([
        "--phase1", str(p1), "--phase2", str(p2), "--out", str(out),
    ])
    assert rc == 0
    rows = _read_jsonl(out / "episode_runs.jsonl")
    assert len(rows) == 2
    assert (out / "two_tier_merge.json").exists()


def test_cli_subprocess_smoke(tmp_path):
    p1 = tmp_path / "phase1"
    p2 = tmp_path / "phase2"
    out = tmp_path / "merged"
    _write_phase_root(p1, [_unit("t0")])
    _write_phase_root(p2, [_unit("t0", output_tokens=5000, max_tokens=64000)])
    result = subprocess.run(
        [sys.executable, "-m", "scripts.merge_two_tier",
         "--phase1", str(p1), "--phase2", str(p2), "--out", str(out)],
        cwd=str(_REPO_ROOT), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "episode_runs.jsonl").exists()
