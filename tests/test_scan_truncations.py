"""Tests for the phase-1 truncation scanner (scripts/scan_truncations.py).

The two-tier Qwen rerun (docs/qwen-two-tier-rerun-design.md) runs all mazes at a
small token cap in phase 1 and re-runs only the mazes that hit the cap at 64k in
phase 2. This scanner flags the cap-hitters from phase-1 episode.json artifacts
and emits the phase-2 rerun manifest.

Load-bearing invariant (design §Detection): token usage lives on QUERY records
(``transcript[i where kind=="query"].usage.output_tokens``); a step record's
``truncated`` field is ENV truncation and must NOT flag.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_pipeline import load_manifest
from scripts.scan_truncations import (
    effective_cap,
    episode_is_truncated,
    main,
    scan_runs,
    write_rerun_manifest,
)


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #
def _query(output_tokens: int, *, parse_ok: bool = True, stop_reason=None,
           token_truncated=None, with_usage: bool = True) -> dict:
    rec: dict = {
        "kind": "query",
        "query_index": 0,
        "assistant_reply": "FINAL_OUTPUT: forward",
        "parse_ok": parse_ok,
    }
    if with_usage:
        rec["usage"] = {"input_tokens": 100, "output_tokens": output_tokens,
                        "total_tokens": 100 + output_tokens}
    if stop_reason is not None:
        rec["stop_reason"] = stop_reason
    if token_truncated is not None:
        rec["token_truncated"] = token_truncated
    return rec


def _step(truncated: bool = False) -> dict:
    return {"kind": "step", "step_index": 0, "truncated": truncated}


def _episode(queries: list[dict], steps: list[dict] | None = None) -> dict:
    transcript = [{"kind": "reset"}]
    for q in queries:
        transcript.append(q)
    for s in steps or []:
        transcript.append(s)
    return {"success": True, "transcript": transcript}


_OMIT = object()  # sentinel: write run_inputs.json but omit model_config.max_tokens


def _write_run(root: Path, task_id: str, model: str, *, queries: list[dict],
               steps: list[dict] | None = None, cap=8000,
               seed: int = 0, variant: str = "egocentric") -> Path:
    """Write one run dir.

    ``cap`` semantics:
    - int  -> run_inputs.json with model_config.max_tokens = cap
    - None -> NO run_inputs.json file at all (missing sidecar)
    - _OMIT -> run_inputs.json present but WITHOUT model_config.max_tokens
    """
    run_dir = root / "runs" / task_id / "minigrid" / model / f"seed_{seed}" / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "episode.json").write_text(
        json.dumps(_episode(queries, steps)), encoding="utf-8"
    )
    if cap is None:
        pass  # no sidecar
    elif cap is _OMIT:
        (run_dir / "run_inputs.json").write_text(
            json.dumps({"task_id": task_id, "model_id": model,
                        "model_config": {"temperature": 0.6}}),
            encoding="utf-8",
        )
    else:
        (run_dir / "run_inputs.json").write_text(
            json.dumps({"task_id": task_id, "model_id": model,
                        "model_config": {"max_tokens": cap}}),
            encoding="utf-8",
        )
    return run_dir


# --------------------------------------------------------------------------- #
# episode_is_truncated
# --------------------------------------------------------------------------- #
def test_cap_hit_at_first_query_flags():
    ep = _episode([_query(8000), _query(10), _query(10)])
    assert episode_is_truncated(ep, cap=8000) is True


def test_cap_hit_at_middle_query_flags():
    ep = _episode([_query(10), _query(8500), _query(10)])
    assert episode_is_truncated(ep, cap=8000) is True


def test_cap_hit_at_last_query_flags():
    ep = _episode([_query(10), _query(10), _query(9001)])
    assert episode_is_truncated(ep, cap=8000) is True


def test_all_under_cap_does_not_flag():
    ep = _episode([_query(10), _query(142), _query(2815)])
    assert episode_is_truncated(ep, cap=8000) is False


def test_parse_ok_cap_hit_still_flags():
    # The load-bearing case: Qwen's lenient parser salvages an action from the
    # cut-off reasoning, so a cap-hit query reports parse_ok=True yet is truncated.
    ep = _episode([_query(8000, parse_ok=True)])
    assert episode_is_truncated(ep, cap=8000) is True


def test_stop_reason_length_flags_even_under_cap():
    # Defensive: an explicit provider stop reason wins even when reported output
    # tokens are below the cap.
    ep = _episode([_query(50, stop_reason="length")])
    assert episode_is_truncated(ep, cap=8000) is True


def test_stop_reason_max_tokens_flags():
    ep = _episode([_query(50, stop_reason="max_tokens")])
    assert episode_is_truncated(ep, cap=8000) is True


def test_token_truncated_flag_flags():
    ep = _episode([_query(50, token_truncated=True)])
    assert episode_is_truncated(ep, cap=8000) is True


def test_natural_stop_reason_does_not_flag():
    ep = _episode([_query(50, stop_reason="stop")])
    assert episode_is_truncated(ep, cap=8000) is False


def test_step_record_truncated_does_not_flag():
    # ENV truncation on a STEP record must NOT flag by itself.
    ep = _episode([_query(10)], steps=[_step(truncated=True)])
    assert episode_is_truncated(ep, cap=8000) is False


def test_query_without_usage_does_not_crash():
    ep = _episode([_query(0, with_usage=False)])
    assert episode_is_truncated(ep, cap=8000) is False


# --------------------------------------------------------------------------- #
# scan_runs
# --------------------------------------------------------------------------- #
def test_scan_resolves_cap_from_run_inputs(tmp_path):
    _write_run(tmp_path, "task_hit", "qwen36_27b_vllm",
               queries=[_query(8000)], cap=8000)
    _write_run(tmp_path, "task_ok", "qwen36_27b_vllm",
               queries=[_query(142)], cap=8000)
    results = scan_runs(tmp_path)
    by_task = {r["task_id"]: r for r in results}
    assert by_task["task_hit"]["flagged"] is True
    assert by_task["task_hit"]["max_output_tokens"] == 8000
    assert by_task["task_ok"]["flagged"] is False
    assert by_task["task_ok"]["max_output_tokens"] == 142


def test_scan_explicit_cap_overrides_run_inputs(tmp_path):
    # run_inputs says 8000, but an explicit --cap of 100 flags the 142-token run.
    _write_run(tmp_path, "task_ok", "qwen36_27b_vllm",
               queries=[_query(142)], cap=8000)
    results = scan_runs(tmp_path, cap=100)
    assert results[0]["flagged"] is True


# The run-dir <model> path segment is the SANITIZED model id (from
# scripts.run_pipeline._sanitize(model_cfg["model"])), NOT the run-config key.
# For R1's qwen block ("model": "Qwen/Qwen3.6-27B") that segment is:
_QWEN_SEGMENT = "Qwen_Qwen3.6-27B"


def test_scan_model_filter_ignores_other_models(tmp_path):
    # Use the real production segment (not the run-config key 'qwen36_27b_vllm'),
    # so this test would catch a scanner that compares against the wrong name.
    _write_run(tmp_path, "task_a", _QWEN_SEGMENT, queries=[_query(8000)], cap=8000)
    _write_run(tmp_path, "task_b", "claude-opus-4-8", queries=[_query(8000)], cap=8000)
    results = scan_runs(tmp_path, model=_QWEN_SEGMENT)
    task_ids = {r["task_id"] for r in results}
    assert task_ids == {"task_a"}


def test_cli_fails_closed_when_model_filter_matches_zero_runs(tmp_path, capsys):
    """A --model that matches no run dir (e.g. the run-config key instead of the
    sanitized segment) must exit nonzero and NOT write a manifest."""
    _write_run(tmp_path, "task_a", _QWEN_SEGMENT, queries=[_query(8000)], cap=8000)
    out = tmp_path / "phase2.json"
    rc = main([
        "--artifacts-root", str(tmp_path),
        "--source-manifest", str(tmp_path / "src.json"),  # never read (guard fires first)
        "--out", str(out),
        "--model", "qwen36_27b_vllm",  # the wrong (run-config key) name -> 0 matches
    ])
    assert rc == 2
    assert not out.exists()
    assert "matched ZERO run dirs" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Unresolved-cap surface (fail-closed for a money-deciding tool)
# --------------------------------------------------------------------------- #
def test_missing_run_inputs_no_signal_is_cap_unresolved(tmp_path):
    _write_run(tmp_path, "task_x", "qwen36_27b_vllm",
               queries=[_query(142)], cap=None)  # no run_inputs.json
    results = scan_runs(tmp_path)
    assert results[0]["status"] == "cap_unresolved"
    assert results[0]["flagged"] is False


def test_run_inputs_without_max_tokens_is_cap_unresolved(tmp_path):
    _write_run(tmp_path, "task_x", "qwen36_27b_vllm",
               queries=[_query(142)], cap=_OMIT)  # run_inputs present, no max_tokens
    results = scan_runs(tmp_path)
    assert results[0]["status"] == "cap_unresolved"
    assert results[0]["flagged"] is False


def test_cap_unresolved_but_stop_reason_length_still_flags(tmp_path):
    # Cap-independent signals must survive an unreadable/absent sidecar.
    _write_run(tmp_path, "task_x", "qwen36_27b_vllm",
               queries=[_query(50, stop_reason="length")], cap=None)
    results = scan_runs(tmp_path)
    assert results[0]["flagged"] is True
    assert results[0]["status"] == "flagged"


def test_cap_unresolved_but_token_truncated_still_flags(tmp_path):
    _write_run(tmp_path, "task_x", "qwen36_27b_vllm",
               queries=[_query(50, token_truncated=True)], cap=_OMIT)
    results = scan_runs(tmp_path)
    assert results[0]["flagged"] is True
    assert results[0]["status"] == "flagged"


def test_explicit_cap_resolves_missing_sidecar(tmp_path):
    # An explicit --cap makes the cap resolvable even with no run_inputs.json.
    _write_run(tmp_path, "task_x", "qwen36_27b_vllm",
               queries=[_query(9000)], cap=None)
    results = scan_runs(tmp_path, cap=8000)
    assert results[0]["status"] == "flagged"
    assert results[0]["flagged"] is True


def test_queries_missing_usage_counted(tmp_path):
    _write_run(tmp_path, "task_x", "qwen36_27b_vllm",
               queries=[_query(0, with_usage=False), _query(142)], cap=8000)
    results = scan_runs(tmp_path)
    assert results[0]["queries_missing_usage"] == 1


def test_cli_fails_closed_on_cap_unresolved(tmp_path, capsys):
    _write_run(tmp_path, "task_x", "qwen36_27b_vllm",
               queries=[_query(142)], cap=None)
    code = main([
        "--artifacts-root", str(tmp_path),
        "--source-manifest", str(_SOURCE_MANIFEST),
    ])
    assert code != 0
    err = capsys.readouterr().err
    assert "cap_unresolved" in err or "WARNING" in err


def test_cli_allow_unresolved_exits_zero(tmp_path):
    _write_run(tmp_path, "task_x", "qwen36_27b_vllm",
               queries=[_query(142)], cap=None)
    code = main([
        "--artifacts-root", str(tmp_path),
        "--source-manifest", str(_SOURCE_MANIFEST),
        "--allow-unresolved",
    ])
    assert code == 0


def test_cli_exits_zero_when_all_caps_resolved(tmp_path):
    _write_run(tmp_path, "task_x", "qwen36_27b_vllm",
               queries=[_query(142)], cap=8000)
    code = main([
        "--artifacts-root", str(tmp_path),
        "--source-manifest", str(_SOURCE_MANIFEST),
    ])
    assert code == 0


def test_effective_cap_uniform_and_mixed(tmp_path):
    _write_run(tmp_path, "task_a", "qwen36_27b_vllm", queries=[_query(8000)], cap=8000)
    _write_run(tmp_path, "task_b", "qwen36_27b_vllm", queries=[_query(142)], cap=8000)
    assert effective_cap(scan_runs(tmp_path)) == 8000
    _write_run(tmp_path, "task_c", "qwen36_27b_vllm", queries=[_query(10)], cap=4096)
    assert effective_cap(scan_runs(tmp_path)) is None


def test_scan_run_dir_reported(tmp_path):
    run_dir = _write_run(tmp_path, "task_hit", "qwen36_27b_vllm",
                         queries=[_query(8000)], cap=8000)
    results = scan_runs(tmp_path)
    assert Path(results[0]["run_dir"]) == run_dir


# --------------------------------------------------------------------------- #
# write_rerun_manifest
# --------------------------------------------------------------------------- #
_SOURCE_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "gridworld" / "fixtures" / "manifest.r1_balanced_03.json"
)


def test_write_rerun_manifest_subset_verbatim_and_loads(tmp_path):
    source_rows = load_manifest(_SOURCE_MANIFEST)
    flagged_ids = [source_rows[0]["task_id"], source_rows[3]["task_id"]]
    flagged = [
        {"task_id": flagged_ids[0], "flagged": True, "max_output_tokens": 8000},
        {"task_id": flagged_ids[1], "flagged": True, "max_output_tokens": 8001},
        # An unflagged entry must be excluded from the emitted manifest.
        {"task_id": source_rows[5]["task_id"], "flagged": False,
         "max_output_tokens": 10},
    ]
    out = tmp_path / "manifest.r1_qwen_phase2.json"
    manifest = write_rerun_manifest(flagged, _SOURCE_MANIFEST, out, cap=8000)

    # Loads through the SAME loader the pipeline uses.
    rows = load_manifest(out)
    assert [r["task_id"] for r in rows] == flagged_ids

    # Rows preserved verbatim from source.
    by_id = {r["task_id"]: r for r in source_rows}
    for row in rows:
        assert row == by_id[row["task_id"]]

    # Provenance block.
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    sel = on_disk["selection"]
    assert sel["derived_from"].endswith("manifest.r1_balanced_03.json")
    assert sel["trigger"] == "output_tokens>=cap"
    assert sel["cap"] == 8000
    assert manifest["selection"]["cap"] == 8000


def test_write_rerun_manifest_empty_flagged(tmp_path):
    out = tmp_path / "empty.json"
    write_rerun_manifest([], _SOURCE_MANIFEST, out, cap=8000)
    assert load_manifest(out) == []
