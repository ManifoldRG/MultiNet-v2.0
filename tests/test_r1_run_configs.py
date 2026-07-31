"""R1 two-tier Qwen run-config assets.

Phase 1 (`run_config.r1.json`) runs all three models with a wide-but-cheap Qwen
cap (8k) so cap-hitters truncate server-side; flagged mazes are re-run in phase
2 (`run_config.r1.qwen_phase2.json`), a qwen-only config with the full 64k cap.

See docs/qwen-two-tier-rerun-design.md for the authoritative values.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_pipeline import (
    _build_agent_from_spec,
    check_run_config_expectations,
    load_run_config,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _REPO_ROOT / "gridworld" / "fixtures"
_PHASE1 = _FIXTURES / "run_config.r1.json"
_PHASE2 = _FIXTURES / "run_config.r1.qwen_phase2.json"
_BALANCED_03 = _FIXTURES / "manifest.r1_balanced_03.json"
_SCAN_TOOL = _REPO_ROOT / "scripts" / "scan_truncations.py"

# The lockstep batch worker needs max_in_flight >= MAX_BATCHES (=50) for the
# claude/kimi API rows (see docs/batch-api-lockstep-runner-design.md).
_MAX_BATCHES = 50

# The path the phase-2 config declares == the scanner's default --out. The
# scanner does not expose a module-level constant, so we assert literal equality
# in both places (config + scan tool source) to keep them in sync.
_PHASE2_MANIFEST = "gridworld/fixtures/manifest.r1_qwen_phase2.json"

# Server keys that are INERT on the served (qwen_vllm_api) path — serve args come
# from env (QWEN_MAX_MODEL_LEN / QWEN_MAX_NUM_SEQS via lib/vllm_serve_args.sh).
# They must not reappear in the R1 qwen block (regression guard).
_INERT_SERVER_KEYS = (
    "max_model_len",
    "gpu_memory_utilization",
    "tensor_parallel_size",
    "dtype",
    "quantization",
    "enforce_eager",
    "enable_prefix_caching",
    "local_files_only",
)


# --------------------------------------------------------------------------- #
# Both configs load
# --------------------------------------------------------------------------- #
def test_both_configs_load():
    load_run_config(_PHASE1)
    load_run_config(_PHASE2)


# --------------------------------------------------------------------------- #
# Phase 1
# --------------------------------------------------------------------------- #
def test_phase1_passes_expectations_with_balanced_03():
    rc = load_run_config(_PHASE1)
    assert (_REPO_ROOT / rc["manifest"]).resolve() == _BALANCED_03.resolve()
    # Unequal caps (qwen 8k vs claude/kimi 64k) are allowed via the flag.
    check_run_config_expectations(rc, _BALANCED_03, None)  # no raise


def test_phase1_allows_unequal_caps():
    rc = load_run_config(_PHASE1)
    assert rc.get("allow_unequal_max_tokens") is True


def test_phase1_qwen_wide_cheap_cap():
    rc = load_run_config(_PHASE1)
    qwen = rc["models"]["qwen36_27b_vllm"]
    assert qwen["max_tokens"] == 8000
    assert qwen["timeout"] == 1800
    assert qwen["max_in_flight"] == 64


def test_kimi_deep_thinking_timeout_above_600s():
    # R1 shipped kimi timeout=600 and needed the KIMI_TIMEOUT_OVERRIDE=2400 env
    # stopgap mid-run; the proper value is folded into run_config here so the
    # env hack can stay removed. A 64k deep-thinking leg must not use <=600s.
    rc = load_run_config(_PHASE1)
    kimi = rc["models"]["kimi_k26"]
    assert kimi["max_tokens"] == 64000
    assert kimi["timeout"] > 600


def test_phase1_api_rows_meet_lockstep_min_in_flight():
    rc = load_run_config(_PHASE1)
    for name in ("kimi_k26", "claude_opus"):
        assert rc["models"][name]["max_in_flight"] >= _MAX_BATCHES


def test_phase1_phase_key():
    rc = load_run_config(_PHASE1)
    assert rc["phase"] == {"pass": 1, "label": "phase1"}


def test_phase1_inert_server_keys_removed_from_qwen():
    rc = load_run_config(_PHASE1)
    qwen = rc["models"]["qwen36_27b_vllm"]
    for key in _INERT_SERVER_KEYS:
        assert key not in qwen, f"inert server key {key!r} must not be in the qwen block"
    assert "_note_serve_args" in qwen


def test_phase1_keeps_kept_qwen_fields():
    rc = load_run_config(_PHASE1)
    qwen = rc["models"]["qwen36_27b_vllm"]
    for key in (
        "provider",
        "model",
        "base_url",
        "temperature",
        "enable_thinking",
        "group",
        "hardware_profile",
        "worker_count",
        "tasks",
    ):
        assert key in qwen


# --------------------------------------------------------------------------- #
# Phase 2
# --------------------------------------------------------------------------- #
def test_phase2_is_single_qwen_model():
    rc = load_run_config(_PHASE2)
    assert set(rc["models"]) == {"qwen36_27b_vllm"}
    assert rc["models"]["qwen36_27b_vllm"]["provider"] == "qwen_vllm_api"


def test_phase2_declares_scanner_out_path():
    rc = load_run_config(_PHASE2)
    assert rc["manifest"] == _PHASE2_MANIFEST
    # Same literal must appear in the scan tool (its default --out) so the two
    # stay in sync — the scanner writes the manifest to exactly this path.
    assert _PHASE2_MANIFEST in _SCAN_TOOL.read_text(encoding="utf-8")


def test_phase2_single_model_caps_guard_auto_passes():
    rc = load_run_config(_PHASE2)
    # The declared manifest does not exist yet (the scanner emits it live); the
    # guard only compares the declared path, it does not stat it. Single model =>
    # equal-caps guard auto-passes.
    check_run_config_expectations(rc, _REPO_ROOT / _PHASE2_MANIFEST, None)  # no raise


def test_phase2_qwen_full_cap_and_retry_budget():
    rc = load_run_config(_PHASE2)
    qwen = rc["models"]["qwen36_27b_vllm"]
    assert qwen["max_tokens"] == 64000
    assert qwen["timeout"] == 2400
    assert qwen["max_attempts"] == 2
    assert qwen["max_in_flight"] == 6


def test_phase2_phase_key():
    rc = load_run_config(_PHASE2)
    assert rc["phase"] == {"pass": 2, "label": "qwen_phase2"}


def test_phase2_experiment_config_verbatim():
    p1 = load_run_config(_PHASE1)
    p2 = load_run_config(_PHASE2)
    assert p2["experiment_config"] == p1["experiment_config"]


def test_phase2_inert_server_keys_absent():
    rc = load_run_config(_PHASE2)
    qwen = rc["models"]["qwen36_27b_vllm"]
    for key in _INERT_SERVER_KEYS:
        assert key not in qwen


# --------------------------------------------------------------------------- #
# max_attempts is real: the served-qwen builder plumbs it from the model config.
# --------------------------------------------------------------------------- #
def test_phase2_max_attempts_is_plumbed_into_agent():
    rc = load_run_config(_PHASE2)
    agent, _ = _build_agent_from_spec("qwen36_27b_vllm", rc["models"]["qwen36_27b_vllm"])
    assert agent.config.max_attempts == 2
    assert agent.config.timeout == 2400
    assert agent.config.max_tokens == 64000


# --------------------------------------------------------------------------- #
# Batch-deadline plumbing (I4) — optional run-config keys reach the agent config
# and NEVER enter the episode/unit hash.
# --------------------------------------------------------------------------- #
def test_batch_deadline_fields_plumbed_into_claude_and_kimi(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    claude_cfg = {
        "provider": "claude", "model": "claude-opus-4-8", "max_tokens": 64000,
        "batch_deadline_s": 5400, "batch_cancel_grace_s": 120,
    }
    agent, _ = _build_agent_from_spec("claude_opus", claude_cfg)
    assert agent.config.batch_deadline_s == 5400.0
    assert agent.config.batch_cancel_grace_s == 120.0

    kimi_cfg = {
        "provider": "kimi", "model": "kimi-k2.6", "max_tokens": 64000,
        "batch_deadline_s": 5400, "batch_cancel_grace_s": 120,
    }
    agent, _ = _build_agent_from_spec("kimi_k26", kimi_cfg)
    assert agent.config.batch_deadline_s == 5400.0
    assert agent.config.batch_cancel_grace_s == 120.0


def test_batch_deadline_fields_are_hash_invariant():
    """The batch scheduling knobs must be stripped by _runtime_model_config (the
    seam _expected_run_hash / unit-id derivation hash), or tuning a deadline
    would re-run every already-paid unit."""
    from scripts.run_pipeline import _runtime_model_config

    base = {
        "provider": "claude", "model": "claude-opus-4-8", "max_tokens": 64000,
        "enable_thinking": True, "effort": "xhigh",
    }
    with_batch = {
        **base, "batch_deadline_s": 5400, "batch_cancel_grace_s": 120,
        "batch_poll_interval_s": 15,
    }
    assert _runtime_model_config(base) == _runtime_model_config(with_batch)
