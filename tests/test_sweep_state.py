import importlib
import json
from pathlib import Path
from scripts import sweep_state as ss


def test_kimictx_topo_selects_three_context_window_kimi_batches(monkeypatch):
    monkeypatch.setenv("SWEEP_TOPO", "kimictx")
    mod = importlib.reload(ss)
    try:
        ns = [b["n"] for b in mod.BATCHES]
        assert ns == [0, 1, 2, 3]                        # smoke + 3 variants only
        assert mod.BATCHES[0]["name"] == "smoke"
        assert "smoke_kimi3" in mod.BATCHES[0]["run_config"]
        variants = [b["prompt_variant"] for b in mod.BATCHES[1:]]
        assert variants == ["last3", "text_summary", "text_summary_and_last3"]
        for b in mod.BATCHES[1:]:
            assert b["conditions"] == "Context window"
            assert "conditional_context_window_kimi" in b["run_config"]
    finally:
        monkeypatch.delenv("SWEEP_TOPO", raising=False)
        importlib.reload(ss)                             # restore default 11-batch family


def test_batches_cover_ten_plus_smoke():
    ns = [b["n"] for b in ss.BATCHES]
    assert ns == list(range(0, 11))                      # 0 (smoke) .. 10
    assert ss.BATCHES[0]["name"] == "smoke"
    assert ss.BATCHES[10]["run_id"] == "cond_baseline_thinking"


def test_reshaped_batches_ablate_from_fair_default():
    names = [b["name"] for b in ss.BATCHES]
    for expected in ["obs_image_only", "ctx_current",
                     "act_cardinal", "icl_zero_shot", "hist_multiturn"]:
        assert expected in names
    # nothing tests a value that is now the fair default
    assert "obs_image_text" not in names
    assert "act_egocentric" not in names
    assert "ctx_last3" not in names
    # text_only is deferred to a future point (variant stays implemented, not run)
    assert "obs_text_only" not in names
    hm = next(b for b in ss.BATCHES if b["name"] == "hist_multiturn")
    assert hm["conditions"] == "History mechanism" and hm["prompt_variant"] == "multiturn"


def test_init_update_roundtrip(tmp_path: Path):
    st = ss.init_state("swp1", "2026-07-03T00:00:00Z")
    assert st["current_batch"] == 0 and st["phase"] == "provisioning"
    ss.update_batch(st, 1, status="complete", started_at="2026-07-03T01:00:00Z",
                    ended_at="2026-07-03T10:00:00Z")
    p = tmp_path / "sweep_state.json"
    ss.save_state(p, st)
    st2 = ss.load_state(p)
    assert ss._batch(st2, 1)["status"] == "complete"


def test_recompute_etas_calibrates_from_completed(tmp_path: Path):
    st = ss.init_state("swp1", "2026-07-03T00:00:00Z")
    # batch 1 (weight 3.0) took 9h -> anchor = 3.0 h per weight-unit
    ss.update_batch(st, 1, status="complete", started_at="2026-07-03T00:00:00Z",
                    ended_at="2026-07-03T09:00:00Z")
    ss.recompute_etas(st)
    b = next(x for x in st["batches"] if x["weight"] == 1.0 and x["status"] != "complete")
    assert 2.9 <= b["eta_hours"] <= 3.1


def test_render_table_has_a_row_per_batch():
    st = ss.init_state("swp1", "2026-07-03T00:00:00Z")
    table = ss.render_table(st)
    for b in ss.BATCHES:
        assert b["run_id"] in table


def test_sweep_topo_api_selects_api_only_configs(monkeypatch):
    import importlib
    monkeypatch.setenv("SWEEP_TOPO", "api")
    importlib.reload(ss)
    try:
        prompt = next(b for b in ss.BATCHES if b["name"] == "prompt")
        assert prompt["run_config"].endswith("_claude_kimi.json")
        assert "qwen" not in prompt["run_config"]
        assert ss.BATCHES[0]["run_config"].endswith("smoke_kimi_claude.json")
    finally:
        monkeypatch.delenv("SWEEP_TOPO", raising=False)
        importlib.reload(ss)  # restore the default (Qwen+Kimi+Claude) configs for other tests


def test_default_topo_uses_qwen_configs():
    # No SWEEP_TOPO -> full fleet configs (guards the reload-restore above too).
    assert ss._CFG.endswith("_claude_kimi_qwen.json")
    assert ss.BATCHES[0]["run_config"].endswith("smoke_qwen36_kimi_claude.json")
