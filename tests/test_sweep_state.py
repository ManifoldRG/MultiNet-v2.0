import json
from pathlib import Path
from scripts import sweep_state as ss


def test_batches_cover_eleven_plus_smoke():
    ns = [b["n"] for b in ss.BATCHES]
    assert ns == list(range(0, 12))                      # 0 (smoke) .. 11
    assert ss.BATCHES[0]["name"] == "smoke"
    assert ss.BATCHES[11]["run_id"] == "cond_baseline_thinking"


def test_reshaped_batches_ablate_from_fair_default():
    names = [b["name"] for b in ss.BATCHES]
    for expected in ["obs_image_only", "obs_text_only", "ctx_current",
                     "act_egocentric", "icl_zero_shot", "hist_multiturn"]:
        assert expected in names
    # nothing tests a value that is now the fair default
    assert "obs_image_text" not in names
    assert "act_cardinal" not in names
    assert "ctx_last3" not in names
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
