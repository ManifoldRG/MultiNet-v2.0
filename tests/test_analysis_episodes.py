import json
from analysis.data import load_episodes


def _cell(tmp_path):
    cell = tmp_path / "cond-sweep-X" / "cond_prompt"
    ep_dir = cell / "runs" / "v01" / "minigrid" / "kimi-k2.6" / "seed_0" / "verbose"
    ep_dir.mkdir(parents=True)
    (ep_dir / "episode.json").write_text(json.dumps({"end_reason": "truncated"}))
    row = {"task_id": "v01", "agent_or_model": "kimi-k2.6", "prompt_variant": "verbose",
           "seed": 0, "success": False, "terminated": False, "truncated": True,
           "reward": 0.0, "steps": 33, "optimal_steps": 11, "optimality_ratio": 0.33,
           "tokens": 5000, "failure_point": None,
           "raw_output_ref": "runs/v01/minigrid/kimi-k2.6/seed_0/verbose/episode.json"}
    (cell / "episode_runs.jsonl").write_text(json.dumps(row) + "\n")
    return tmp_path


def test_load_episodes_shapes_row(tmp_path):
    df = load_episodes(_cell(tmp_path), cache=tmp_path / "c.parquet", force=True)
    assert len(df) == 1
    r = df.iloc[0]
    # The cond_prompt cell is split per prompt_variant — never aggregated to bare "cond_prompt".
    assert r.task_id == "v01" and r.model == "kimi-k2.6" and r.config == "cond_prompt·verbose"
    assert r.prompt_variant == "verbose"
    assert r.sweep == "cond-sweep-X" and r.end_reason == "truncated" and r.steps == 33
