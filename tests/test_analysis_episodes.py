import json
from analysis.data import load_episodes


def _cell(tmp_path):
    cell = tmp_path / "cond-sweep-X" / "cond_prompt"
    ep_dir = cell / "runs" / "v01" / "minigrid" / "kimi-k2.6" / "seed_0" / "current"
    ep_dir.mkdir(parents=True)
    (ep_dir / "episode.json").write_text(json.dumps({"end_reason": "truncated"}))
    row = {"task_id": "v01", "agent_or_model": "kimi-k2.6", "prompt_variant": "current",
           "seed": 0, "success": False, "terminated": False, "truncated": True,
           "reward": 0.0, "steps": 33, "optimal_steps": 11, "optimality_ratio": 0.33,
           "tokens": 5000, "failure_point": None,
           "raw_output_ref": "runs/v01/minigrid/kimi-k2.6/seed_0/current/episode.json"}
    (cell / "episode_runs.jsonl").write_text(json.dumps(row) + "\n")
    return tmp_path


def test_load_episodes_shapes_row(tmp_path):
    df = load_episodes(_cell(tmp_path), cache=tmp_path / "c.parquet", force=True)
    assert len(df) == 1
    r = df.iloc[0]
    assert r.task_id == "v01" and r.model == "kimi-k2.6" and r.config == "cond_prompt"
    assert r.sweep == "cond-sweep-X" and r.end_reason == "truncated" and r.steps == 33
