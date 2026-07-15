import json
from analysis.data import load_episodes


def _cell(tmp_path, row_end_reason="_unset", episode_json_end_reason="truncated",
          write_episode_json=True):
    """Build a minimal cell dir with one episode_runs.jsonl row (and optionally an
    episode.json). ``row_end_reason`` controls the jsonl row's ``end_reason`` field:
    the sentinel ``"_unset"`` omits the key entirely (older aggregate files lack it).
    """
    cell = tmp_path / "cond-sweep-X" / "cond_prompt"
    ep_dir = cell / "runs" / "v01" / "minigrid" / "kimi-k2.6" / "seed_0" / "verbose"
    ep_dir.mkdir(parents=True)
    if write_episode_json:
        (ep_dir / "episode.json").write_text(
            json.dumps({"end_reason": episode_json_end_reason}))
    row = {"task_id": "v01", "agent_or_model": "kimi-k2.6", "prompt_variant": "verbose",
           "seed": 0, "success": False, "terminated": False, "truncated": True,
           "reward": 0.0, "steps": 33, "optimal_steps": 11, "optimality_ratio": 0.33,
           "tokens": 5000, "failure_point": None,
           "raw_output_ref": "runs/v01/minigrid/kimi-k2.6/seed_0/verbose/episode.json"}
    if row_end_reason != "_unset":
        row["end_reason"] = row_end_reason
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


def test_load_episodes_prefers_row_end_reason(tmp_path):
    # The aggregate jsonl row carries end_reason="stalled" directly; no episode.json
    # exists at all, so the loaded value can only have come from the row.
    root = _cell(tmp_path, row_end_reason="stalled", write_episode_json=False)
    df = load_episodes(root, cache=tmp_path / "c.parquet", force=True)
    assert len(df) == 1
    assert df.iloc[0].end_reason == "stalled"


def test_load_episodes_falls_back_to_run_dir_when_row_lacks_end_reason(tmp_path):
    # Older aggregate rows lack end_reason entirely; the loader must fall back to
    # reading it from the per-run episode.json file.
    root = _cell(tmp_path, row_end_reason="_unset",
                 episode_json_end_reason="terminated_failure")
    df = load_episodes(root, cache=tmp_path / "c.parquet", force=True)
    assert len(df) == 1
    assert df.iloc[0].end_reason == "terminated_failure"
