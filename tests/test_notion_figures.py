import json

import pandas as pd

from analysis.notion_figures import maze_matrix, load_kimictx_episodes
from analysis.data import CONDITIONAL_CONFIGS


def _fake_ep():
    rows = []
    for maze, opt in [("m_easy", 10), ("m_hard", 60)]:
        for model in ["claude", "kimi"]:
            for cfg in CONDITIONAL_CONFIGS[:3]:
                rows.append(dict(task_id=maze, model=model, config=cfg,
                                 success=(opt == 10), steps=opt, optimal_steps=opt,
                                 tokens=1000, seed=0))
    return pd.DataFrame(rows)


def test_maze_matrix_shape_and_order():
    m = maze_matrix(_fake_ep())
    assert list(m.index) == ["m_easy", "m_hard"]          # sorted by optimal_steps
    assert list(m.columns) == CONDITIONAL_CONFIGS          # full config axis, NaN where absent
    assert m.loc["m_easy", CONDITIONAL_CONFIGS[0]] == 1.0
    assert m.loc["m_hard", CONDITIONAL_CONFIGS[0]] == 0.0
    assert m[CONDITIONAL_CONFIGS[5]].isna().all()          # config with no episodes stays NaN


def test_maze_matrix_model_filter():
    m = maze_matrix(_fake_ep(), model="claude")
    assert m.shape[0] == 2


def test_load_kimictx_episodes(tmp_path):
    p = tmp_path / "episode_runs.jsonl"
    # Verified kimictx schema: `condition` is the maze family (S/D/M/B/default),
    # `prompt_variant` is the context variant, `agent_or_model` is the model.
    rec = {"task_id": "conditional_s_s5_14x14_corridor_1", "agent_or_model": "kimi-k2.6",
           "condition": "S", "prompt_variant": "last3", "success": True, "steps": 12,
           "optimal_steps": 11, "tokens": 5000, "seed": 0}
    p.write_text(json.dumps(rec) + "\n")
    df = load_kimictx_episodes(p)
    assert df.loc[0, "config"] == "ctx_last3"    # derived from prompt_variant
    assert df.loc[0, "model"] == "kimi-k2.6"     # derived from agent_or_model
    assert df.loc[0, "success"] == True  # noqa: E712
