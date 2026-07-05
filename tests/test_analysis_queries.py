import json
from analysis.data import load_queries, load_query_text


def test_load_queries_and_text(tmp_path):
    qd = (tmp_path / "swp" / "cond_prompt" / "runs" / "v01" / "minigrid"
          / "kimi-k2.6" / "seed_0" / "current" / "queries" / "query_000")
    qd.mkdir(parents=True)
    (qd / "query.json").write_text(json.dumps({
        "query_index": 0, "env_step_count": 0, "parse_ok": True,
        "parsed_actions": ["MOVE_FORWARD"], "has_image": True, "llm_latency_s": 1.2,
        "agent_messages": [{"role": "user", "content": "go north"}],
        "assistant_reply": "FINAL_OUTPUT: MOVE_FORWARD"}))
    df = load_queries(tmp_path, cache=tmp_path / "q.parquet", force=True)
    assert len(df) == 1
    r = df.iloc[0]
    assert r.query_index == 0 and r.parse_ok and r.n_actions == 1 and r.env_step == 0
    msgs, reply = load_query_text(r.query_path)
    assert reply.startswith("FINAL_OUTPUT") and msgs[0]["content"] == "go north"
