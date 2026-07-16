import json

from analysis.data import load_queries, load_query_text
from analysis.queries_enrich import _parse_query


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


def test_query_enrichment_parses_final_output_history_items(tmp_path):
    query_path = tmp_path / "query.json"
    query_path.write_text(json.dumps({
        "agent_messages": [{
            "role": "user",
            "content": (
                "Recent history (last 3 steps, oldest first):\n"
                "Position after: (1, 2), facing EAST\n"
                "FINAL_OUTPUT: MOVE_FORWARD\n"
                "Feedback: MOVED — MOVE_FORWARD: Moved to (1, 2).\n\n"
                "You are at (1, 2) facing EAST."
            ),
        }],
        "parsed_actions": ["TURN_LEFT"],
        "assistant_reply": "FINAL_OUTPUT: TURN_LEFT",
    }))

    parsed = _parse_query(str(query_path))

    assert parsed["prev_action"] == "MOVE_FORWARD"
    assert parsed["prev_outcome"] == "MOVED"
    assert parsed["pos_x"] == 1
    assert parsed["pos_y"] == 2
    assert parsed["facing"] == "EAST"


def test_query_enrichment_falls_back_to_legacy_arrow_history(tmp_path):
    # Every corpus collected before the FINAL_OUTPUT-shaped history template
    # (the 2026-07 sweeps) embeds this arrow form; enrichment must not silently
    # return None for it.
    query_path = tmp_path / "query.json"
    query_path.write_text(json.dumps({
        "agent_messages": [{
            "role": "user",
            "content": (
                "Recent history (last 3 steps, oldest first):\n"
                "  (3, 14) facing EAST -> MOVE_FORWARD -> MOVED — "
                "MOVE_FORWARD: Moved to (3, 14).\n\n"
                "You are at (3, 14) facing EAST."
            ),
        }],
        "parsed_actions": ["TURN_LEFT"],
        "assistant_reply": "FINAL_OUTPUT: TURN_LEFT",
    }))

    parsed = _parse_query(str(query_path))

    assert parsed["prev_action"] == "MOVE_FORWARD"
    assert parsed["prev_outcome"] == "MOVED"


def test_hist_regex_matches_current_history_template():
    # _HIST hard-codes the RECENT_HISTORY_STEP wording; this round-trip catches
    # template/regex drift, which otherwise fails open (fields silently None).
    from analysis.queries_enrich import _HIST
    from prompting_experiments.prompt_templates.observation import RECENT_HISTORY_STEP

    rendered = RECENT_HISTORY_STEP.format(
        row=1, col=2, facing="EAST", action="MOVE_FORWARD",
        feedback="MOVED — MOVE_FORWARD: Moved to (1, 2).",
    )

    matches = _HIST.findall(rendered)

    assert matches, "queries_enrich._HIST no longer matches RECENT_HISTORY_STEP output"
    assert matches[-1][3] == "MOVE_FORWARD"
    assert matches[-1][4].upper() == "MOVED"
