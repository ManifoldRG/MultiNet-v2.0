from scripts.prepare_combined_job import merge_plans


def _plan(uid, tid, max_steps):
    return {"units": [{"unit_id": uid, "task_id": tid, "task_payload": {"max_steps": max_steps}}],
            "static_by_task": {tid: {}}, "scorer_config": {"x": 1},
            "difficulty_max_static_score": 1000, "pipeline_version": "v"}


def test_merge_combines_lpt_and_fresh_state():
    plan, state = merge_plans([_plan("a", "empty", 30), _plan("b", "s5", 267),
                               _plan("c", "mid", 120)], "job_massive_x")
    assert [u["task_id"] for u in plan["units"]] == ["s5", "mid", "empty"]  # LPT
    assert plan["job_id"] == "job_massive_x"
    assert set(state["units"]) == {"a", "b", "c"}
    assert all(v["status"] == "pending" for v in state["units"].values())
    assert plan["scorer_config"] == {"x": 1} and plan["difficulty_max_static_score"] == 1000
