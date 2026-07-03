from scripts.summarize_run import summarize, to_markdown

ROWS = [
    {"agent_or_model": "Qwen_Qwen3.6-27B", "task_id": "m1", "success": True,
     "truncated": False, "optimality_ratio": 0.9, "steps": 11, "tokens": 6000},
    {"agent_or_model": "Qwen_Qwen3.6-27B", "task_id": "m2", "success": False,
     "truncated": True, "optimality_ratio": 0.0, "steps": 200, "tokens": 90000},
    {"agent_or_model": "claude-opus-4-8", "task_id": "m1", "success": True,
     "truncated": False, "optimality_ratio": 1.0, "steps": 11, "tokens": 500},
]


def test_summarize_filters_by_model_and_aggregates():
    s = summarize(ROWS, "Qwen", "prompt")
    assert s["model"] == "Qwen" and s["batch"] == "prompt"
    assert s["n_episodes"] == 2 and s["n_success"] == 1
    assert s["success_rate"] == 0.5
    assert s["n_truncated"] == 1
    assert abs(s["mean_optimality_ratio"] - 0.45) < 1e-9
    assert s["total_tokens"] == 96000
    assert s["per_task"] == {"m1": True, "m2": False}


def test_to_markdown_mentions_key_numbers():
    md = to_markdown(summarize(ROWS, "Qwen", "prompt"))
    assert "prompt" in md and "50" in md and "truncated" in md.lower()
