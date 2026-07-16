import pandas as pd
import pytest
from analysis.notion_build import compute_stats, inject


def _ep():
    return pd.DataFrame([
        dict(model="claude", config="baseline_thinking", success=True, tokens=1000),
        dict(model="claude", config="baseline_thinking", success=False, tokens=3000),
        dict(model="kimi", config="baseline_thinking", success=True, tokens=2000),
    ])


def test_compute_stats_solve_and_tokens():
    s = compute_stats(_ep(), {"claude_api_usd": 100.0, "kimi_api_usd": None,
                              "qwen_a100_usd": 265.0})
    assert s["solve_claude_pct"] == "50.0%"
    assert s["total_episodes"] == "3"
    assert s["cost_claude_usd"] == "$100.00"
    assert s["cost_kimi_usd"] == "pending actual billing"
    assert "pending" in s["cost_total_usd"]


def test_inject_replaces_and_fails_loud():
    stats = {"total_episodes": "540"}
    assert inject("n = <!-- STAT: total_episodes -->.", stats) == "n = 540."
    with pytest.raises(KeyError):
        inject("<!-- STAT: nope -->", stats)


def test_build_tree_and_zip(tmp_path):
    from analysis.notion_build import build
    src = tmp_path / "notion"
    sub = src / "Experiment 3 — MultiNet Conditional Eval"
    (sub / "figures").mkdir(parents=True)
    (src / "Experiment 3 — MultiNet Conditional Eval.md").write_text("# Home\nn=<!-- STAT: total_episodes -->\n")
    (sub / "Bugs.md").write_text("![m](figures/x.png)\n")
    (sub / "figures" / "x.png").write_bytes(b"\x89PNG")
    z = build(src=src, out_zip=tmp_path / "out.zip", stats={"total_episodes": "540"})
    import zipfile
    names = zipfile.ZipFile(z).namelist()
    assert "Experiment 3 — MultiNet Conditional Eval.md" in names
    assert "Experiment 3 — MultiNet Conditional Eval/Bugs.md" in names


def test_build_fails_on_missing_figure(tmp_path):
    from analysis.notion_build import build
    src = tmp_path / "notion"
    sub = src / "Experiment 3 — MultiNet Conditional Eval"
    sub.mkdir(parents=True)
    (sub / "Bugs.md").write_text("![m](figures/missing.png)\n")
    with pytest.raises(FileNotFoundError):
        build(src=src, out_zip=tmp_path / "out.zip", stats={})
