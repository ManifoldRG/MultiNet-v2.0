from pathlib import Path
from analysis.data import discover_sweeps


def test_discover_skips_issues_and_finds_cells(tmp_path):
    (tmp_path / "sweepA" / "cond_prompt" / "runs").mkdir(parents=True)
    (tmp_path / "sweepA" / "issues_" / "cond_bad" / "runs").mkdir(parents=True)
    (tmp_path / "sweepB" / "cond_ctx_current" / "runs").mkdir(parents=True)
    # issues_ cells are included by default (relabeled downstream for mining)…
    found = {p.name for p in discover_sweeps(tmp_path)}
    assert found == {"cond_prompt", "cond_ctx_current", "cond_bad"}
    # …and dropped entirely with include_issues=False (legacy behavior).
    found = {p.name for p in discover_sweeps(tmp_path, include_issues=False)}
    assert found == {"cond_prompt", "cond_ctx_current"}
