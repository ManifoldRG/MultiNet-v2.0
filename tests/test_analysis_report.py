import pandas as pd
from analysis.report import summarise


def test_summarise_pass_rates():
    df = pd.DataFrame([
        {"config": "c", "model": "kimi", "success": True, "end_reason": "success", "tokens": 10},
        {"config": "c", "model": "kimi", "success": False, "end_reason": "truncated", "tokens": 20},
    ])
    out = summarise(df)
    pr = out["pass_by_config_model"]
    assert float(pr.loc[("c", "kimi")]) == 0.5
    assert out["failure_taxonomy"].loc["kimi", "truncated"] == 1
