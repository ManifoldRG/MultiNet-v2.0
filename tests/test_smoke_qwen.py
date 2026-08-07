from __future__ import annotations

from deploy.smoke_qwen import TARGET_TOK_S, format_verdict, kernels_active, main


def test_format_verdict_below_target():
    msg = format_verdict(42.0)
    assert "BELOW" in msg
    assert "42.0" in msg
    assert "single-stream HF generate" in msg


def test_format_verdict_meets_target():
    msg = format_verdict(TARGET_TOK_S + 1.0)
    assert "MEETS" in msg
    assert "101.0" in msg


def test_kernels_active_reports_bools_for_expected_keys():
    status = kernels_active()
    assert set(status) == {"flash_attn", "fla", "causal_conv1d"}
    assert all(isinstance(v, bool) for v in status.values())


def test_self_test_mode_runs_without_gpu(capsys):
    code = main(["--self-test"])
    assert code == 0
    out = capsys.readouterr().out
    assert "kernels" in out.lower()
