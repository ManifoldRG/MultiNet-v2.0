import pytest
from interface.config import ExperimentConfig
from interface.r2_config import R2_CHAT_HISTORY, R2_CONFIG


def test_from_dict_roundtrips_to_dict():
    cfg = ExperimentConfig(context_window="text_summary", observation="image_only")
    assert ExperimentConfig.from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize("bad", [True, False, 0, -1, 2.0])
def test_progress_stall_k_rejects_invalid(bad):
    with pytest.raises(ValueError):
        ExperimentConfig(progress_stall_k=bad)


@pytest.mark.parametrize("ok", [None, 1, 20, 63])
def test_progress_stall_k_accepts_valid(ok):
    assert ExperimentConfig(progress_stall_k=ok).progress_stall_k == ok


@pytest.mark.parametrize("bad", [True, False, 0, -1, 2.0])
def test_max_history_tokens_rejects_invalid(bad):
    with pytest.raises(ValueError):
        ExperimentConfig(max_history_tokens=bad)


def test_r2_config_uses_full_history_profile():
    assert R2_CHAT_HISTORY == "rolling"
    assert R2_CONFIG.context_window == "full"
    assert R2_CONFIG.max_history_tokens == 100_000
    assert R2_CONFIG.chat_history == R2_CHAT_HISTORY
    assert R2_CONFIG.progress_stall_k is None
