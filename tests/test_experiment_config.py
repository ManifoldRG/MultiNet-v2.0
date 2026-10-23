import pytest
from interface.config import DEFAULT_MAX_HISTORY_TOKENS, ExperimentConfig


def test_from_dict_roundtrips_to_dict():
    cfg = ExperimentConfig(context_window="text_summary", observation="image_only")
    assert ExperimentConfig.from_dict(cfg.to_dict()) == cfg


def test_legacy_context_window_names_are_normalized():
    for legacy_name in ("last3", "text_summary_and_last3"):
        cfg = ExperimentConfig.from_dict({"context_window": legacy_name})
        assert cfg.context_window == legacy_name.replace("last3", "last_n")
        assert cfg.context_n == 3


def test_full_context_window_keeps_shared_history_budget_default():
    cfg = ExperimentConfig(context_window="full")

    assert cfg.max_history_tokens == DEFAULT_MAX_HISTORY_TOKENS


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


@pytest.mark.parametrize("bad", [True, False, 0, -1, 2.0])
def test_context_n_rejects_invalid(bad):
    with pytest.raises(ValueError):
        ExperimentConfig(context_n=bad)
