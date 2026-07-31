import pytest
from interface.config import ExperimentConfig


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
