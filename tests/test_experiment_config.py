from interface.config import ExperimentConfig


def test_from_dict_roundtrips_to_dict():
    cfg = ExperimentConfig(context_window="text_summary", observation="image_only")
    assert ExperimentConfig.from_dict(cfg.to_dict()) == cfg
