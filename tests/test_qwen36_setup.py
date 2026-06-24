from __future__ import annotations

from interface.agents.qwen35_vl import QWEN_MODEL_CLASS_NAMES
from scripts.run_pipeline import load_run_config
from deploy import REPO_ROOT


def test_qwen36_class_is_preferred_over_qwen35():
    names = QWEN_MODEL_CLASS_NAMES
    assert "Qwen3_6ForConditionalGeneration" in names
    assert "Qwen3_5ForConditionalGeneration" in names
    assert names.index("Qwen3_6ForConditionalGeneration") < names.index("Qwen3_5ForConditionalGeneration")
    # Auto* fallbacks remain, after the explicit classes.
    assert names.index("Qwen3_5ForConditionalGeneration") < names.index("AutoModelForCausalLM")


def test_smoke_qwen36_run_config_is_no_quant_and_correct_model():
    cfg = load_run_config(REPO_ROOT / "gridworld" / "fixtures" / "run_config.smoke_qwen36_kimi.json")
    qwen = cfg["models"]["qwen36_27b_hf"]
    assert qwen["model"] == "Qwen/Qwen3.6-27B"
    assert qwen["group"] == "qwen36-27b"
    assert qwen["load_in_4bit"] is False
    assert cfg["models"]["kimi_k26"]["group"] == "kimi-api"
