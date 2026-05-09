"""Tests for model interface, evaluation harness, and NL domain."""

import pytest
import sys
import os
import json
import tempfile
import urllib.error
import io
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
from model_interface import ModelInterface, ModelInput, ModelOutput, RandomModelInterface
from evaluation_harness import (
    BenchmarkEvaluationResult,
    EvaluationHarness,
    EvaluationResult,
    TierMetrics,
)
from gridworld.task_spec import TaskSpecification
from gridworld.actions import ACTION_NAMES
from adapters.lmstudio_vlm_adapter import LMStudioVLMAdapter
from adapters.ollama_vlm_adapter import OllamaVLMAdapter


class TestModelInput:
    def test_create_model_input(self):
        inp = ModelInput(
            image=np.zeros((64, 64, 3), dtype=np.uint8),
            text_prompt="Navigate to the goal",
            action_space=ACTION_NAMES,
            step_number=1,
            max_steps=100,
        )
        assert inp.image.shape == (64, 64, 3)
        assert inp.step_number == 1

    def test_optional_context(self):
        inp = ModelInput(
            image=np.zeros((64, 64, 3), dtype=np.uint8),
            text_prompt="test",
            action_space={0: "left"},
            step_number=0,
            max_steps=10,
            additional_context="Extra info",
        )
        assert inp.additional_context == "Extra info"


class TestRandomModel:
    def test_random_model_name(self):
        model = RandomModelInterface(seed=42)
        assert model.model_name == "random"

    def test_random_model_predict(self):
        model = RandomModelInterface(seed=42)
        inp = ModelInput(
            image=np.zeros((64, 64, 3), dtype=np.uint8),
            text_prompt="test",
            action_space=ACTION_NAMES,
            step_number=1,
            max_steps=100,
        )
        output = model.predict(inp)
        assert isinstance(output, ModelOutput)
        assert output.action in ACTION_NAMES

    def test_random_model_deterministic(self):
        """Same seed should produce same sequence."""
        model1 = RandomModelInterface(seed=123)
        model2 = RandomModelInterface(seed=123)
        inp = ModelInput(
            image=np.zeros((64, 64, 3), dtype=np.uint8),
            text_prompt="test",
            action_space=ACTION_NAMES,
            step_number=1,
            max_steps=100,
        )
        actions1 = [model1.predict(inp).action for _ in range(10)]
        actions2 = [model2.predict(inp).action for _ in range(10)]
        assert actions1 == actions2

    def test_random_model_batch(self):
        model = RandomModelInterface(seed=42)
        inp = ModelInput(
            image=np.zeros((64, 64, 3), dtype=np.uint8),
            text_prompt="test",
            action_space=ACTION_NAMES,
            step_number=1,
            max_steps=100,
        )
        outputs = model.predict_batch([inp, inp, inp])
        assert len(outputs) == 3
        assert all(isinstance(o, ModelOutput) for o in outputs)


class TestLMStudioAdapter:
    def test_setup_fails_clearly_when_server_unavailable(self):
        adapter = LMStudioVLMAdapter(model="google/gemma-3-4b-it", base_url="http://localhost:9")
        with pytest.raises(RuntimeError, match="Could not reach LM Studio"):
            adapter.setup()

    def test_http_error_includes_response_body(self):
        adapter = LMStudioVLMAdapter(model="test-model")
        error = urllib.error.HTTPError(
            url="http://localhost:1234/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"too many images"}'),
        )

        detail = adapter._format_request_error(error, prior_count=2)

        assert "400" in detail
        assert "too many images" in detail
        assert "2 prior image" in detail

    def test_predict_retries_with_fewer_prior_images(self):
        class RetryAdapter(LMStudioVLMAdapter):
            def __init__(self):
                super().__init__(model="test-model", max_prior_images=2)
                self.attempts = []

            def _predict_once(self, input: ModelInput, text_prompt: str, prior_images: list[np.ndarray]) -> str:
                self.attempts.append(len(prior_images))
                if len(prior_images) > 1:
                    raise urllib.error.HTTPError(
                        url="http://localhost:1234/v1/chat/completions",
                        code=400,
                        msg="Bad Request",
                        hdrs=None,
                        fp=io.BytesIO(b'{"error":"payload too large"}'),
                    )
                return "2\nmove forward"

        adapter = RetryAdapter()
        output = adapter.predict(
            ModelInput(
                image=np.zeros((64, 64, 3), dtype=np.uint8),
                text_prompt="Reach the goal",
                action_space=ACTION_NAMES,
                step_number=1,
                max_steps=20,
                prior_images=[
                    np.zeros((64, 64, 3), dtype=np.uint8),
                    np.ones((64, 64, 3), dtype=np.uint8),
                ],
            )
        )

        assert output.action == 2
        assert adapter.attempts == [2, 1]
        assert "reduced prior images from 2 to 1" in output.reasoning


class TestOllamaAdapter:
    def test_parse_response_prefers_explicit_action_line(self):
        adapter = OllamaVLMAdapter(model="test-model")
        action, confidence, reasoning = adapter._parse_response(
            "I see the agent facing right.\nAction: 1",
            ACTION_NAMES,
        )
        assert action == 1
        assert confidence is None
        assert "Action: 1" in reasoning

    def test_build_prompt_matches_image_only_policy(self):
        adapter = OllamaVLMAdapter(model="test-model")
        prompt = adapter._build_prompt(
            ModelInput(
                image=np.zeros((64, 64, 3), dtype=np.uint8),
                text_prompt="unused mission",
                action_space=ACTION_NAMES,
                step_number=3,
                max_steps=20,
            )
        )
        assert "blue agent from images only" in prompt
        assert "green square goal" in prompt
        assert "complete route to the goal" in prompt
        assert "Action: <action_id>" in prompt
        assert "Mission:" not in prompt

    def test_extract_message_text_ignores_thinking_field(self):
        adapter = OllamaVLMAdapter(model="test-model")
        text = adapter._extract_message_text({
            "message": {"content": "", "thinking": "Action: 2"}
        })
        assert text == ""

    def test_extract_message_text_strips_inline_think_tags(self):
        adapter = OllamaVLMAdapter(model="test-model")
        text = adapter._extract_message_text({
            "message": {"content": "<think>hidden</think>\nAction: 1", "thinking": "Action: 2"}
        })
        assert text == "Action: 1"

    def test_build_messages_uses_previous_and_current_images(self):
        adapter = OllamaVLMAdapter(model="test-model")
        messages = adapter._build_messages(
            ModelInput(
                image=np.zeros((8, 8, 3), dtype=np.uint8),
                text_prompt="unused mission",
                action_space=ACTION_NAMES,
                step_number=3,
                max_steps=20,
                additional_context="Recent steps:\nstep 2: action=turn_right, agent_direction=0, agent_position=(1, 1)",
                prior_images=[np.ones((8, 8, 3), dtype=np.uint8)],
            )
        )
        assert len(messages) == 3
        assert messages[1]["content"] == "This is the previous image after the action turn_right was taken."
        assert len(messages[1]["images"]) == 1
        assert messages[2]["content"].startswith("This is the current image.")
        assert len(messages[2]["images"]) == 1


class TestEvaluationHarness:
    @pytest.fixture
    def simple_spec(self):
        return TaskSpecification.from_dict({
            "task_id": "test_simple",
            "seed": 42,
            "difficulty_tier": 1,
            "maze": {
                "dimensions": [6, 6],
                "walls": [],
                "start": [1, 1],
                "goal": [4, 4],
            },
            "goal": {"type": "reach_position", "target": [4, 4]},
            "max_steps": 20,
        })

    def write_task(self, path: Path, spec: TaskSpecification) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        spec.to_json(str(path))

    @pytest.fixture
    def tier_task_dir(self, tmp_path, simple_spec):
        task_root = tmp_path / "tasks"
        for idx in range(3):
            data = simple_spec.to_dict()
            data["task_id"] = f"tier1_task_{idx}"
            self.write_task(
                task_root / "tier1" / f"task_{idx}.json",
                TaskSpecification.from_dict(data),
            )
        return task_root

    def test_evaluate_single_task(self, simple_spec):
        model = RandomModelInterface(seed=42)
        harness = EvaluationHarness(model)
        result = harness.evaluate_task(simple_spec)
        assert result.task_id == "test_simple"
        assert result.steps_taken > 0
        assert result.steps_taken <= 20
        harness.close()

    def test_evaluate_task_records_model_outputs_in_trajectory(self, simple_spec):
        class StubModel(ModelInterface):
            @property
            def model_name(self) -> str:
                return "stub"

            def predict(self, input: ModelInput) -> ModelOutput:
                return ModelOutput(
                    action=6,
                    confidence=0.25,
                    reasoning="API error: channel closed",
                    raw_output="channel closed",
                )

        harness = EvaluationHarness(StubModel())
        result = harness.evaluate_task(simple_spec)
        assert result.trajectory
        first_info = result.trajectory[0].info
        assert first_info["model_confidence"] == 0.25
        assert first_info["model_reasoning"] == "API error: channel closed"
        assert first_info["model_raw_output"] == "channel closed"
        assert first_info["model_error"] == "API error: channel closed"
        assert first_info["model_latency_s"] >= 0.0
        assert first_info["model_latency_ms"] >= 0.0
        harness.close()

    def test_history_configuration_controls_model_input(self, simple_spec):
        class RecordingModel(ModelInterface):
            def __init__(self):
                self.inputs = []

            @property
            def model_name(self) -> str:
                return "recorder"

            def predict(self, input: ModelInput) -> ModelOutput:
                self.inputs.append(input)
                return ModelOutput(action=2, confidence=1.0, reasoning="move", raw_output="2")

        model = RecordingModel()
        harness = EvaluationHarness(model, history_images=0, history_text=False)
        try:
            harness.evaluate_task(simple_spec)
        finally:
            harness.close()

        assert model.inputs
        assert all(not inp.prior_images for inp in model.inputs)
        assert all(inp.additional_context is None for inp in model.inputs)

    def test_evaluate_tier(self, tier_task_dir):
        model = RandomModelInterface(seed=42)
        harness = EvaluationHarness(model)
        metrics = harness.evaluate_tier(tier=1, task_dir=str(tier_task_dir))
        assert isinstance(metrics, TierMetrics)
        assert metrics.tier == 1
        assert metrics.num_tasks == 3  # 3 tier1 tasks
        assert 0.0 <= metrics.success_rate <= 1.0
        harness.close()

    def test_evaluate_all(self, tier_task_dir):
        model = RandomModelInterface(seed=42)
        harness = EvaluationHarness(model)
        result = harness.evaluate_all(task_dir=str(tier_task_dir), tiers=[1])
        assert isinstance(result, EvaluationResult)
        assert result.model_name == "random"
        assert 1 in result.tier_metrics
        harness.close()

    def test_result_serialization(self, tier_task_dir):
        model = RandomModelInterface(seed=42)
        harness = EvaluationHarness(model)
        result = harness.evaluate_all(task_dir=str(tier_task_dir), tiers=[1])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            result.save(f.name)
            with open(f.name) as fp:
                data = json.load(fp)
            assert "model_name" in data
            assert "tier_metrics" in data
            os.unlink(f.name)
        harness.close()

    def test_evaluate_task_set_uses_point_scoring(self, simple_spec):
        model = RandomModelInterface(seed=42)
        harness = EvaluationHarness(model)
        result = harness.evaluate_task_set([simple_spec], benchmark_name="unit")
        assert isinstance(result, BenchmarkEvaluationResult)
        assert result.benchmark_name == "unit"
        assert result.num_tasks == 1
        assert result.total_available_points > 0
        assert len(result.task_results) == 1
        task_result = result.task_results[0]
        assert task_result.task_id == simple_spec.task_id
        assert task_result.available_points >= task_result.points_earned
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            result.save(f.name)
            with open(f.name) as fp:
                data = json.load(fp)
            assert data["benchmark_name"] == "unit"
            os.unlink(f.name)
        harness.close()

    def test_evaluate_task_dir_loads_directory(self, tmp_path, simple_spec):
        model = RandomModelInterface(seed=42)
        harness = EvaluationHarness(model)
        task_dir = tmp_path / "task_dir"
        for idx in range(2):
            data = simple_spec.to_dict()
            data["task_id"] = f"directory_task_{idx}"
            self.write_task(
                task_dir / f"task_{idx}.json",
                TaskSpecification.from_dict(data),
            )
        result = harness.evaluate_task_dir(task_dir=str(task_dir), benchmark_name="unit_dir")
        assert result.benchmark_name == "unit_dir"
        assert result.num_tasks == 2
        assert len(result.task_results) == 2
        harness.close()


class TestCrossDomain:
    def test_canonical_roundtrip(self):
        from cross_domain.canonical_task_spec import CanonicalTaskSpec, CanonicalGoal, CanonicalObject
        from cross_domain.gridworld_adapter import GridWorldDomainAdapter

        spec = TaskSpecification.from_dict({
            "task_id": "test_roundtrip",
            "seed": 42,
            "difficulty_tier": 1,
            "maze": {
                "dimensions": [10, 10],
                "walls": [[3, 3], [3, 4]],
                "start": [1, 1],
                "goal": [8, 8],
            },
            "mechanisms": {
                "keys": [{"id": "k1", "position": [2, 2], "color": "yellow"}],
            },
            "goal": {"type": "reach_position", "target": [8, 8]},
            "max_steps": 100,
        })

        adapter = GridWorldDomainAdapter()
        canonical = adapter.to_canonical(spec)

        assert canonical.task_id == "test_roundtrip"
        assert canonical.difficulty == 1
        assert 0.0 <= canonical.agent_start[0] <= 1.0
        assert 0.0 <= canonical.agent_start[1] <= 1.0
        assert canonical.goal.goal_type == "reach"
        assert len(canonical.objects) > 0  # walls + key

        # Find the key in canonical objects
        key_objs = [o for o in canonical.objects if o.obj_type == "collectible"]
        assert len(key_objs) == 1
        assert key_objs[0].id == "k1"

    def test_gridworld_canonical_roundtrip_preserves_domain_features(self):
        from cross_domain.gridworld_adapter import GridWorldDomainAdapter

        spec = TaskSpecification.from_dict({
            "task_id": "feature_roundtrip",
            "seed": 7,
            "difficulty_tier": 5,
            "version": "2.0",
            "description": "Preserve gridworld-specific features.",
            "maze": {
                "dimensions": [10, 10],
                "walls": [[9, 9]],
                "start": [1, 1],
                "goal": [8, 8],
            },
            "mechanisms": {
                "keys": [{"id": "k1", "position": [2, 2], "color": "red"}],
                "doors": [{"id": "d1", "position": [3, 2], "requires_key": "red"}],
                "switches": [
                    {
                        "id": "s1",
                        "position": [8, 7],
                        "controls": ["g1"],
                        "color": "white",
                        "switch_type": "one_shot",
                        "initial_state": "off",
                    }
                ],
                "gates": [{"id": "g1", "position": [8, 8], "initial_state": "closed"}],
                "blocks": [{"id": "b1", "position": [4, 4], "pushable": False, "color": "grey"}],
            },
            "rules": {
                "key_consumption": False,
                "switch_type": "one_shot",
                "hidden_mechanisms": ["s1"],
                "observability": "view_cone",
                "view_size": 5,
            },
            "goal": {
                "type": "push_block_to",
                "target_ids": ["b1"],
                "target_positions": [[8, 8]],
            },
            "dependency_chain": {
                "depth": 2,
                "sequence": [
                    {"step": 1, "type": "key-door", "element": "k1", "unlocks": "d1"},
                    {"step": 2, "type": "switch-gate", "element": "s1", "unlocks": "g1"},
                ],
                "notation": "k1 -> d1 -> s1 -> g1",
            },
            "distractors": [
                {"type": "inactive_switch", "element_id": "s1", "description": "hidden switch"}
            ],
            "metadata": {"chain_pattern": "mixed"},
            "max_steps": 120,
        })

        adapter = GridWorldDomainAdapter()
        canonical = adapter.to_canonical(spec)
        restored = adapter.from_canonical(canonical)

        assert restored.rules.observability == "view_cone"
        assert restored.rules.view_size == 5
        assert restored.rules.key_consumption is False
        assert restored.rules.hidden_mechanisms == ["s1"]
        assert restored.dependency_chain is not None
        assert restored.dependency_chain.depth == 2
        assert restored.distractors is not None
        assert restored.distractors[0].element_id == "s1"
        assert restored.metadata == {"chain_pattern": "mixed"}
        assert restored.version == "2.0"
        assert restored.maze.walls[0].to_tuple() == (9, 9)
        assert restored.mechanisms.switches[0].position.to_tuple() == (8, 7)
        assert restored.mechanisms.switches[0].color == "white"
        assert restored.mechanisms.switches[0].switch_type == "one_shot"
        assert restored.mechanisms.blocks[0].pushable is False
        assert restored.goal.goal_type == "push_block_to"
        assert restored.goal.target_ids == ["b1"]
        assert restored.goal.target_positions[0].to_tuple() == (8, 8)
