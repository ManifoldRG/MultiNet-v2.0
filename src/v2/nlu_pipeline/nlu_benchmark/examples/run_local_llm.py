from nlu_benchmark.config import ExperimentConfig
from nlu_benchmark.runner import ExperimentRunner
from nlu_benchmark.agents import LocalTransformersAgent, LocalLLMConfig

runner = ExperimentRunner.from_json(
    "nlu_benchmark/sample mazes/V01_empty_room.json",
    config=ExperimentConfig(observation="text_only"),
)

# Small local model (no HF inference credits required).
agent = LocalTransformersAgent(
    config=LocalLLMConfig(
        model="HuggingFaceTB/SmolLM2-360M-Instruct",
        max_new_tokens=16,
    )
)

result = runner.run(agent)
print(result["success"])

