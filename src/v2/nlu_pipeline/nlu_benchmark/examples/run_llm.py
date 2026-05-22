import os
from pathlib import Path

# Load Anthropic API key from repo-root api_key.txt if ANTHROPIC_API_KEY is unset.
if not os.environ.get("ANTHROPIC_API_KEY"):
    for directory in Path(__file__).resolve().parents:
        key_file = directory / "api_key.txt"
        if key_file.is_file():
            os.environ["ANTHROPIC_API_KEY"] = key_file.read_text().strip()
            break

from nlu_benchmark.runner import ExperimentRunner
from nlu_benchmark.agents import ClaudeAnthropicAgent, ClaudeAnthropicConfig

runner = ExperimentRunner.from_json("nlu_benchmark/sample mazes/V01_empty_room.json")

# Override model=... on ClaudeAnthropicConfig if needed (see Anthropic model IDs).
agent = ClaudeAnthropicAgent(config=ClaudeAnthropicConfig())

result = runner.run(agent)
print(result["success"])
