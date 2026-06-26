from interface.agents import _normalized_usage
from interface.config import ExperimentConfig
from interface.loader import default_maze_path, load_task
from interface.runner import build_runner
from interface.smoke_tests.plans import v01_empty_room_trajectory
from interface.smoke_tests.smoke_llm import _AgentRecorder


class UsageReplayAgent:
    def __init__(self):
        self._actions = iter(v01_empty_room_trajectory())
        self.last_usage = None

    def __call__(self, messages):
        self.last_usage = {
            "input_tokens": 8,
            "output_tokens": 2,
            "total_tokens": 10,
        }
        return f"FINAL_OUTPUT: {next(self._actions)}"


def test_normalized_usage_accepts_provider_token_keys():
    assert _normalized_usage({"input_tokens": 8, "output_tokens": 2}) == {
        "input_tokens": 8,
        "output_tokens": 2,
        "total_tokens": 10,
    }


def test_agent_recorder_forwards_usage_metadata():
    records = []
    recorder = _AgentRecorder(UsageReplayAgent(), records)

    recorder([])

    assert recorder.last_usage == {
        "input_tokens": 8,
        "output_tokens": 2,
        "total_tokens": 10,
    }
    assert records[0]["usage"]["total_tokens"] == 10


def test_runner_persists_agent_usage_in_query_transcript():
    maze_path = default_maze_path("V01_empty_room.json")
    backend, spec = load_task(maze_path)
    runner = build_runner(
        ExperimentConfig(
            observation="text_only",
            context_window="current",
            querying="step_by_step",
            chat_history="stateless",
        ),
        backend,
        spec,
    )

    result = runner.run(UsageReplayAgent(), verbose=False, maze_path=maze_path)
    query_records = [item for item in result["transcript"] if item.get("kind") == "query"]

    assert result["success"] is True
    assert query_records
    assert query_records[0]["usage"]["total_tokens"] == 10
