"""M11 and its M12-M14 variants load and validate."""
from pathlib import Path
import pytest
from gridworld.task_spec import TaskSpecification
MAZES = sorted((Path(__file__).resolve().parents[1] / "ogbench/ogbench/procgen/maze_jsons").glob("M1[1-4]/*.json"))
@pytest.mark.parametrize("path", MAZES, ids=lambda p: p.stem)
def test_maze_is_valid(path):
    ok, errors = TaskSpecification.from_json(str(path)).validate()
    assert ok, errors
