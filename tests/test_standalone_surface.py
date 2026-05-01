import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gridworld.backends.base import GridState
from gridworld.backends.multigrid_backend import MultiGridBackend


def test_import_multigrid_without_gridworld_side_effects():
    """Plain `import multigrid` should bootstrap Gymnasium plugin filtering itself."""
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(ROOT)!r}); "
        "import multigrid; "
        "print(multigrid.MultiGridEnv.__name__)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "MultiGridEnv"

def test_multigrid_invalid_actions_default_to_wait():
    backend = MultiGridBackend()
    backend._configured = True

    captured = {}

    class DummyEnv:
        def step(self, action):
            captured["action"] = action
            return np.zeros((4, 4, 3), dtype=np.uint8), 0.0, False, False, {}

    backend.env = DummyEnv()
    backend._build_grid_state = lambda: GridState(agent_position=(0, 0), agent_direction=0)

    _, _, _, _, state, _ = backend.step(999)

    assert captured["action"] == 8
    assert state.step_count == 1
