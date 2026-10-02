"""Golden: runs WITHOUT a start map keep byte-identical prompts.

The goldens were recorded from the code before the start-map feature existed:
every request the model was sent (system + user text, image positions, and
which frame each image is) and every per-query ``agent_messages`` as logged,
for a 3D R1-cell episode, a 3D rolling-chat episode, and two 2D episodes.

Regenerate (only when a prompt change is intended) with
``REGEN_START_MAP_GOLDEN=1 python -m pytest tests/test_start_map_golden.py``.
"""

from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("minigrid")

from gridworld.backends import get_backend  # noqa: E402
from start_map_test_utils import GOLDEN, R1_CELL, describe_episode, r1_spec, run_scripted  # noqa: E402

FRAME_PIXELS = 8 * 32  # the R1 render resolution "grid" for an 8x8 maze

EPISODES = {
    "3d_first_person_r1_cell_stateless": (
        lambda: get_backend("mujoco3d", camera="first_person", resolution=FRAME_PIXELS),
        R1_CELL,
    ),
    "3d_first_person_rolling": (
        lambda: get_backend("mujoco3d", camera="first_person", resolution=FRAME_PIXELS),
        {**R1_CELL, "chat_history": "rolling", "chat_turns_max": 2},
    ),
    "2d_verbose_image_text_rolling_one_shot": (
        lambda: get_backend("minigrid", render_mode="rgb_array"),
        dict(
            prompting="verbose",
            observation="image_text",
            context_window="text_summary_and_last_n",
            chat_history="rolling",
            chat_turns_max=2,
            in_context_learning="one_shot",
        ),
    ),
    "2d_standard_image_only_stateless_last_n": (
        lambda: get_backend("minigrid", render_mode="rgb_array"),
        dict(
            prompting="standard",
            observation="image_only",
            context_window="last_n",
            chat_history="stateless",
            in_context_learning="zero_shot",
        ),
    ),
}


def _describe(name: str) -> dict:
    make_backend, config = EPISODES[name]
    backend = make_backend()
    try:
        return describe_episode(*run_scripted(backend, r1_spec(), config))
    finally:
        backend.close()


def test_regenerate_golden_on_request():
    if not os.environ.get("REGEN_START_MAP_GOLDEN"):
        pytest.skip("set REGEN_START_MAP_GOLDEN=1 to rewrite the goldens")
    golden = {name: _describe(name) for name in EPISODES}
    GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.parametrize("name", list(EPISODES))
def test_prompts_match_golden(name):
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert _describe(name) == golden[name]
