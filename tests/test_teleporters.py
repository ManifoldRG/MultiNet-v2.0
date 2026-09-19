"""Tests for teleporter functionality in MiniGrid backend."""

import pytest
import sys
import os
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from gridworld.task_spec import TaskSpecification
from gridworld.task_parser import TaskParser
from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.actions import MiniGridActions
from gridworld.custom_env import TeleporterObj


@pytest.fixture
def teleporter_spec():
    """Create a simple task with a teleporter."""
    return TaskSpecification.from_dict({
        "task_id": "test_teleporter",
        "seed": 42,
        "difficulty_tier": 5,
        "maze": {
            "dimensions": [8, 8],
            "walls": [],
            "start": [1, 1],
            "goal": [6, 6],
        },
        "mechanisms": {
            "teleporters": [
                {
                    "id": "tp1",
                    "position_a": [2, 1],
                    "position_b": [5, 5],
                    "bidirectional": True,
                }
            ]
        },
        "goal": {"type": "reach_position", "target": [6, 6]},
        "max_steps": 50,
    })


@pytest.fixture
def oneway_teleporter_spec():
    """Create a task with a one-way teleporter."""
    return TaskSpecification.from_dict({
        "task_id": "test_oneway_teleporter",
        "seed": 42,
        "difficulty_tier": 5,
        "maze": {
            "dimensions": [8, 8],
            "walls": [],
            "start": [1, 1],
            "goal": [6, 6],
        },
        "mechanisms": {
            "teleporters": [
                {
                    "id": "tp1",
                    "position_a": [2, 1],
                    "position_b": [5, 5],
                    "bidirectional": False,
                }
            ]
        },
        "goal": {"type": "reach_position", "target": [6, 6]},
        "max_steps": 50,
    })


class TestTeleporterValidation:
    """Test teleporter position validation in task_spec."""

    def test_valid_teleporter_passes_validation(self, teleporter_spec):
        is_valid, errors = teleporter_spec.validate()
        assert is_valid, f"Validation errors: {errors}"

    def test_oob_teleporter_a_fails(self):
        spec = TaskSpecification.from_dict({
            "task_id": "test",
            "seed": 42,
            "difficulty_tier": 5,
            "maze": {"dimensions": [8, 8], "walls": [], "start": [1, 1], "goal": [6, 6]},
            "mechanisms": {
                "teleporters": [{"id": "tp", "position_a": [10, 10], "position_b": [3, 3]}]
            },
            "goal": {"type": "reach_position", "target": [6, 6]},
            "max_steps": 50,
        })
        is_valid, errors = spec.validate()
        assert not is_valid
        assert any("Teleporter" in e and "endpoint A" in e for e in errors)

    def test_oob_teleporter_b_fails(self):
        spec = TaskSpecification.from_dict({
            "task_id": "test",
            "seed": 42,
            "difficulty_tier": 5,
            "maze": {"dimensions": [8, 8], "walls": [], "start": [1, 1], "goal": [6, 6]},
            "mechanisms": {
                "teleporters": [{"id": "tp", "position_a": [3, 3], "position_b": [10, 10]}]
            },
            "goal": {"type": "reach_position", "target": [6, 6]},
            "max_steps": 50,
        })
        is_valid, errors = spec.validate()
        assert not is_valid
        assert any("Teleporter" in e and "endpoint B" in e for e in errors)


class TestTeleporterPlacement:
    """Test that teleporters are placed in the environment."""

    def test_teleporter_objects_placed(self, teleporter_spec):
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(teleporter_spec)
        obs, state, info = backend.reset(seed=42)

        assert len(backend.env.teleporters) == 2  # Two endpoints
        assert "tp1_a" in backend.env.teleporters
        assert "tp1_b" in backend.env.teleporters

    def test_teleporter_objects_are_correct_type(self, teleporter_spec):
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(teleporter_spec)
        backend.reset(seed=42)

        for tp in backend.env.teleporters.values():
            assert isinstance(tp, TeleporterObj)

    def test_bidirectional_partners(self, teleporter_spec):
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(teleporter_spec)
        backend.reset(seed=42)

        tp_a = backend.env.teleporters["tp1_a"]
        tp_b = backend.env.teleporters["tp1_b"]
        assert tp_a.partner is tp_b
        assert tp_b.partner is tp_a

    def test_oneway_partner(self, oneway_teleporter_spec):
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(oneway_teleporter_spec)
        backend.reset(seed=42)

        tp_a = backend.env.teleporters["tp1_a"]
        tp_b = backend.env.teleporters["tp1_b"]
        assert tp_a.partner is tp_b
        assert tp_b.partner is None  # One-way: B doesn't teleport to A


class TestTeleporterMechanics:
    """Test teleporter step mechanics."""

    def test_agent_teleports_on_step(self, teleporter_spec):
        """Agent at (1,1) facing right, move forward to (2,1) which is teleporter A -> should teleport to (5,5)."""
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(teleporter_spec)
        obs, state, info = backend.reset(seed=42)

        # Agent starts at (1,1) facing right (dir=0)
        assert state.agent_position == (1, 1)

        # Move forward: agent goes to (2,1) where teleporter A is
        obs, reward, term, trunc, state, info = backend.step(MiniGridActions.MOVE_FORWARD)

        # Should have been teleported to (5,5)
        assert state.agent_position == (5, 5), f"Expected (5,5), got {state.agent_position}"

    def test_teleporter_cooldown_in_state(self, teleporter_spec):
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(teleporter_spec)
        obs, state, info = backend.reset(seed=42)

        # Check that teleporter cooldowns are tracked
        assert "tp1_a" in state.teleporter_cooldowns
        assert "tp1_b" in state.teleporter_cooldowns
        assert state.teleporter_cooldowns["tp1_a"] == 0
        assert state.teleporter_cooldowns["tp1_b"] == 0


class TestTeleporterTaskFile:
    """Test loading the tier5 teleporter task JSON."""

    def test_load_teleporter_task(self):
        task_path = Path(__file__).resolve().parent.parent / "gridworld" / "tasks" / "tier5" / "teleporter_004.json"
        spec = TaskSpecification.from_json(str(task_path))
        assert spec.task_id == "tier5_teleporter_004"
        assert len(spec.mechanisms.teleporters) == 2

    def test_teleporter_task_validates(self):
        task_path = Path(__file__).resolve().parent.parent / "gridworld" / "tasks" / "tier5" / "teleporter_004.json"
        spec = TaskSpecification.from_json(str(task_path))
        is_valid, errors = spec.validate()
        assert is_valid, f"Validation errors: {errors}"

    def test_teleporter_task_runs(self):
        task_path = Path(__file__).resolve().parent.parent / "gridworld" / "tasks" / "tier5" / "teleporter_004.json"
        spec = TaskSpecification.from_json(str(task_path))
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(spec)
        obs, state, info = backend.reset(seed=42)
        assert state.agent_position == (1, 1)
        assert len(backend.env.teleporters) == 4  # 2 teleporters * 2 endpoints


_M7_PORTALS = (
    Path(__file__).resolve().parent.parent
    / "M7-10 New Mechanisms"
    / "M7-10 New Mechanisms"
    / "M7"
    / "18x18_dense_kr_sg_kb_portals_0.json"
)


class TestPortalRules:
    """Matching-color pads warp on entry; arrival does not immediately bounce back."""

    def test_portals_json_key_is_accepted(self):
        spec = TaskSpecification.from_dict({
            "task_id": "portals_alias",
            "seed": 1,
            "difficulty_tier": 1,
            "maze": {"dimensions": [8, 8], "walls": [], "start": [1, 1], "goal": [6, 6]},
            "mechanisms": {
                "portals": [{
                    "id": "portal_purple",
                    "position_a": [2, 1],
                    "position_b": [5, 5],
                    "color": "purple",
                    "bidirectional": True,
                }]
            },
            "goal": {"type": "reach_position", "target": [6, 6]},
            "max_steps": 50,
        })
        assert len(spec.mechanisms.portals) == 1
        assert spec.mechanisms.portals[0].color == "purple"

    def test_color_is_placed_on_both_pads(self, teleporter_spec):
        spec = TaskSpecification.from_dict({
            "task_id": "colored_portal",
            "seed": 1,
            "difficulty_tier": 1,
            "maze": {"dimensions": [8, 8], "walls": [], "start": [1, 1], "goal": [6, 6]},
            "mechanisms": {
                "teleporters": [{
                    "id": "portal_cyan",
                    "position_a": [2, 1],
                    "position_b": [5, 5],
                    "color": "cyan",
                    "bidirectional": True,
                }]
            },
            "goal": {"type": "reach_position", "target": [6, 6]},
            "max_steps": 50,
        })
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(spec)
        backend.reset(seed=1)
        colors = {tp.visual_color for tp in backend.env.teleporters.values()}
        assert colors == {"cyan"}

    def test_arrival_does_not_bounce_back(self, teleporter_spec):
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(teleporter_spec)
        backend.reset(seed=42)
        _, _, _, _, state, _ = backend.step(MiniGridActions.MOVE_FORWARD)
        assert state.agent_position == (5, 5)
        _, _, _, _, state, _ = backend.step(MiniGridActions.MOVE_FORWARD)
        assert state.agent_position == (6, 5)

    def test_return_trip_after_leaving_the_pad(self, teleporter_spec):
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(teleporter_spec)
        backend.reset(seed=42)
        backend.step(MiniGridActions.MOVE_FORWARD)  # (1,1) -> warp to (5,5)
        backend.step(MiniGridActions.MOVE_FORWARD)  # leave pad to (6,5)
        backend.step(MiniGridActions.TURN_LEFT)
        backend.step(MiniGridActions.TURN_LEFT)
        _, _, _, _, state, _ = backend.step(MiniGridActions.MOVE_FORWARD)  # re-enter (5,5)
        assert state.agent_position == (2, 1)

    def test_bfs_uses_the_portal_shortcut(self):
        from gridworld.baselines import plan_bfs_path

        spec = TaskSpecification.from_dict({
            "task_id": "portal_shortcut",
            "seed": 1,
            "difficulty_tier": 1,
            "maze": {
                "dimensions": [8, 8],
                "walls": [[3, 1], [4, 1]],
                "start": [1, 1],
                "goal": [6, 1],
            },
            "mechanisms": {
                "teleporters": [{
                    "id": "p1",
                    "position_a": [2, 1],
                    "position_b": [5, 1],
                    "bidirectional": True,
                }]
            },
            "goal": {"type": "reach_position", "target": [6, 1]},
            "max_steps": 50,
        })
        path = plan_bfs_path(spec)
        assert path.success
        assert path.positions[-1] == (6, 1)
        assert len(path.actions) == 2

    @pytest.mark.skipif(not _M7_PORTALS.exists(), reason="local M7 example not present")
    def test_m7_portal_example_loads_and_warps(self):
        spec = TaskSpecification.from_json(str(_M7_PORTALS))
        ok, errors = spec.validate()
        assert ok, errors
        assert {p.color for p in spec.mechanisms.portals} == {"purple", "cyan"}
        backend = MiniGridBackend(render_mode="rgb_array")
        backend.configure(spec)
        backend.reset(seed=0)
        # start (1,1) facing east; walk to purple pad at (4,4) is not one step,
        # but both pads must exist on the grid.
        assert len(backend.env.teleporters) == 4
        colors = {tp.visual_color for tp in backend.env.teleporters.values()}
        assert colors == {"purple", "cyan"}
