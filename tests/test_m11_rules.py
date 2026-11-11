"""Disappear, shut, n-end teleporters, and phased patches."""

from gridworld.actions import MiniGridActions
from gridworld.backends.minigrid_backend import MiniGridBackend
from gridworld.custom_env import Wall
from gridworld.task_spec import TaskSpecification

A = MiniGridActions


def _spec(**extra):
    data = {
        "task_id": "rule",
        "seed": 1,
        "difficulty_tier": 1,
        "maze": {"dimensions": [10, 6], "walls": [], "start": [1, 1], "goal": [8, 1]},
        "mechanisms": {},
        "rules": {},
        "goal": {"type": "reach_position", "target": [8, 1]},
        "max_steps": 40,
    }
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key].update(value)
        else:
            data[key] = value
    return TaskSpecification.from_dict(data)


def _play(spec):
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    backend.reset(seed=1)
    return backend


def _go(backend, *actions):
    for action in actions:
        backend.step(int(action))
    return backend.env


def test_floor_disappears_after_n_steps_and_can_be_left():
    backend = _play(_spec(rules={"disappear_after": 2}))
    env = _go(backend, A.MOVE_FORWARD, A.TURN_LEFT, A.TURN_LEFT)
    assert env.agent_pos == (2, 1)
    assert (2, 1) in env.sealed_cells
    assert isinstance(env.grid.get(2, 1), Wall)
    _go(backend, A.MOVE_FORWARD)
    assert backend.env.agent_pos == (1, 1)


def test_opened_door_and_switch_shut():
    spec = _spec(
        rules={"shut_after": 1},
        mechanisms={
            "keys": [{"id": "kR", "position": [2, 1], "color": "red"}],
            "doors": [{"id": "DR", "position": [3, 1], "requires_key": "red"}],
            "switches": [{"id": "s1", "position": [1, 2], "controls": ["g1"]}],
            "gates": [{"id": "g1", "position": [3, 2]}],
        },
    )
    backend = _play(spec)
    env = _go(backend, A.MOVE_FORWARD, A.PICKUP, A.TOGGLE, A.TURN_LEFT)
    state = env._read_planner_state(env._planning_context())
    assert "DR" in state.shut_doors
    assert "DR" not in state.open_doors

    backend = _play(spec)
    env = _go(backend, A.TURN_RIGHT, A.MOVE_FORWARD, A.TOGGLE, A.TURN_LEFT)
    state = env._read_planner_state(env._planning_context())
    assert "s1" in state.shut_switches
    assert "g1" not in state.open_gates


def test_chain_lands_on_the_end_faced():
    spec = _spec(
        maze={"dimensions": [12, 5], "walls": [], "start": [1, 1], "goal": [10, 1]},
        goal={"type": "reach_position", "target": [10, 1]},
        mechanisms={
            "teleporters": [{
                "id": "chain",
                "positions": [[2, 1], [4, 1], [6, 1], [8, 1]],
            }]
        },
    )
    backend = _play(spec)
    _go(backend, A.MOVE_FORWARD)
    assert backend.env.agent_pos == (4, 1)
    _go(backend, A.MOVE_FORWARD, A.TURN_LEFT, A.TURN_LEFT, A.MOVE_FORWARD)
    assert backend.env.agent_pos == (2, 1)


def test_phased_cell_blocks_only_while_closed():
    spec = _spec(
        mechanisms={
            "phased_patches": [{
                "cells": [[2, 1]],
                "open_steps": 1,
                "closed_steps": 1,
            }]
        }
    )
    backend = _play(spec)
    _go(backend, A.MOVE_FORWARD)
    assert backend.env.agent_pos == (2, 1)

    backend = _play(spec)
    _go(backend, A.TURN_LEFT, A.MOVE_FORWARD)
    assert backend.env.agent_pos == (1, 1)
