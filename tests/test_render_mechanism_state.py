"""Regression tests: image_only frames must encode dynamic mechanism state.

Root cause of the R1 (r1-20260717) render investigation: ``Switch.render``
drew a fixed circle with no branch on ``is_active``, so a switch's on/off
state was invisible in image_only observations (and ``Switch.encode`` omitted
``is_active``, so the MiniGrid tile cache would not bust on a state change
either). Doors and gates were verified to render their open/closed state
correctly; these tests pin all three so the switch bug cannot silently return.
"""
import numpy as np

from gridworld.custom_env import CustomMiniGridEnv


def _env():
    env = CustomMiniGridEnv(
        width=8,
        height=8,
        max_steps=100,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        goal_pos=(6, 6),
        render_mode="rgb_array",
        highlight=False,
    )
    env.reset(seed=0)
    return env


def _cell_crop(env, x, y):
    frame = env.render()
    cell_px = frame.shape[1] // env.width
    return frame[y * cell_px : (y + 1) * cell_px, x * cell_px : (x + 1) * cell_px].copy()


def _mean_abs_diff(before, after):
    return float(np.abs(before.astype(int) - after.astype(int)).mean())


def test_switch_render_changes_with_is_active():
    """A switch toggled on must not render pixel-identical to its off state."""
    env = _env()
    env.place_switch(5, 1, "s1", controls=["g1"], switch_type="toggle", initial_state="off")
    before = _cell_crop(env, 5, 1)
    env.grid.get(5, 1).activate()
    after = _cell_crop(env, 5, 1)
    assert _mean_abs_diff(before, after) > 0.5, (
        "switch on/off renders identically — is_active not drawn (R1 render bug)"
    )


def test_switch_encode_reflects_is_active():
    """encode() must change with is_active so the MiniGrid tile cache busts."""
    env = _env()
    env.place_switch(5, 1, "s1", controls=["g1"], switch_type="toggle", initial_state="off")
    sw = env.grid.get(5, 1)
    off = sw.encode()
    sw.activate()
    on = sw.encode()
    assert off != on, "switch encode() identical on/off — tile cache will not refresh"


def test_door_render_changes_with_open_state():
    env = _env()
    env.place_door(3, 1, "red", is_locked=True)
    before = _cell_crop(env, 3, 1)
    door = env.grid.get(3, 1)
    door.is_locked = False
    door.is_open = True
    after = _cell_crop(env, 3, 1)
    assert _mean_abs_diff(before, after) > 0.5, "door open/closed renders identically"


def test_gate_render_changes_with_open_state():
    env = _env()
    env.place_gate(3, 1, "g1", is_open=False, color="purple")
    before = _cell_crop(env, 3, 1)
    env.grid.get(3, 1).is_open = True
    after = _cell_crop(env, 3, 1)
    assert _mean_abs_diff(before, after) > 0.5, "gate open/closed renders identically"
