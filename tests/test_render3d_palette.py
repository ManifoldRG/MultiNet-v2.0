from __future__ import annotations

import pytest

from gridworld.render3d import palette
from gridworld.task_spec import TaskSpecification
from maze_test_utils import MAZE_JSON_DIR


def test_rgba_normalises_gray_and_case():
    assert palette.rgba("Gray") == palette.rgba("grey")


def test_unknown_colour_raises_instead_of_falling_back():
    with pytest.raises(ValueError, match="unknown colour"):
        palette.rgba("chartreuse")


def test_dim_scales_rgb_and_keeps_alpha():
    red = palette.rgba("red")
    dimmed = palette.dim(red, 0.5)
    assert dimmed[:3] == pytest.approx(tuple(c * 0.5 for c in red[:3]))
    assert dimmed[3] == red[3]


def test_agent_colour_is_not_a_spec_colour():
    spec_colours = {palette.rgba(n) for n in ("red", "green", "blue", "purple", "yellow", "grey")}
    assert palette.AGENT not in spec_colours


def test_every_corpus_colour_is_in_the_palette():
    paths = sorted(MAZE_JSON_DIR.rglob("*.json"))
    if not paths:
        pytest.skip("maze corpus submodule not checked out")
    for path in paths:
        mech = TaskSpecification.from_json(str(path)).mechanisms
        names = (
            {k.color for k in mech.keys}
            | {d.requires_key for d in mech.doors}
            | {s.color for s in mech.switches}
            | {g.color for g in mech.gates}
        )
        for name in names:
            palette.rgba(name)  # raises on an unknown colour
