"""demo.theme inlines MiniGrid's COLORS; they must stay byte-identical."""

from __future__ import annotations

import pytest


def test_inlined_palette_matches_minigrid_colors():
    constants = pytest.importorskip("minigrid.core.constants")
    from demo.theme import _MINIGRID_COLORS

    expected = {k: tuple(int(x) for x in v) for k, v in constants.COLORS.items()}
    assert _MINIGRID_COLORS == expected
