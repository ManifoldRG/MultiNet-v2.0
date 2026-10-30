"""Neutral tile labels for the NL rule-learning arm (``tile_labels="neutral"``).

R2's NL text names each special tile by what it does ("death portal", "frozen
tile", "rotating tile", "purple teleporter between A and B"), so a model can
plan from the names alone. With neutral labels every tile kind is shown under a
made-up word instead, and the step feedback reports what a step did, never
which tile did it (``interface.feedback``). The model has to learn each kind's
effect from what happens when it steps on one.

* Labels are pseudo-words from ``LABEL_POOL`` (novel words in the style of the
  word-learning literature: "dax", "wug", "blicket"), not shapes or colours,
  which carry their own associations (red = danger, blue-white = ice, a ring =
  a portal).
* Each maze draws its own assignment: the pool is ranked by
  ``sha256(f"{seed}:{task_id}:{word}")`` and the first words go to the kinds the
  maze has, in ``KINDS`` order. Same seed and maze, same labels, on any machine
  and Python version.
* Every kind is listed the same way: one line per cell, grouped by label in
  alphabetical order (the label draw, not the kind, sets the order), cells in
  row-major order. The rotating kind keeps its current direction ("pointing
  EAST"): the user-facing rule needs it and it names no effect.
* Portal pads are listed like any other cell. Which pad leads where is not
  shown: one step onto a pad shows it (the landing cell is reported).
* Keys, doors, the switch and the gate keep their R2 names. Only the tile kinds
  are relabelled; the world model and the maze are untouched, so every maze is
  exactly as solvable as in R2.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from interface.coords import rotating_pointing, to_row_col
from prompting_experiments.prompt_templates import observation as observation_templates

if TYPE_CHECKING:
    from gridworld.backends.base import GridState
    from gridworld.task_spec import TaskSpecification

# Changing the pool or the draw changes which label every maze gets: the
# label assignments are pinned in tests/test_tile_labels.py.
LABEL_POOL: tuple[str, ...] = ("blicket", "dax", "fep", "modi", "pilk", "toma", "wug", "zav")

# World-model tile kinds, in assignment order: kill_cells, frozen_tiles,
# rotating_tiles, teleporter pads.
KINDS: tuple[str, ...] = ("death", "frozen", "rotating", "portal")


def _kind_cells(task_spec: TaskSpecification) -> dict[str, list[tuple[int, int]]]:
    """(row, col) cells of each tile kind the maze has, row-major."""
    mech = task_spec.mechanisms
    pads = [
        pad
        for teleporter in mech.teleporters
        for pad in (teleporter.position_a, teleporter.position_b)
    ]
    cells = {
        "death": [to_row_col(c) for c in mech.kill_cells],
        "frozen": [to_row_col(c) for c in mech.frozen_tiles],
        "rotating": [to_row_col(c) for c in mech.rotating_tiles],
        "portal": [to_row_col(p) for p in pads],
    }
    return {kind: sorted(set(cells[kind])) for kind in KINDS if cells[kind]}


def draw_labels(seed: int, task_id: str, kinds: tuple[str, ...] | list[str]) -> dict[str, str]:
    """kind -> label for ``kinds``: the pool ranked by a per-maze hash."""
    if len(kinds) > len(LABEL_POOL):
        raise ValueError(f"{len(kinds)} tile kinds but only {len(LABEL_POOL)} labels")
    ranked = sorted(
        LABEL_POOL,
        key=lambda word: hashlib.sha256(f"{seed}:{task_id}:{word}".encode("utf-8")).hexdigest(),
    )
    return dict(zip(kinds, ranked))


@dataclass(frozen=True)
class TileLabeling:
    """The neutral labels of one maze, and how to render its tiles with them."""

    seed: int
    task_id: str
    label_by_kind: dict[str, str]
    cells_by_kind: dict[str, list[tuple[int, int]]]
    # (spec colour, pad a, pad b, bidirectional) per teleporter, (row, col):
    # recorded for grading, never rendered.
    portal_links: tuple[tuple[str, tuple[int, int], tuple[int, int], bool], ...] = ()

    @classmethod
    def for_task(cls, task_spec: TaskSpecification, seed: int) -> TileLabeling:
        cells = _kind_cells(task_spec)
        return cls(
            seed=int(seed),
            task_id=task_spec.task_id,
            label_by_kind=draw_labels(seed, task_spec.task_id, list(cells)),
            cells_by_kind=cells,
            portal_links=tuple(
                (
                    t.color,
                    to_row_col(t.position_a),
                    to_row_col(t.position_b),
                    bool(t.bidirectional),
                )
                for t in task_spec.mechanisms.teleporters
            ),
        )

    def tile_lines(self, task_spec: TaskSpecification, state: GridState | None) -> list[str]:
        """One line per labelled cell, by label then row-major; rotating cells
        carry their direction as of ``state`` (initial when None)."""
        lines: list[str] = []
        for kind, label in sorted(self.label_by_kind.items(), key=lambda item: item[1]):
            for row, col in self.cells_by_kind[kind]:
                if kind == "rotating":
                    lines.append(
                        observation_templates.NEUTRAL_POINTING_TILE_LINE.format(
                            label=label,
                            row=row,
                            col=col,
                            direction=rotating_pointing(task_spec, state, row, col),
                        )
                    )
                else:
                    lines.append(
                        observation_templates.NEUTRAL_TILE_LINE.format(label=label, row=row, col=col)
                    )
        return lines

    def to_dict(self) -> dict:
        """The mapping for ``episode.json``: what each label really was, where
        its cells are, and how the portal pads are wired (row, col)."""
        return {
            "mode": "neutral",
            "seed": self.seed,
            "task_id": self.task_id,
            "label_pool": list(LABEL_POOL),
            "labels": {
                label: {"effect": kind, "cells": [list(c) for c in self.cells_by_kind[kind]]}
                for kind, label in sorted(self.label_by_kind.items(), key=lambda item: item[1])
            },
            "portal_links": [
                {
                    "label": self.label_by_kind["portal"],
                    "spec_color": color,
                    "a": list(pad_a),
                    "b": list(pad_b),
                    "bidirectional": bidirectional,
                }
                for color, pad_a, pad_b, bidirectional in self.portal_links
            ],
        }
