from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

Coord = Tuple[int, int]


class Backbone(str, Enum):
    WINDING_CORRIDOR = "winding_corridor"
    DENSE_MAZE = "dense_maze"


class LogicChain(str, Enum):
    NONE = "none"
    KD = "kd"
    SG = "sg"
    KS = "ks"
    SK = "sk"
    KK = "kk"
    KSK = "ksk"


class DistractorMode(str, Enum):
    NONE = "none"
    WRONG_KEYS = "wrong_keys"
    WRONG_SWITCHES = "wrong_switches"
    DISTRACTOR_CHAIN = "distractor_chain"


@dataclass
class WindingCorridorParams:
    # False is the horizontal lane map; True is that map with x and y swapped.
    vertical: bool = False


@dataclass
class DenseMazeParams:
    add_loops: bool = False
    loop_count: int = 0


@dataclass
class MazeGenSpec:
    backbone: Backbone
    logic_chain: LogicChain
    difficulty_tier: int
    grid_width: int
    grid_height: int
    seed: int
    distractor_mode: DistractorMode = DistractorMode.NONE
    max_distractors: int = 0
    backbone_params: object = None

    def rng(self):
        import random
        return random.Random(self.seed)