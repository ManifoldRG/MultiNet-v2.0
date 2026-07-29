#!/usr/bin/env python3
"""
Interactive MiniGrid Task Player — Playable Human Demo

A pygame-based interactive player for MiniGrid task JSON files that mirrors
the *exact* observation the model-facing interface (``interface/``) would
build for a given ``ExperimentConfig``. This lets a human play a maze under
the same information constraints as an LLM agent (e.g. text-only, or with a
``text_summary`` of prior activity instead of full history), to get a feel
for how hard a task actually is.

This file is just the CLI entry point: it parses args and wires together
``demo.session.MiniGridPlaySession`` (task loading/stepping/transcript logic,
no pygame) and ``demo.ui.MiniGridPlayerUI`` (the pygame window and all
drawing/input handling). See those modules for the actual implementation.

Usage:
    python play_task.py mazes/exp_maze_jsons/D1/10x10_dense_wrong_ky_kr_sg_kb_0.json
    python play_task.py mazes/exp_maze_jsons/D1/10x10_dense_wrong_ky_kr_sg_kb_0.json --record

    # Play with the same information the model would get in text-only mode,
    # with a text_summary of prior activity instead of raw history:
    python play_task.py mazes/exp_maze_jsons/D1/10x10_dense_wrong_ky_kr_sg_kb_0.json \\
        --observation text_only --context-window text_summary

    # Browse a whole directory of task files with [ / ] (non-recursive: point
    # at the leaf directory that directly contains the task JSONs). Only mazes
    # present in the sibling Multinet-v2-results R1 table are playable:
    python play_task.py --tasks-dir mazes/exp_maze_jsons/D1

    # Browse a manifest task catalog with [ / ] instead of a directory -- rows
    # can point at files in different folders, and the info panel shows each
    # row's experiment/condition/expected_mechanisms (mirrors the task
    # selection scripts/run_pipeline.py uses for real runs):
    python play_task.py --manifest gridworld/fixtures/manifest.json --experiment test1
    python play_task.py --manifest gridworld/fixtures/manifest.json --experiment r1

Controls:
    Arrow Up / W        : Move forward (egocentric) / North (cardinal)
    Arrow Down / S       : South (cardinal action space only)
    Arrow Left / A       : Turn left (egocentric) / West (cardinal)
    Arrow Right / D      : Turn right (egocentric) / East (cardinal)
    Space               : Pick up item
    T / E               : Toggle (open door, press switch) / Interact (cardinal)
    X                   : Drop item (human-only -- not in the model's action space)
    Backspace           : Wait / done (no-op)
    R                   : Reset current task
    [ / ]               : Previous / next task in the current directory / manifest
    Tab                 : Toggle the settings overlay (cycle observation/context/etc.)
    M                   : Toggle a full-screen view of the exact model-facing text
    Q                   : Quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent

# Ensure the repository root is on sys.path for gridworld/interface/demo imports
_script_dir_str = str(_SCRIPT_DIR)
if _script_dir_str not in sys.path:
    sys.path.insert(0, _script_dir_str)

from demo.session import MiniGridPlaySession
from demo.ui import MiniGridPlayerUI

from interface.config import ExperimentConfig

# Reused only for the --experiment choices list, matching real eval runs.
from scripts.run_pipeline import _EXPERIMENT_KEYWORDS


def main():
    parser = argparse.ArgumentParser(
        description="Interactive MiniGrid task player -- playable human demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "task_file",
        nargs="?",
        default=None,
        help="Path to a task JSON file (default: a small validation_10 maze, or the first "
        "task in --manifest if that's given)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record the transcript to a JSON file on exit or task switch",
    )
    parser.add_argument(
        "--tasks-dir",
        type=str,
        default=None,
        help="Directory to browse with [ / ] instead of the task file's parent directory "
        "(non-recursive: only *.json files directly inside it). Mutually exclusive with "
        "--manifest.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Browse a task-catalog manifest (e.g. gridworld/fixtures/manifest.json) with "
        "[ / ] instead of a directory -- resolves each row's 'source' file across folders "
        "and shows its catalog metadata (experiment/condition/expected_mechanisms) in the "
        "info panel, mirroring scripts/run_pipeline.py's task selection for real runs. "
        "Mutually exclusive with --tasks-dir.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=sorted(_EXPERIMENT_KEYWORDS),
        default=None,
        help="Filter --manifest to one experiment keyword (default: 'all'). Ignored without "
        "--manifest.",
    )
    parser.add_argument(
        "--observation",
        choices=["text_only", "image_text", "image_only"],
        default=ExperimentConfig.__dataclass_fields__["observation"].default,
        help="Mirrors ExperimentConfig.observation (default: image_only, today's plain grid view)",
    )
    parser.add_argument(
        "--context-window",
        dest="context_window",
        choices=["current", "last3", "text_summary", "text_summary_and_last3"],
        default=ExperimentConfig.__dataclass_fields__["context_window"].default,
        help="Mirrors ExperimentConfig.context_window; 'text_summary' shows the model's activity summary",
    )
    parser.add_argument(
        "--include-current-observation-description",
        action="store_true",
        help="Mirrors ExperimentConfig.include_current_observation_description",
    )
    parser.add_argument(
        "--observation-text-includes-facing",
        action="store_true",
        help="Mirrors ExperimentConfig.observation_text_includes_facing",
    )
    parser.add_argument(
        "--action-space",
        dest="action_space",
        choices=["egocentric", "cardinal"],
        default=ExperimentConfig.__dataclass_fields__["action_space"].default,
        help="Mirrors ExperimentConfig.action_space; cardinal remaps arrow keys to absolute N/S/E/W",
    )
    args = parser.parse_args()

    if args.manifest and args.tasks_dir:
        parser.error("--manifest and --tasks-dir are mutually exclusive.")

    config = ExperimentConfig(
        observation=args.observation,
        context_window=args.context_window,
        include_current_observation_description=args.include_current_observation_description,
        observation_text_includes_facing=args.observation_text_includes_facing,
        action_space=args.action_space,
    )

    session = MiniGridPlaySession(
        task_path=args.task_file,
        record=args.record,
        config=config,
        tasks_dir=args.tasks_dir,
        manifest=args.manifest,
        experiment=args.experiment,
    )
    ui = MiniGridPlayerUI(session)
    ui.run()


if __name__ == "__main__":
    main()
