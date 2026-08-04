#!/usr/bin/env python3
"""Interactive R1 human-play demo. See ``demo.session`` / ``demo.ui`` for implementation.

    python play_task.py --manifest gridworld/fixtures/manifest.json --experiment r1
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from demo.r1_config import R1_CONFIG
from demo.session import MiniGridPlaySession
from demo.ui import MiniGridPlayerUI
from scripts.run_pipeline import _EXPERIMENT_KEYWORDS


def main() -> None:
    parser = argparse.ArgumentParser(description="MultiNet R1 human-play demo")
    parser.add_argument("task_file", nargs="?", default=None)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--tasks-dir", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--experiment", choices=sorted(_EXPERIMENT_KEYWORDS), default=None)
    args = parser.parse_args()
    if args.manifest and args.tasks_dir:
        parser.error("--manifest and --tasks-dir are mutually exclusive.")

    MiniGridPlayerUI(
        MiniGridPlaySession(
            task_path=args.task_file,
            record=args.record,
            config=dataclasses.replace(R1_CONFIG),
            tasks_dir=args.tasks_dir,
            manifest=args.manifest,
            experiment=args.experiment,
        )
    ).run()


if __name__ == "__main__":
    main()
