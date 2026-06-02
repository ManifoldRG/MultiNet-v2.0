#!/usr/bin/env python3
"""Write Stage 5 scorer reports from static and runtime JSON artifacts."""

from __future__ import annotations

import argparse

from scorer.aggregate import write_experiment_reports
from scorer.io import json_files, load_json


def _load_artifacts(paths: list[str], kind: str) -> list[dict]:
    artifacts = []
    for path in json_files(paths):
        payload = load_json(path)
        if kind == "static" and (
            "dimensions_12" in payload or "static_score" in payload
        ):
            artifacts.append(payload)
        elif kind == "runtime" and "signals" in payload and "composite" in payload:
            artifacts.append(payload)
    if not artifacts:
        raise FileNotFoundError(f"No matching {kind} scorer artifacts found")
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate MultiNet scorer artifacts.")
    parser.add_argument(
        "--static",
        nargs="+",
        required=True,
        help="Static scorer JSON files or directories.",
    )
    parser.add_argument(
        "--runtime",
        nargs="+",
        required=True,
        help="Runtime scorer JSON files or directories.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for leaderboard and experiment summary JSON files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = write_experiment_reports(
        _load_artifacts(args.static, "static"),
        _load_artifacts(args.runtime, "runtime"),
        args.output_dir,
    )
    for path in paths:
        print(f"Wrote report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
