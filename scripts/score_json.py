#!/usr/bin/env python3
"""CLI for scoring task and run JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gridworld.scoring import (
    ScorerConfig,
    compute_runtime_score,
    load_scorer_config,
    score_runtime_file,
    score_task_file,
)


def _json_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        else:
            files.append(path)
    return files


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _load_config(args: argparse.Namespace) -> ScorerConfig:
    return load_scorer_config(args.config)


def _static(args: argparse.Namespace) -> int:
    config = _load_config(args)
    files = _json_files(args.inputs)
    if not files:
        raise FileNotFoundError("No JSON files matched the static scoring inputs")

    output_root = Path(args.output_dir) if args.output_dir else None
    multiple = len(files) > 1
    for task_path in files:
        if output_root is None:
            target_dir = task_path.with_suffix("").with_name(f"{task_path.stem}_score")
        elif multiple:
            target_dir = output_root / task_path.stem
        else:
            target_dir = output_root

        canonical, static_score = score_task_file(
            task_path,
            output_dir=target_dir,
            config=config,
        )
        print(
            f"{static_score.task_id}: static_score={static_score.static_score:.3f}, "
            f"beatable={static_score.is_beatable}, optimal_steps={canonical.optimal_steps} -> {target_dir}"
        )
    return 0


def _runtime(args: argparse.Namespace) -> int:
    config = _load_config(args)
    output_path = Path(args.output) if args.output else Path(args.run).with_name("run_score.json")

    if args.static_score and args.canonical_paths:
        score = score_runtime_file(
            args.run,
            static_score_path=args.static_score,
            canonical_paths_path=args.canonical_paths,
            output_path=output_path,
            config=config,
            difficulty_max_static_score=args.difficulty_max_static_score,
        )
    else:
        if not args.task:
            raise ValueError(
                "Runtime scoring needs --static-score and --canonical-paths, "
                "or --task so those artifacts can be computed."
            )
        canonical, static_score = score_task_file(
            args.task,
            output_dir=args.artifact_dir,
            config=config,
        )
        run = _load_json(args.run)
        score = compute_runtime_score(
            run,
            static_score=static_score,
            canonical_paths=canonical,
            config=config,
            difficulty_max_static_score=args.difficulty_max_static_score,
        )
        _dump_json(output_path, score.to_dict())

    print(f"{score.task_id}: runtime_score={score.composite:.3f} -> {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score MultiNet task and run JSON artifacts.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional scorer config JSON/YAML path. Defaults to gridworld/scorer_config.json.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    static_parser = subparsers.add_parser(
        "static",
        help="Write canonical_paths.json and scored_static.json for task JSON files.",
    )
    static_parser.add_argument("inputs", nargs="+", help="Task JSON files or directories.")
    static_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for score artifacts. Multiple inputs are written under per-file subdirectories.",
    )
    static_parser.set_defaults(func=_static)

    runtime_parser = subparsers.add_parser(
        "runtime",
        help="Write run_score.json for one run/episode JSON artifact.",
    )
    runtime_parser.add_argument("run", help="Run or episode JSON file.")
    runtime_parser.add_argument("--task", default=None, help="Task JSON file, used when static artifacts are omitted.")
    runtime_parser.add_argument("--static-score", default=None, help="Existing scored_static.json path.")
    runtime_parser.add_argument("--canonical-paths", default=None, help="Existing canonical_paths.json path.")
    runtime_parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Optional directory to write computed static artifacts when --task is used.",
    )
    runtime_parser.add_argument("--output", default=None, help="Output run_score.json path.")
    runtime_parser.add_argument(
        "--difficulty-max-static-score",
        type=float,
        default=None,
        help="Optional suite max static score for difficulty normalization.",
    )
    runtime_parser.set_defaults(func=_runtime)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
