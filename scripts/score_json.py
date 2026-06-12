#!/usr/bin/env python3
"""CLI for scoring task and run JSON artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scorer.io import dump_json, json_files, load_json
from scorer.scoring import (
    ScorerConfig,
    compute_runtime_score,
    load_scorer_config,
    score_runtime_file,
    score_task_file,
)


def _load_config(args: argparse.Namespace) -> ScorerConfig:
    return load_scorer_config(args.config)


def _static_target_dirs(files: list[Path], output_root: Path | None) -> list[Path]:
    if output_root is None:
        return [path.with_suffix("").with_name(f"{path.stem}_score") for path in files]
    if len(files) == 1:
        return [output_root]

    target_dirs = [output_root / path.stem for path in files]
    duplicates = sorted(
        {
            str(target)
            for target in target_dirs
            if target_dirs.count(target) > 1
        }
    )
    if duplicates:
        raise ValueError(
            "Static output directories collide for same-stem inputs: "
            f"{', '.join(duplicates)}. Score those inputs separately or use distinct filenames."
        )
    return target_dirs


def _default_runtime_output(run_path: str | Path) -> Path:
    path = Path(run_path)
    return path.with_name(f"{path.stem}_score.json")


def _static(args: argparse.Namespace) -> int:
    config = _load_config(args)
    files = json_files(args.inputs)
    if not files:
        raise FileNotFoundError("No JSON files matched the static scoring inputs")

    output_root = Path(args.output_dir) if args.output_dir else None
    succeeded = 0
    failed = 0
    for task_path, target_dir in zip(files, _static_target_dirs(files, output_root)):
        try:
            canonical, static_score = score_task_file(
                task_path,
                output_dir=target_dir,
                config=config,
            )
        except Exception as exc:
            failed += 1
            print(
                "static: error "
                f"input={task_path} output_dir={target_dir} "
                f"error={exc.__class__.__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue

        succeeded += 1
        print(
            "static: ok "
            f"input={task_path} task_id={static_score.task_id} "
            f"static_score={static_score.static_score:.3f} "
            f"beatable={static_score.is_beatable} "
            f"optimal_steps={canonical.optimal_steps} output_dir={target_dir}",
            flush=True,
        )

    summary = f"static: summary scored={succeeded} failed={failed} total={len(files)}"
    print(summary, file=sys.stderr if failed else sys.stdout, flush=True)
    return 1 if failed else 0


def _runtime(args: argparse.Namespace) -> int:
    config = _load_config(args)
    output_path = Path(args.output) if args.output else _default_runtime_output(args.run)
    if (args.static_score is None) != (args.canonical_paths is None):
        raise ValueError("--static-score and --canonical-paths must be provided together")
    if (
        args.difficulty_max_static_score is None
        and config.difficulty_max_static_score is None
    ):
        raise ValueError(
            "Runtime scoring needs a suite maximum. Pass --difficulty-max-static-score "
            "or set difficulty_max_static_score in scorer config."
        )

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
        run = load_json(args.run)
        score = compute_runtime_score(
            run,
            static_score=static_score,
            canonical_paths=canonical,
            config=config,
            difficulty_max_static_score=args.difficulty_max_static_score,
        )
        dump_json(output_path, score.to_dict())

    print(f"{score.task_id}: runtime_score={score.composite:.3f} -> {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score MultiNet task and run JSON artifacts.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional scorer config JSON/YAML path. Defaults to scorer/scorer_config.json.",
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
        help="Suite max static score for difficulty normalization. Required unless configured.",
    )
    runtime_parser.set_defaults(func=_runtime)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
