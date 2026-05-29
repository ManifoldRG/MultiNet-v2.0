#!/usr/bin/env python3
"""Create quick plots and CSV tables from scorer JSON artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _json_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
        else:
            files.append(path)
    return files


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _artifact_kind(data: dict[str, Any]) -> str | None:
    if "signals" in data and "composite" in data:
        return "runtime"
    if "dimensions_12" in data or "static_score" in data:
        return "static"
    return None


def _collect_rows(paths: list[str], kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _json_files(paths):
        data = _load_json(path)
        artifact_kind = _artifact_kind(data)
        if artifact_kind is None:
            continue
        if kind != "auto" and kind != artifact_kind:
            continue
        if artifact_kind == "static":
            dimensions = data.get("dimensions_12", data.get("dimensions", {}))
            rows.append(
                {
                    "kind": "static",
                    "path": str(path),
                    "task_id": data.get("task_id", path.stem),
                    "score": float(data.get("static_score", data.get("composite", 0.0))),
                    "success": bool(data.get("is_beatable", False)),
                    "dimensions": dimensions if isinstance(dimensions, dict) else {},
                }
            )
        else:
            signals = data.get("signals", {})
            rows.append(
                {
                    "kind": "runtime",
                    "path": str(path),
                    "task_id": data.get("task_id", path.stem),
                    "score": float(data.get("composite", 0.0)),
                    "success": bool(signals.get("success", False)),
                    "model_id": data.get("model_id", ""),
                    "backend": data.get("backend", ""),
                    "steps": signals.get("steps"),
                    "step_ratio": signals.get("step_ratio"),
                    "cell_overlap_bfs": signals.get("cell_overlap_bfs"),
                    "token_efficiency": signals.get("token_efficiency"),
                }
            )
    return rows


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kind",
        "task_id",
        "score",
        "success",
        "model_id",
        "backend",
        "steps",
        "step_ratio",
        "cell_overlap_bfs",
        "token_efficiency",
        "path",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _plot_rows(path: str | Path, rows: list[dict[str, Any]], title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Score visualization requires matplotlib. Install the visual extra or use --csv only."
        ) from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_rows = sorted(rows, key=lambda row: (str(row["kind"]), str(row["task_id"])))
    labels = [
        f"{row.get('model_id') or row['task_id']}\n{row['task_id']}"
        if row["kind"] == "runtime" and row.get("model_id")
        else str(row["task_id"])
        for row in sorted_rows
    ]
    scores = [float(row["score"]) for row in sorted_rows]
    colors = ["#2f7f6f" if row["kind"] == "static" else "#5d5f9f" for row in sorted_rows]

    width = max(8, min(20, 0.55 * len(sorted_rows) + 3))
    fig, ax = plt.subplots(figsize=(width, 5))
    ax.bar(range(len(sorted_rows)), scores, color=colors)
    ax.set_title(title)
    ax.set_ylabel("score")
    ax.set_xticks(range(len(sorted_rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_dimensions(path: str | Path, row: dict[str, Any]) -> None:
    dimensions = row.get("dimensions") or {}
    if not dimensions:
        raise ValueError("Dimension plots require a scored_static.json artifact")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Score visualization requires matplotlib. Install the visual extra."
        ) from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(dimensions)
    values = [float(dimensions[name]) for name in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(names)), values, color="#2f7f6f")
    ax.set_title(f"Static dimensions: {row['task_id']}")
    ax.set_ylabel("raw value")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize MultiNet scorer artifacts.")
    parser.add_argument("inputs", nargs="+", help="Score artifact JSON files or directories.")
    parser.add_argument(
        "--kind",
        choices=["auto", "static", "runtime", "dimensions"],
        default="auto",
        help="Artifact kind to visualize.",
    )
    parser.add_argument("--output", default="scores.png", help="Output plot path.")
    parser.add_argument("--csv", default=None, help="Optional CSV output path.")
    parser.add_argument("--title", default="MultiNet Scores", help="Plot title.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    row_kind = "static" if args.kind == "dimensions" else args.kind
    rows = _collect_rows(args.inputs, row_kind)
    if not rows:
        raise FileNotFoundError("No matching scorer artifacts found")

    if args.csv:
        _write_csv(args.csv, rows)
        print(f"Wrote CSV: {args.csv}")

    if args.kind == "dimensions":
        if len(rows) != 1:
            raise ValueError("--kind dimensions expects exactly one static score artifact")
        _plot_dimensions(args.output, rows[0])
    else:
        _plot_rows(args.output, rows, args.title)
    print(f"Wrote plot: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
