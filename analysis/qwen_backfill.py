from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable


_QWEN_BACKFILL_TARGETS = {
    ("standard", "validation_10_v02_winding_corridor"): "cond_massive",
    ("standard", "conditional_s_s5_14x14_corridor_1"): "cond_prompt",
    ("minimal", "conditional_s_s5_14x14_corridor_1"): "cond_prompt",
}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _row_key(row: dict) -> tuple:
    return (
        row.get("task_id"),
        row.get("agent_or_model"),
        row.get("seed"),
        row.get("prompt_variant"),
    )


def _target_cell(row: dict) -> str:
    key = (str(row.get("prompt_variant")), str(row.get("task_id")))
    try:
        return _QWEN_BACKFILL_TARGETS[key]
    except KeyError as exc:
        raise ValueError(f"Unexpected Qwen backfill row: {key}") from exc


def merge_qwen_backfill(
    source_root: str | Path,
    target_sweep_root: str | Path = "artifacts-pulled/cond-sweep-20260704-qwen",
) -> dict[str, int]:
    """Merge finalized 3-row Qwen backfill artifacts into the existing Qwen cells.

    The A100 backfill is run as one small combined job for scheduling efficiency, but
    analysis expects the rows under their original cells:
    ``cond_massive`` for baseline-thinking and ``cond_prompt`` for prompt ablations.
    """
    source_root = Path(source_root)
    target_sweep_root = Path(target_sweep_root)
    rows = _read_jsonl(source_root / "episode_runs.jsonl")
    if len(rows) != 3:
        raise ValueError(f"Expected exactly 3 backfill rows, got {len(rows)} from {source_root}")

    merged: dict[str, int] = {}
    by_cell: dict[str, list[dict]] = {}
    for row in rows:
        by_cell.setdefault(_target_cell(row), []).append(row)

    for cell, new_rows in by_cell.items():
        dest_root = target_sweep_root / cell
        existing_path = dest_root / "episode_runs.jsonl"
        existing_rows = _read_jsonl(existing_path)
        replacements = {_row_key(row): row for row in new_rows}
        out_rows = [replacements.pop(_row_key(row), row) for row in existing_rows]
        out_rows.extend(replacements.values())
        _write_jsonl(existing_path, out_rows)
        merged[cell] = len(new_rows)

        for row in new_rows:
            rel = Path(row["raw_output_ref"])
            src_run = source_root / rel.parent
            dst_run = dest_root / rel.parent
            if not src_run.exists():
                raise FileNotFoundError(f"Missing source run directory: {src_run}")
            if dst_run.exists():
                shutil.rmtree(dst_run)
            shutil.copytree(src_run, dst_run)

            task_id = row["task_id"]
            src_task = source_root / "tasks" / task_id
            dst_task = dest_root / "tasks" / task_id
            if src_task.exists() and not dst_task.exists():
                shutil.copytree(src_task, dst_task)

    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge the 3-row Qwen backfill into pulled results.")
    parser.add_argument("source_root", help="Finalized backfill artifact root.")
    parser.add_argument(
        "--target-sweep-root",
        default="artifacts-pulled/cond-sweep-20260704-qwen",
        help="Existing pulled Qwen sweep root.",
    )
    args = parser.parse_args(argv)
    merged = merge_qwen_backfill(args.source_root, args.target_sweep_root)
    print(json.dumps(merged, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
