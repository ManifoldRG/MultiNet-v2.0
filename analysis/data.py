from __future__ import annotations
from pathlib import Path


def discover_sweeps(root: str | Path = "artifacts-pulled") -> list[Path]:
    """Every cell dir (contains a ``runs/`` child) under ``root``, skipping ``issues_``."""
    root = Path(root)
    out: list[Path] = []
    for runs in root.rglob("runs"):
        if not runs.is_dir():
            continue
        if any(part == "issues_" for part in runs.parts):
            continue
        out.append(runs.parent)
    return sorted(out)
