"""R1 model-comparison lookup for the human-play demo's end screen.

Reads the sibling Multinet-v2-results canonical metrics table so a finished
episode can show human steps vs BFS optimal vs Claude / Kimi / Qwen.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


MODEL_DISPLAY: tuple[tuple[str, str], ...] = (
    ("claude-opus-4-8", "Claude"),
    ("kimi-k2.6", "Kimi"),
    ("Qwen_Qwen3.6-27B", "Qwen"),
)

_FAILURE_BLURBS = {
    "wall_walking": "kept walking into walls",
    "spin_in_place": "spun in place",
    "mixed_churn": "got stuck looping",
    "short_loop_oscillation": "oscillated in a short loop",
    "late_stall": "stalled late in the run",
    "immediate_stall": "stalled almost immediately",
    "productive_wander_cap": "wandered until the step limit",
    "near_miss": "almost reached the goal",
    "bfs_cap_kill": "hit the step cap",
    "infra_terminated": "run ended early",
    "solved": "solved",
}

# Sibling checkout next to MultiNet-v2.0.
_DEFAULT_CSV = (
    Path(__file__).resolve().parents[1].parent
    / "Multinet-v2-results"
    / "r1-20260717"
    / "analysis"
    / "metrics"
    / "canonical_results_table.csv"
)


def r1_task_id(task_path: Path) -> str:
    """Canonical R1 id: ``r1_{parent_folder}_{stem}``."""
    return f"r1_{task_path.parent.name}_{task_path.stem}"


@dataclass(frozen=True)
class ModelResult:
    model_id: str
    display_name: str
    success: bool
    steps: int
    optimal_steps: int
    outcome: str
    primary_class: str
    stall_subclass: str

    @property
    def failure_blurb(self) -> str:
        if self.success:
            return "solved"
        for key in (self.stall_subclass, self.primary_class, self.outcome):
            if key in _FAILURE_BLURBS:
                return _FAILURE_BLURBS[key]
        raise KeyError(
            f"No failure blurb for {self.display_name}: "
            f"stall={self.stall_subclass!r} class={self.primary_class!r} outcome={self.outcome!r}"
        )

    @property
    def summary_line(self) -> str:
        if self.success:
            return f"{self.steps} steps"
        return f"{self.steps} steps  -  {self.failure_blurb}"


@dataclass(frozen=True)
class TaskComparison:
    task_id: str
    optimal_steps: int
    models: tuple[ModelResult, ...]
    grid_size: int
    topology: str
    condition: str
    chain_depth: int
    n_keys: int
    n_doors: int
    n_switches: int
    n_gates: int
    n_distractors: int


class R1ResultCatalog:
    """Index of R1 canonical results, keyed by ``r1_*`` task_id."""

    def __init__(self, csv_path: Path | None = None):
        self.csv_path = Path(csv_path) if csv_path else _DEFAULT_CSV
        if not self.csv_path.is_file():
            raise FileNotFoundError(
                f"R1 results CSV not found at {self.csv_path}. "
                "Expected sibling Multinet-v2-results checkout."
            )
        self._by_task: dict[str, list[dict]] = {}
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tid = row["task_id"].strip()
                self._by_task.setdefault(tid, []).append(row)

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._by_task

    def lookup(self, task_id: str) -> TaskComparison:
        rows = self._by_task.get(task_id)
        if rows is None:
            raise KeyError(f"Task {task_id!r} not in R1 results table ({self.csv_path})")
        return self._build(task_id, rows)

    def _build(self, task_id: str, rows: list[dict]) -> TaskComparison:
        by_model = {r["model"]: r for r in rows}
        models: list[ModelResult] = []
        optimal = 0
        for model_id, display in MODEL_DISPLAY:
            row = by_model[model_id]
            steps = int(float(row["steps"]))
            opt = int(float(row["optimal_steps"]))
            if opt:
                optimal = opt
            models.append(
                ModelResult(
                    model_id=model_id,
                    display_name=display,
                    success=row["success"] == "True",
                    steps=steps,
                    optimal_steps=opt,
                    outcome=row["outcome"],
                    primary_class=row["primary_class"],
                    stall_subclass=row["stall_subclass"],
                )
            )
        if len(models) != len(MODEL_DISPLAY):
            raise ValueError(f"{task_id}: expected {len(MODEL_DISPLAY)} models, got {len(models)}")
        meta = rows[0]
        return TaskComparison(
            task_id=task_id,
            optimal_steps=optimal,
            models=tuple(models),
            grid_size=int(float(meta["grid_size"])),
            topology=meta["topology"].strip(),
            condition=meta["condition"].strip(),
            chain_depth=int(float(meta["chain_depth"])),
            n_keys=int(float(meta["n_keys"])),
            n_doors=int(float(meta["n_doors"])),
            n_switches=int(float(meta["n_switches"])),
            n_gates=int(float(meta["n_gates"])),
            n_distractors=int(float(meta["n_distractors"])),
        )
