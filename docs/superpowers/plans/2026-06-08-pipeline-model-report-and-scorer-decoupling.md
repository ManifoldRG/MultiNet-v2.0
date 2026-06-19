# Per-model Report + Scorer/Interface Decoupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the solver/scorer path runnable without importing the `interface` stack, and emit one machine-readable JSON report per model separate from the scorer-calibration artifacts.

**Architecture:** Approach 3 (lazy imports) — move three `interface` imports into the functions that use them so Stages 1–2 load no interface code. Add a pure `model_report` aggregator to `pipeline/reports.py` and wire it into `_write_aggregate` to write `artifacts/reports/<run_set_id>/models/<model>.json`.

**Tech Stack:** Python 3.10, pytest, stdlib (`subprocess`, `json`, `statistics`, `collections.defaultdict`), numpy (already used by reports).

**Spec:** `docs/superpowers/specs/2026-06-08-pipeline-model-report-and-scorer-decoupling-design.md`

---

## File Structure

- `tests/test_import_isolation.py` — **create**. Subprocess-based tests asserting `scorer`, `pipeline.episode_metrics`, and `scripts.run_pipeline` import no `interface` modules.
- `scorer/runtime.py` — **modify**. Lazy `token_count_from_record` import inside `_sum_record_tokens`.
- `pipeline/episode_metrics.py` — **modify**. Lazy `token_count_from_record` import inside `episode_token_count`.
- `scripts/run_pipeline.py` — **modify**. Lazy `ExperimentConfig` (in `_condition_configs`) and `run_episode` (in `_run_one_model`) imports; wire per-model reports into `_write_aggregate`.
- `pipeline/reports.py` — **modify**. Add `_summary` helper and `model_report` function.
- `tests/test_reports.py` — **modify**. Unit test for `model_report`.
- `tests/test_run_pipeline.py` — **modify**. E2E assertion that the per-model report file is written.

---

## Task A1: Make `import scorer` interface-free

**Files:**
- Create: `tests/test_import_isolation.py`
- Modify: `scorer/runtime.py` (remove line 11 import; add lazy import in `_sum_record_tokens`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_import_isolation.py`:

```python
"""The solver/scorer path must not import the heavy interface stack.

Each check runs in a fresh interpreter (subprocess) because the rest of the
suite imports `interface`, which would pollute sys.modules within one process.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _pulls_interface(module: str) -> bool:
    code = (
        f"import {module}, sys; "
        "hit = [m for m in sys.modules if m == 'interface' or m.startswith('interface.')]; "
        "print('IFACE' if hit else 'CLEAN')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    return "IFACE" in result.stdout


def test_scorer_import_is_interface_free():
    assert not _pulls_interface("scorer"), "import scorer pulled in interface"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_import_isolation.py::test_scorer_import_is_interface_free -v`
Expected: FAIL (`scorer/runtime.py` imports `interface.telemetry` at module load, so `interface` is in `sys.modules`).

- [ ] **Step 3: Make the scorer import lazy**

In `scorer/runtime.py`, delete the module-level import (line 11):

```python
from interface.telemetry import token_count_from_record
```

Then add it inside `_sum_record_tokens` as the first line of the function body:

```python
def _sum_record_tokens(records: Any, kind: str | None = None) -> int | None:
    from interface.telemetry import token_count_from_record

    if not isinstance(records, list):
        return None
    total = 0
    found = False
    for item in records:
        if not isinstance(item, dict):
            continue
        if kind is not None and item.get("kind") != kind:
            continue
        item_tokens = token_count_from_record(item)
        if item_tokens is not None:
            total += item_tokens
            found = True
    return total if found else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_import_isolation.py::test_scorer_import_is_interface_free tests/test_scoring_system.py -q`
Expected: PASS (import isolation holds; scorer behavior unchanged).

- [ ] **Step 5: Commit**

```bash
git add tests/test_import_isolation.py scorer/runtime.py
git commit -m "Make scorer import interface-free via lazy telemetry import"
```

---

## Task A2: Make `import pipeline.episode_metrics` interface-free

**Files:**
- Modify: `tests/test_import_isolation.py` (add one test)
- Modify: `pipeline/episode_metrics.py` (remove line 18 import; add lazy import in `episode_token_count`)

- [ ] **Step 1: Add the failing test**

Append to `tests/test_import_isolation.py`:

```python
def test_episode_metrics_import_is_interface_free():
    assert not _pulls_interface("pipeline.episode_metrics"), (
        "import pipeline.episode_metrics pulled in interface"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_import_isolation.py::test_episode_metrics_import_is_interface_free -v`
Expected: FAIL (`pipeline/episode_metrics.py` imports `interface.telemetry` at module load).

- [ ] **Step 3: Make the import lazy**

In `pipeline/episode_metrics.py`, delete the module-level import (line 18):

```python
from interface.telemetry import token_count_from_record
```

Then add it as the first line of `episode_token_count`:

```python
def episode_token_count(episode: dict[str, Any]) -> Optional[int]:
    """Sum token usage over ``kind == "query"`` transcript records."""
    from interface.telemetry import token_count_from_record

    total = 0
    found = False
    for rec in episode.get("transcript", []):
        if not isinstance(rec, dict) or rec.get("kind") != "query":
            continue
        count = token_count_from_record(rec)
        if count is not None:
            total += count
            found = True
    return total if found else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_import_isolation.py tests/test_episode_metrics.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_import_isolation.py pipeline/episode_metrics.py
git commit -m "Make episode_metrics import interface-free via lazy telemetry import"
```

---

## Task A3: Make `import scripts.run_pipeline` interface-free (Stage 1–2)

**Files:**
- Modify: `tests/test_import_isolation.py` (add one test)
- Modify: `scripts/run_pipeline.py` (defer `ExperimentConfig` and `run_episode` imports)

- [ ] **Step 1: Add the failing test**

Append to `tests/test_import_isolation.py`:

```python
def test_run_pipeline_import_is_interface_free():
    assert not _pulls_interface("scripts.run_pipeline"), (
        "import scripts.run_pipeline pulled in interface"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_import_isolation.py::test_run_pipeline_import_is_interface_free -v`
Expected: FAIL (`run_pipeline` imports `interface.config` and `pipeline.run_stage3` at module load).

- [ ] **Step 3: Defer the interface imports**

In `scripts/run_pipeline.py`, delete these two module-level imports (lines 29 and 36):

```python
from interface.config import ExperimentConfig
```
```python
from pipeline.run_stage3 import run_episode
```

The return annotation `list[tuple[str, ExperimentConfig]]` on `_condition_configs` stays valid because the module already has `from __future__ import annotations` (annotations are strings, not evaluated).

Add the `ExperimentConfig` import inside `_condition_configs`:

```python
def _condition_configs(conditions: Optional[str]) -> list[tuple[str, ExperimentConfig]]:
    from interface.config import ExperimentConfig

    if not conditions:
        return [("default", ExperimentConfig())]
    if conditions not in CONDITION_SETS:
        raise ValueError(
            f"Unknown --conditions {conditions!r}; available: {sorted(CONDITION_SETS)}."
        )
    return list(iter_condition_configs(conditions, ExperimentConfig()))
```

Add the `run_episode` import inside `_run_one_model`, as the first line of the function body (before `condition_configs = _condition_configs(conditions)`):

```python
    from pipeline.run_stage3 import run_episode

    condition_configs = _condition_configs(conditions)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_import_isolation.py tests/test_run_pipeline.py -q`
Expected: PASS (all three isolation checks clean; pipeline behavior unchanged).

- [ ] **Step 5: Commit**

```bash
git add tests/test_import_isolation.py scripts/run_pipeline.py
git commit -m "Defer interface imports so run_pipeline Stage 1-2 is interface-free"
```

---

## Task B1: `model_report` aggregator

**Files:**
- Modify: `pipeline/reports.py` (add `_summary` and `model_report`)
- Test: `tests/test_reports.py` (add unit test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_reports.py`:

```python
def test_model_report_aggregates_per_model():
    rows = [
        _row(task_id="a", agent_or_model="m1", experiment="test1",
             success=True, optimality_ratio=1.0, steps=3, optimal_steps=3, tokens=10),
        _row(task_id="b", agent_or_model="m1", experiment="test1",
             success=False, optimality_ratio=0.0, steps=9, optimal_steps=3, tokens=20),
        _row(task_id="a", agent_or_model="m2", experiment="test1",
             success=True, optimality_ratio=0.5, steps=6, optimal_steps=3, tokens=5),
    ]
    composites = {
        ("a", "m1", 0, "default", "default"): 0.4,
        ("b", "m1", 0, "default", "default"): 0.0,
        ("a", "m2", 0, "default", "default"): 0.2,
    }

    rep = reports.model_report(rows, composites, "m1", "rs")
    assert rep["schema_version"] == "0.1.0"
    assert rep["model_id"] == "m1"
    assert rep["run_set_id"] == "rs"
    assert rep["provisional"] is True
    assert rep["run_count"] == 2
    assert rep["task_count"] == 2
    assert rep["overall"]["success_rate"] == 0.5
    assert rep["overall"]["optimality_ratio_mean"] == 1.0  # successful runs only
    assert rep["overall"]["tokens_total"] == 30.0
    assert rep["overall"]["composite_mean"] == 0.2  # mean(0.4, 0.0)
    assert "test1" in rep["by_experiment"]
    assert "default" in rep["by_prompt_variant"]
    assert len(rep["tasks"]) == 2
    assert {t["task_id"] for t in rep["tasks"]} == {"a", "b"}

    # A second model is fully independent (no collision).
    rep2 = reports.model_report(rows, composites, "m2", "rs")
    assert rep2["run_count"] == 1
    assert rep2["overall"]["success_rate"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reports.py::test_model_report_aggregates_per_model -v`
Expected: FAIL with `AttributeError: module 'pipeline.reports' has no attribute 'model_report'`.

- [ ] **Step 3: Implement `_summary` and `model_report`**

Append to `pipeline/reports.py` (after `mechanism_ordering_pairs`):

```python
def _summary(
    rows: list[dict[str, Any]], composites: dict[tuple, Optional[float]]
) -> dict[str, Any]:
    """Aggregate model-performance metrics over a set of run rows."""
    opt = [
        float(r["optimality_ratio"])
        for r in rows
        if r.get("success") and r.get("optimality_ratio") is not None
    ]
    tokens = [int(r["tokens"]) for r in rows if r.get("tokens") is not None]
    comps = [
        c for c in (composites.get(_run_key(r)) for r in rows) if c is not None
    ]
    return {
        "n": len(rows),
        "success_rate": _mean([float(bool(r.get("success"))) for r in rows]),
        "optimality_ratio_mean": _mean(opt),
        "optimality_ratio_median": _median(opt),
        "steps_mean": _mean([float(r.get("steps", 0)) for r in rows]),
        "tokens_mean": _mean([float(t) for t in tokens]),
        "tokens_total": float(sum(tokens)) if tokens else None,
        "composite_mean": _mean([float(c) for c in comps]),
    }


def model_report(
    run_rows: list[dict[str, Any]],
    composites: dict[tuple, Optional[float]],
    model_id: str,
    run_set_id: str,
) -> dict[str, Any]:
    """Machine-readable per-model performance report.

    Provisional: the raw metrics (success/steps/optimality/tokens) are
    meaningful now, but composite fields are placeholders until the scorer is
    tuned. Shares one schema across models so an external tool can compare them.
    """
    rows = [r for r in run_rows if r.get("agent_or_model") == model_id]

    def _group(key: str) -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            buckets[str(r.get(key))].append(r)
        return {name: _summary(group, composites) for name, group in buckets.items()}

    return {
        "schema_version": "0.1.0",
        "model_id": model_id,
        "run_set_id": run_set_id,
        "backend": rows[0].get("backend", "minigrid") if rows else "minigrid",
        "seeds": sorted({r.get("seed") for r in rows if r.get("seed") is not None}),
        "task_count": len({r.get("task_id") for r in rows}),
        "run_count": len(rows),
        "provisional": True,
        "overall": _summary(rows, composites),
        "by_experiment": _group("experiment"),
        "by_prompt_variant": _group("prompt_variant"),
        "tasks": [
            {
                "task_id": r.get("task_id"),
                "experiment": r.get("experiment"),
                "condition": r.get("condition"),
                "prompt_variant": r.get("prompt_variant"),
                "seed": r.get("seed"),
                "success": bool(r.get("success")),
                "steps": r.get("steps"),
                "optimal_steps": r.get("optimal_steps"),
                "optimality_ratio": r.get("optimality_ratio"),
                "path_choice": r.get("path_choice"),
                "tokens": r.get("tokens"),
                "composite": composites.get(_run_key(r)),
            }
            for r in rows
        ],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_reports.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/reports.py tests/test_reports.py
git commit -m "Add per-model report aggregator (model_report)"
```

---

## Task B2: Wire per-model reports into the pipeline

**Files:**
- Modify: `scripts/run_pipeline.py` (`_write_aggregate`)
- Test: `tests/test_run_pipeline.py` (add E2E assertion)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run_pipeline.py`:

```python
def test_pipeline_writes_per_model_report(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    artifacts = tmp_path / "artifacts"

    payloads = run_pipeline(
        manifest_path=manifest_path,
        experiment="test1",
        agent=ReplayAgent(v01_empty_room_trajectory()),
        agent_name="replay-stub",
        seeds=[0],
        artifacts_root=artifacts,
        run_set_id="smoke",
    )

    report_path = artifacts / "reports" / "smoke" / "models" / "replay-stub.json"
    assert report_path.exists()
    rep = json.loads(report_path.read_text())
    assert rep["schema_version"] == "0.1.0"
    assert rep["model_id"] == "replay-stub"
    assert rep["provisional"] is True
    assert rep["run_count"] == 1
    assert "overall" in rep and "by_experiment" in rep and "tasks" in rep
    assert payloads["model_reports"]["replay-stub"]["run_count"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_pipeline.py::test_pipeline_writes_per_model_report -v`
Expected: FAIL (`models/replay-stub.json` not written; `payloads` has no `model_reports` key).

- [ ] **Step 3: Wire the reports into `_write_aggregate`**

In `scripts/run_pipeline.py`, replace the tail of `_write_aggregate`:

```python
    for name, payload in payloads.items():
        (report_dir / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payloads
```

with:

```python
    for name, payload in payloads.items():
        (report_dir / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Per-model reports: machine-readable, one file per model, kept separate
    # from the scorer-calibration ("tuning") artifacts above.
    models_dir = report_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_reports: dict[str, Any] = {}
    for model_id in sorted({str(r.get("agent_or_model")) for r in run_rows}):
        report = reports.model_report(run_rows, composites, model_id, run_set_id)
        (models_dir / f"{_sanitize(model_id)}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        model_reports[model_id] = report
    payloads["model_reports"] = model_reports
    return payloads
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_run_pipeline.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_pipeline.py tests/test_run_pipeline.py
git commit -m "Write a machine-readable per-model report per run set"
```

---

## Task C: Full-suite verification

- [ ] **Step 1: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: PASS (all prior tests plus the new import-isolation, model_report, and per-model-report tests).

- [ ] **Step 2: Confirm the solver path is genuinely interface-free**

Run: `python -m pytest tests/test_import_isolation.py -v`
Expected: 3 PASS (`scorer`, `pipeline.episode_metrics`, `scripts.run_pipeline` all CLEAN).

- [ ] **Step 3: No commit** (verification only; nothing changed).

---

## Notes for the implementer

- `_mean`/`_median` return `None` for empty inputs — leave those `None`s in the report (they mean "no data"), do not coerce to 0.
- `tokens` is `None` for runs without query telemetry; it is skipped in `tokens_mean`/`tokens_total`, never coerced to 0.
- The import-isolation tests MUST use a subprocess; importing these modules in-process from a suite that also imports `interface` would always show `interface` in `sys.modules`.
- Do not relocate `interface/telemetry.py` — that neutral-module extraction is deferred to a later PR (see spec "Deferred").
