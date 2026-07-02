"""Bare-bones run pipeline for MultiNet v2.0 (tests 1-3).

Sequential, inspectable orchestration that wires the canonical pipeline stages
over the ``interface/`` runner (Stack A) and the ``scorer/`` package:

- Stage 1: fixtures + manifest (``gridworld/fixtures/manifest.json``)
- Stage 2: static solve & score  -> ``scorer.score_task_file``
- Stage 3: runtime runs (live models) -> ``pipeline.run_stage3``
- Stage 3 instrumentation -> ``pipeline.episode_metrics``
- Stage 4: runtime score -> ``scorer.compute_runtime_score``
- Stage 5: reports -> ``pipeline.reports``

See ``scripts/run_pipeline.py`` for the orchestrator CLI.
"""
