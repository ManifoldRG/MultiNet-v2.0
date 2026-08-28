# MultiNet v2.0 — Operator Guide

How to install, run, and score evaluations. The public front page is
[README.md](./README.md); design docs are indexed in [docs/](./docs/README.md).

## 1. Setup

```bash
git clone --recurse-submodules https://github.com/ManifoldRG/MultiNet-v2.0.git
cd MultiNet-v2.0
# already cloned without submodules? run: git submodule update --init

conda create -n multinet-v2 python=3.10 && conda activate multinet-v2
# (or: python -m venv .venv && source .venv/bin/activate)
pip install -e ".[dev,visual]"
```

The `ogbench` submodule (~50 MB) supplies the evaluation maze corpus; the
test suite and the R1 manifests both resolve mazes from it.

## 2. Verify the install

```bash
pytest                # full suite; load-sensitive perf benchmarks are excluded
pytest -m slow        # opt-in: performance/scalability benchmarks
```

The full suite needs no API keys, GPU, or network.

## 3. Run an evaluation (canonical pipeline)

Every run is a **run-config** (models + conditions) over a **manifest**
(task list). Fixtures live in `gridworld/fixtures/`.

```bash
export ANTHROPIC_API_KEY=...   # only for the providers you actually run

# Smoke run: 1 Claude model, 3 mazes, 4k token cap — verifies the whole
# loop (prompt assembly -> parsing -> stepping -> scoring -> artifacts)
# at minimal cost. Not for measurement.
python -m scripts.run_pipeline \
  --run-config gridworld/fixtures/run_config.smoke_claude_sonnet.json \
  --manifest gridworld/fixtures/manifest.smoke_eval.json \
  --difficulty-max-static-score 1000.0 \
  --seeds 0
```

Always pass an explicit `--run-config` and `--manifest`: the default
`gridworld/fixtures/manifest.json` is a browse catalog for the demo and now
includes the 42-maze R1 panel.

`--difficulty-max-static-score` is also required. The shipped scorer config
leaves the suite maximum unset on purpose (see `docs/system_design.md`), so
runtime scoring has no denominator until you supply one and the pipeline
aborts before it queries any model. `1000.0` is the value the smoke launchers
use; a measurement run wants a maximum calibrated to its own panel.

Artifacts land under
`artifacts/runs/<task>/<backend>/<model>/seed_<n>/<variant>/episode.json`
(git-ignored) and aggregate into `episode_runs.jsonl` with a per-run
`run_score.json` written by `scorer/`.

### Reproduce the R1 evaluation (paid)

The R1 cell runs three models over the 50-maze balanced panel — Claude and
Kimi at 64k output caps, the served Qwen tier starting at an 8k cap with a
phase-2 widen for cap-hitters (see the run-config's inline notes) — and it
requires an Anthropic key (Opus 4.8), a Moonshot key, and a locally served
Qwen3.6-27B vLLM endpoint (A100-class GPU), and it spends real money. Read
the run-config before launching.

```bash
export ANTHROPIC_API_KEY=... MOONSHOT_API_KEY=...
python -m scripts.run_pipeline \
  --run-config gridworld/fixtures/run_config.r1.json \
  --manifest gridworld/fixtures/manifest.r1_balanced_03.json \
  --difficulty-max-static-score 1000.0 \
  --seeds 0
```

## 4. Scoring

Scoring runs inside the pipeline (`scorer/`): static maze/difficulty scores
plus runtime episode scoring. To re-score an existing episode JSON:

```bash
multinet-score-json --help
```

## 5. Distributed / fleet runs (GCP)

`scripts/distributed_run_pipeline.py` provides coordinator/worker jobs;
`sweep_run.sh` drives fleets (provision / next-batch / run-massive /
finalize-batch / publish / teardown) with `lib/cost_safety.sh` rails.
Batch-starting subcommands **require** `BATCH_CAP` and `MAX_RUN_DURATION` —
they have no defaults by design. STOP VMs rather than deleting them, and
pull artifacts before spindown.

## 6. Evaluate your own model

Implement an agent in `interface/agents/` exposing
`generate(messages) -> Reply` (see `interface/agents/claude.py` and
`interface/agents/reply.py`), add a provider branch for it in
`scripts/run_pipeline.py`'s `_build_agent_from_spec`, register that
provider name in a run-config's `models` block, and run the pipeline. The
harness handles prompting, parsing, stepping, scoring, and artifacts.

## 7. Playable human demo (pygame)

`play_task.py` is a human-playable client wired into the same `interface/`
code the LLM pipeline uses to build observations. It fixes its
`ExperimentConfig` to the R1 cell (`image_only` observation,
`text_summary_and_last3` context window, `egocentric` action space) so a
human run mirrors what the R1 models actually saw — `Tab` opens a read-only
settings overlay showing that config, `M` shows the exact model-facing text,
and `[ / ]` steps through the current task list. When an R1 results table is
available, task browsing can be restricted to mazes that appear in it —
point it at the R1 corpus (`ogbench/ogbench/procgen/maze_jsons/<family>` or
the manifest's `r1` experiment) to get the end-of-episode comparison against
Claude/Kimi/Qwen. Without a table (e.g. a fresh clone with no
`Multinet-v2-results` checkout), any maze still plays, just without that
comparison.

```bash
python play_task.py ogbench/ogbench/procgen/maze_jsons/D1/10x10_dense_wrong_ky_kr_sg_kb_0.json
python play_task.py --tasks-dir ogbench/ogbench/procgen/maze_jsons/M1   # browse a family with [ / ]
python play_task.py --manifest gridworld/fixtures/manifest.json --experiment r1
python play_task.py --record ogbench/ogbench/procgen/maze_jsons/S4/10x10_dense_1.json
```

## Appendix: legacy local/VLM demo harness

An earlier single-machine harness predates the canonical pipeline and
remains usable for local exploration with Ollama / LM Studio VLMs and
exotic tilings. It is **not** the stack behind any published number.

```bash
python run_eval.py --model random --benchmark tiers --tier 1  # random baseline on tier tasks
python run_eval.py --model ollama --backend multigrid --tiling hex
python visualize_all_tilings.py
python -m scripts.vlm_sanity_check --model ollama  # VLM vision sanity check
```

Task tiers for this harness live in `gridworld/tasks/tier1..tier5/`;
`gridworld.task_validator.validate_all_tasks()` checks beatability.
