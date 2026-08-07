<div align="center">

<img src="docs/figures/multinet_logo.png" alt="MultiNet" width="200" />
<!-- HUMAN: commit the logo asset at docs/figures/multinet_logo.png, or delete the img tag -->

# MultiNet v2.0 — Gridworld

[![Website](https://img.shields.io/badge/Website-multinet.ai-blue)](https://multinet.ai)
[![v1 Paper](https://img.shields.io/badge/CVPR%202026W-MultiNet%20v1-red)](https://openaccess.thecvf.com/content/CVPR2026W/MMFM5/papers/Guruprasad_Do_Multimodal_Foundation_Models_Truly_Generalize_Exposing_Failure_Modes_Across_CVPRW_2026_paper.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](./LICENSE)
<!-- HUMAN: re-add Paper + Discord badges when the arXiv link and invite URL exist -->

</div>

MultiNet v2.0 evaluates how VLM/LLM agents perceive, plan, and act in
procedurally generated gridworld mazes — keys, doors, switches, gates, and
decoys — from raster observations, one action at a time, under strictly
controlled prompt, observation, and query conditions.

*MultiNet is a collaborative initiative from [Manifold Research](https://www.manifoldrg.com/)
and partner institutions — see the [MultiNet v1 repository](https://github.com/ManifoldRG/MultiNet)
for the broader benchmark program.*

## 📢 Updates

- **2026-XX-XX** 🚀 **v2.0 release** — R1 evaluation of Claude Opus 4.8,
  Kimi k2.6, and Qwen3.6-27B on 50 difficulty-balanced mazes.
  <!-- HUMAN: date + release-page link -->

## 🔍 Overview

This repository provides:

1. **Task specification + validation** — mechanism chains (key→door,
   switch→gate), decoy objects, and dead-end distractors; every shipped
   maze is BFS-verified solvable, with difficulty measured in executable
   actions.
2. **A controlled evaluation harness** — prompt strategies, observation
   renderers (raster image / text summary / hybrids), query modes,
   chat-history regimes, and a progress-stall watchdog, all config-driven.
3. **Model adapters** — Anthropic (incl. Batch API), Moonshot/Kimi, and
   local vLLM backends behind one interface, with strict `FINAL_OUTPUT`
   action parsing and per-episode artifact logging.
4. **Scoring** — canonical per-run results tables and a mechanism-aware
   progress score (analysis notebooks and figures are published with the
   results, separately from this repo).
5. **Fleet tooling** — GCP provisioning/teardown with cost-safety rails
   for large sweeps.

**Headline R1 result:** all three frontier models are near the floor —
Claude Opus 4.8 solves 4/50, Kimi k2.6 and Qwen3.6-27B 1/50 each — with a
shared 2D-raster perception bottleneck expressed as three distinct failure
styles. Full analysis: results page + paper.
<!-- HUMAN: confirm the three counts against the FINAL post-Kimi-rerun tables, then link results page + arXiv -->

## 🚀 Getting Started

### Installation

```bash
git clone --recurse-submodules https://github.com/ManifoldRG/MultiNet-v2.0.git
cd MultiNet-v2.0
# already cloned without submodules? run: git submodule update --init

conda create -n multinet-v2 python=3.10 && conda activate multinet-v2
# (or: python -m venv .venv && source .venv/bin/activate)
pip install -e ".[dev,visual]"
pytest   # verify the install — no API keys or GPU needed
```

The `ogbench` submodule (~50 MB) supplies the evaluation maze corpus.

### Quickstart (cheap smoke run)

```bash
export ANTHROPIC_API_KEY=...

# 1 model x 3 mazes x 4k-token cap: exercises the full loop
# (prompting -> parsing -> stepping -> scoring -> artifacts) at minimal
# cost. Not for measurement.
python -m scripts.run_pipeline \
  --run-config gridworld/fixtures/run_config.smoke_claude_sonnet.json \
  --manifest gridworld/fixtures/manifest.smoke_eval.json \
  --seeds 0
```

Per-episode artifacts (full transcript, frames, queries, scores) land under
`artifacts/runs/<task>/<backend>/<model>/seed_<n>/<variant>/episode.json`
(git-ignored) and aggregate into `episode_runs.jsonl`.

### Reproduce the R1 evaluation (paid)

Reproducing R1 runs three models over the 50-maze panel — Claude and Kimi
at 64k output caps, the served Qwen tier starting at an 8k cap with a
phase-2 widen for cap-hitters — and requires an Anthropic key (Opus 4.8), a
Moonshot key, and a locally served Qwen3.6-27B vLLM endpoint (A100-class
GPU). **Expect real API spend.** See [RUNME.md](./RUNME.md) for the full
operator guide (fleet sweeps, scoring, budgeting).

```bash
export ANTHROPIC_API_KEY=... MOONSHOT_API_KEY=...
python -m scripts.run_pipeline \
  --run-config gridworld/fixtures/run_config.r1.json \
  --manifest gridworld/fixtures/manifest.r1_balanced_03.json \
  --seeds 0
```

### Evaluate your own model

Implement an agent in `interface/agents/` exposing
`generate(messages) -> Reply` (see `interface/agents/claude.py` and
`interface/agents/reply.py`), add a provider branch for it in
`scripts/run_pipeline.py`'s `_build_agent_from_spec`, register that
provider name in a run config, and the harness handles prompting, parsing,
stepping, scoring, and artifacts.

## 🗺️ Repository structure

| Path | Contents |
|---|---|
| `gridworld/` | task specs, runtime env, validator, BFS planners, fixtures |
| `interface/` | episode runner, prompt assembly, parsing, model agents |
| `prompting_experiments/prompt_templates/` | every prompt string (none inline) |
| `scorer/` | static + runtime scoring, per-run reports |
| `scripts/` | local + distributed run pipelines |
| `ogbench/` | submodule: the procgen maze corpus the R1 manifests resolve |
| `mazes/` | validation + one-shot example mazes used by tests and docs |
| `tests/` | pytest suite (1000+ tests) |
| `docs/` | design documentation ([index](./docs/README.md)) |

Published results and analysis live in a separate results repository — this
repo is the harness. Running it writes artifacts locally (git-ignored).

**A note on evaluation data:** the current maze panels are deliberately
public (here and in the `ogbench` fork) for reproducibility of the R1
results. Evaluation panels will be rotated for the next release cycle, so
treat the current mazes as reproducibility artifacts, not held-out data.

## ⚠️ Limitations

- Single-agent, fully synthetic 2D gridworld — results speak to raster
  perception + sequential decision-making, not general embodiment.
- R1 runs one seed per cell; per-cell variance is not characterized.
- Difficulty is measured in executable actions; mechanism-specific effects
  are partially confounded with path length.
- The shared 2D-raster perception bottleneck dominates current results:
  models are near the floor, which compresses between-model contrasts.

## 🙏 Acknowledgments

MultiNet v2.0 builds on [OGBench](https://github.com/seohongpark/ogbench)
(MIT License, © 2024 OGBench Authors); we vendor a fork at
[ManifoldRG/ogbench](https://github.com/ManifoldRG/ogbench) with
maze-generation and correctness fixes (see `ogbench/LICENSE`). The runtime
builds on [MiniGrid](https://github.com/Farama-Foundation/Minigrid) and
[Gymnasium](https://github.com/Farama-Foundation/Gymnasium).

## 📜 Citation

If you use MultiNet v2.0 in your research, please cite:

```bibtex
@misc{multinet_v2_2026,
  title  = {TODO — v2.0 paper title},
  author = {TODO — author list},
  year   = {2026},
  note   = {arXiv link TBD}
}

@inproceedings{guruprasad2026multinet,
  title     = {Do Multimodal Foundation Models Truly Generalize?
               Exposing Failure Modes Across Embodied Tasks},
  author    = {Guruprasad et al.},
  booktitle = {CVPR Workshops (MMFM5)},
  year      = {2026}
}
```
<!-- HUMAN: v2 BibTeX (title/authors/link); replace 'Guruprasad et al.' with the full v1 author list -->

## 🤝 Contributing & contact

Issues and PRs welcome. For benchmark submissions, collaboration, or
evaluation services, reach us via [multinet.ai](https://multinet.ai).

Released under the [MIT License](./LICENSE).
