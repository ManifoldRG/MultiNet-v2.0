<p align="center">
  <kbd>
  <img src="assets/multinet_logo.png" alt="MultiNet Logo" style="height:200px; border-radius:50%;">
  <h1 align="center" style="display: inline-block; vertical-align: middle; margin-left: 20px;">MultiNet 2.0 Preview: Goal Progress Decays with Task Horizon for Frontier VLMs in Interactive 2D Environments</h1>
  </kbd>
</p>

<p align="center">
  <a href="https://multinet.ai/"><img src="https://img.shields.io/badge/Website-009688?style=flat-square&logo=googlechrome&logoColor=white" alt="Website"></a>
  <a href="https://www.fig.inc/multinet-v2-preview/"><img src="https://img.shields.io/badge/Technical%20Report-8B0000?style=flat-square&logo=googledocs&logoColor=white" alt="Technical Report"></a>
  <a href="https://github.com/ManifoldRG/MultiNet"><img src="https://img.shields.io/badge/MultiNet%20v1.0-181717?style=flat-square&logo=github&logoColor=white" alt="MultiNet v1.0"></a>
  <a href="https://discord.gg/Rk4gAq5aYr"><img src="https://img.shields.io/badge/Contribute-5865F2?style=flat-square&logo=discord&logoColor=white" alt="Contribute"></a>
</p>

### MultiNet is a collaborative initiative with contributions from leading research teams at institutions like:

<p align="center">
  <a href="https://www.fig.inc/" target="_blank">
    <kbd>
    <img src="assets/fig_logo.png" alt="Fig Logo" height="40">
    </kbd>
  </a>
  <a href="https://www.manifoldrg.com/" target="_blank">
    <kbd>
    <img src="assets/manifold_logo.png" alt="Manifold Research Logo" height="40">
    </kbd>
  </a>
  <a href="https://www.gatech.edu/" target="_blank">
    <kbd>
    <img src="assets/gt_logo.png" alt="Georgia Tech Logo" style="height:40px; border-radius:50%;">
    </kbd>
  </a>
  <a href="https://www.tufts.edu/" target="_blank">
    <kbd>
    <img src="assets/tufts_logo.jpg" alt="Tufts Logo" style="height:40px; border-radius:50%;">
    </kbd>
  </a>
</p>

### _Interested in evaluating your agent on long horizon tasks, or collaborating on research? [Let's work together](https://sparkly-broccoli-3c7.notion.site/3bf4b1d3c487800596bbe4a150962cc0)_

<p align="center">
  <img src="assets/r1_failure_reels.gif" alt="Failure replay reels for Claude Opus 4.8, Kimi K2.6 and Qwen3.6-27B" width="100%">
  <br>
  <em>Claude Opus 4.8, Kimi K2.6 and Qwen 3.6 27B models failing on 2D mazes.</em>
</p>

## 📢 News

- 🌀 2026-08-26: A Preview of our Multi-Domain Agentic Benchmark - We evaluate 3 frontier VLMs on 50 2D mazes to understand how, where, and why they break in an environment that requires exploration, planning, long-horizon action taking, and causal reasoning. Read our technical report [here](https://www.fig.inc/multinet-v2-preview/).
- 🎓 2026-04-03: Paper accepted at CVPR 2026! Our work has been accepted at the [MMFM Workshop](https://mmfm-workshop.github.io/) at CVPR 2026! Read our paper [here](https://arxiv.org/abs/2512.11315).
- 🌟 2025-10-13: Multinet v1.0 - We release our most comprehensive benchmark yet - evaluating a SoTA VLM, VLA, and generalist model on a wide variety of multimodal understanding and action datasets. Read more [here](https://multinet.ai/static/pages/Multinetv1.html)
- 🏅 2025-06-10: Paper accepted at ICML 2025! Our paper detailing the Open-Source contributions of Multinet that benefit the AI community has been accepted at the [CodeML Workshop](https://codeml-workshop.github.io/codeml2025/) at ICML 2025! Read our paper [here](https://multinet.ai/static/pdfs/An%20Open-Source%20Software%20Toolkit%20&%20Benchmark%20Suite%20for%20the%20Evaluation%20and%20Adaptation%20of%20Multimodal%20Action%20Models.pdf).
- 🏆 2025-05-22: Multinet v0.2 - We systematically profile state-of-the-art VLAs and VLMs to understand how they perform in procedurally generated OOD game environments! Read more about our release [here](https://multinet.ai/static/pages/Multinetv02.html)
- 🎉 2024-11-08: We release the first version of MultiNet where we profiled SoTA VLMs and VLAs on real-world robotics tasks - Multinet v0.1! Check our [release page](https://multinet.ai/static/pages/Multinetv01.html) for more details.
- 🚀 2024-03-22: Introducing Multinet! A new generalist benchmark to evaluate Vision-Language & Action models. Learn more [here](https://multinet.ai)

## 🔍 A Preview of our Multi-Domain Agentic Benchmark

With MultiNet v2.0 we aim to build interactive environments that are proxies for real-world scenarios. However, at the same time we are keen to keep the setup controllable, which will allow us to deterministically vary parameters in our environment in order to make it easier or more difficult for models to succeed in.

The capabilities we aim to benchmark are long-horizon action taking and causal reasoning, which involves various sub-capabilities such as planning, action execution, error recovery, visual object association, and so much more. A simple underlying substrate that brings all these aspects together for an environment and benchmarking task is a maze with mechanisms. Additionally, mazes with mechanisms are projectable into multiple domains: the same maze can be re-rendered in language or 3D or many other domains, quantifying cross-domain generalization. In this release, MultiNet 2.0 Preview: Interactive 2D Mazes, we evaluated 3 frontier VLMs on 50 2D mazes.

## 🧩 What we built

- **The environment:** 8×8 to 14×14 [MiniGrid](https://github.com/Farama-Foundation/Minigrid) mazes with an action space containing 6 valid actions: turn left, turn right, move forward, pickup, toggle, and done. The agent must navigate corridors, dead ends, distractors and decoys, operate mechanisms in the right order and reach a goal tile.
- **A validator and BFS oracle:** every maze is confirmed solvable, with checks for mechanism necessity, chain ordering, and distractor safety. The oracle yields the exact optimal action sequence from any reachable state, giving objective difficulty, partial credit, and the ability to label a single move as strictly wrong.
- **An evaluation harness:** a config-driven episode runner (prompt assembly, strict action parsing, per-episode artifact logging, a progress-stall watchdog, difficulty-relative step caps), model adapters behind one interface, mechanism-aware scoring, and the distributed run infrastructure that executed the evaluation across a fleet of VMs and GPUs.
- **An ablation-derived protocol:** extensive experiments were run across 540 episodes to finalize the evaluation protocol for the final run on 50 mazes.

## 📊 A peek into the results

We evaluated **Claude Opus 4.8** (xhigh thinking), **Kimi k2.6** (thinking), and **Qwen3.6-27B** (thinking) on 50 difficulty-balanced mazes, with an equal 64k output-token budget.

<div align="center">
<table>
  <thead>
    <tr>
      <th></th>
      <th>Claude Opus 4.8</th>
      <th>Kimi k2.6</th>
      <th>Qwen3.6-27B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Mazes solved (/50)</td>
      <td align="center">4</td>
      <td align="center">1</td>
      <td align="center">1</td>
    </tr>
    <tr>
      <td>Mean action progress</td>
      <td align="center">0.19</td>
      <td align="center">0.23</td>
      <td align="center">0.23</td>
    </tr>
  </tbody>
</table>
</div>

**6 solves out of 150 episodes. 45 of the 50 mazes were solved by no model at all.** These are puzzles a person who has never seen one solves in a few minutes. Try out some of the mazes [here](https://multinet.ai/#play-the-maze) and see how you fare!

<p align="center">
  <img src="assets/r1_progress_grid.png" alt="Progress score per maze × model" width="100%">
  <br>
  <em>Progress per maze (columns) per model (rows); stars mark the six solves.</em>
</p>

For a deeper dive, read our [technical report](https://www.fig.inc/multinet-v2-preview/).

## 🚀 Quickstart

```bash
git clone https://github.com/ManifoldRG/MultiNet-v2.0.git
cd MultiNet-v2.0

conda create -n multinet-v2 python=3.10 && conda activate multinet-v2
# (or: python -m venv .venv && source .venv/bin/activate)
pip install -e ".[dev,visual]"
```

Mazes are declarative JSON task specifications. Validate every example spec in the repo and rank them by difficulty:

```bash
python -m gridworld.task_validator
```

```
  [PASS] tier3_key_switch_001: optimal=30 steps, mechanisms=4, score=70.61
  ...
=== Summary: 16/16 tasks beatable ===
```

To build your own maze, copy a spec from `gridworld/tasks/`, edit the layout and mechanisms, then validate and render it:

```python
from PIL import Image

from gridworld.task_spec import TaskSpecification
from gridworld.task_validator import compute_difficulty
from gridworld.backends.minigrid_backend import MiniGridBackend

spec = TaskSpecification.from_json("gridworld/tasks/tier3/key_switch_001.json")

report = compute_difficulty(spec)
print(report.is_beatable, report.optimal_steps, report.mechanism_count)

backend = MiniGridBackend()
backend.configure(spec)
backend.reset(seed=0)
Image.fromarray(backend.render()).save("maze.png")
```

`compute_difficulty` runs the BFS oracle: if your maze is unsolvable, has a decorative mechanism, or has a distractor that can strand the agent, it will tell you.

## 📁 Repository structure

| Path | Contents |
|---|---|
| `gridworld/` | task specification, maze validator, BFS oracle, MiniGrid + MultiGrid backends |
| `interface/` | episode runner, prompt assembly, action parsing, model adapters |
| `prompting_experiments/` | every prompt template used in the protocol sweep |
| `scorer/` | static and runtime scoring, mechanism-aware progress |
| `demo/` | the playable maze demo embedded on the website |
| `scripts/` | evaluation pipeline entrypoints and run tooling |
| `deploy/` | distributed run infrastructure: VM and GPU fleet provisioning, teardown, and cost-safety rails |

## 📚 MultiNet v1.0

Our previous research with MultiNet v1.0 and earlier versions all live in the [MultiNet v1.0 repository](https://github.com/ManifoldRG/MultiNet): evaluations of VLMs, VLAs, and generalist models across a wide variety of domains such as robotics, multimodal understanding, game play, and tool-calling to understand their cross-domain generalization capabilities.

## 📜 Citation

Please cite this work as:

```
Guruprasad, P., Rivera, S., Lu, H., Jain, A., Ren, H. and Sikka, H. (2026) MultiNet 2.0 Preview: Goal Progress Decays with Task Horizon for Frontier VLMs in Interactive 2D Environments. Available at: https://www.fig.inc/multinet-v2-preview
```

Or use the BibTeX citation:

```bibtex
@online{multinet_v2_preview_technical_report_2026,
  title   = {MultiNet 2.0 Preview: Goal Progress Decays with Task Horizon for Frontier VLMs in Interactive 2D Environments},
  author  = {Pranav Guruprasad and Sean Rivera and Helen Lu and Arushi Jain and Hangliang Ren and Harshvardhan Sikka},
  year    = {2026},
  url     = {www.fig.inc/multinet-v2-preview},
  note    = {MultiNet 2.0 Preview: Interactive 2D Mazes}
}
```

## 🤝 Interested in evaluating your agent on long horizon tasks, or collaborating on research?

If you build models or agents, or work on benchmarking and evaluation, we would love to hear from you - whether that means getting your model on the MultiNet 2.0 Benchmark, collaborating on the next version of the benchmark, or working with us on what comes after.

<p align="center">
  <a href="https://sparkly-broccoli-3c7.notion.site/3bf4b1d3c487800596bbe4a150962cc0">🧪 Let's work together</a>
  &nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <a href="mailto:contact@fig.inc?subject=Collaborating%20on%20MultiNet%20v2.0">✉️ Work with us</a>
  &nbsp;&nbsp;&middot;&nbsp;&nbsp;
  <a href="https://discord.gg/Rk4gAq5aYr">💬 Join the Discord</a>
</p>

Released under the [MIT License](./LICENSE).
