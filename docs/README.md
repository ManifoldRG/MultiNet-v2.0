# MultiNet v2.0 — Documentation Index

MultiNet v2.0 is a gridworld benchmark harness: VLM/LLM agents navigate
procedurally generated mazes (keys, doors, switches, gates) under controlled
prompt, observation, and query conditions. Start with the repository
[README](../README.md) and [RUNME](../RUNME.md); this directory holds the
design references.

## Architecture & design

| Doc | What it covers |
|---|---|
| [system_design.md](system_design.md) | System architecture: pipeline stages, backend × adapter axes, artifact DAG |
| [technical_design.md](technical_design.md) | Detailed technical design of the harness |
| [task_parser.md](task_parser.md) | Task specification format and the spec → runtime env parser |
| [gridworld_backends.md](gridworld_backends.md) | Backend overview |
| [minigrid_backend.md](minigrid_backend.md) | MiniGrid backend reference |
| [multigrid_backend.md](multigrid_backend.md) | MultiGrid backend reference (exotic tilings) |
| [interfaces.md](interfaces.md) | Model-interface contracts (note: describes the legacy `ModelInterface` stack; the canonical agent contract is `interface/agents/` — `generate(messages) -> Reply`) |
| [batch-api-lockstep-runner-design.md](batch-api-lockstep-runner-design.md) | Batch-API lockstep runner design |
| [qwen-two-tier-rerun-design.md](qwen-two-tier-rerun-design.md) | Two-tier token-cap design for served-vLLM runs |

## Where everything else lives

- **Operator guide** (install, runs, scoring, fleet): [RUNME.md](../RUNME.md)
- **Results & analysis**: published separately with the R1 results — not in
  this repository.
- **Historical campaign runbooks** (run preparation, cost projections,
  rollout checklists): relocated to the results repository's archive.
