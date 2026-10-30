# MultiGrid hex port: R2 mechanics on a hexagonal tiling

| Field | Value |
|---|---|
| Status | In progress |
| Date | 2026-10-30 |
| Branch | `feat/multigrid-hex-port` (from `main` at 4b8b54d) |
| Goal | Run an R2-style mechanics-15 maze on a hex tiling through the same run pipeline, harness and scoring as the R2 2D run, so the only intended difference from R2 2D is the tiling |

This is the working record for the port: what the MiniGrid path does today, what the
MultiGrid path lacks, the plan, and the decisions taken on the way.

## 1. How a MiniGrid maze flows today

| Stage | Code | What it does |
|---|---|---|
| Task spec | `gridworld/task_spec.py` | `TaskSpecification` with integer `(x, y)` cells, an implicit border wall ring, mechanisms (keys, doors, switches, gates, teleporters, `death_portals` → `kill_cells`, `frozen_tiles`, `rotating_tiles` + `rotating_initial_directions`). `validate()` checks bounds, overlaps, ids. No notion of tiling (ogbench mazes carry an informational `metadata.tiling: "square"`). |
| Rulebook | `gridworld/world_model.py` | Pure `(PlannerState, action) -> PlannerState` (`apply`, `successors`, `classify_mechanic`, `shortest_plan`). Square only: `DIRECTION_VECTORS` (E, S, W, N), turns are `% 4`, rotators spin through 4 directions. |
| Env step | `gridworld/custom_env.py` | `CustomMiniGridEnv.step` reads a `PlannerState` off the grid, calls `world_model.apply`, writes it back, names the mechanic with `classify_mechanic`. One step implementation shared with BFS and the validator (PR #57). |
| Parser / backend | `gridworld/task_parser.py`, `gridworld/backends/minigrid_backend.py` | Builds the env from the spec; `MiniGridBackend` returns RGB frames and a `GridState` (carrying = key colour, `key_positions`, `freeze_remaining`, `rotator_dirs`). |
| Backend choice | `gridworld/render_settings.py`, `pipeline/run_stage3.py` | Run-config `render` block: absent = `minigrid` (2D), `mujoco3d` = 3D over a MiniGrid state backend. The label is the artifact directory and a run-hash input. |
| Frames | MiniGrid `Grid.render` + `CustomMiniGridEnv.get_frame` | 32 px tiles, 3x supersampled; grey walls, black floor with grey grid lines, green goal, MiniGrid key/door shapes, switch ball with on/off core, portal rings, skull death cell, ice tile (+ frost overlay on the agent while frozen), orange rotating tile with arrow (small red triangle when the agent stands on it), red agent triangle. |
| Prompt / actions | `interface/prompt_strategies.py`, `interface/action_space.py`, `interface/parser.py` | R2 cell: minimal prompt ("Task: Solve the maze by reaching the goal." + valid actions), `image_only`, egocentric actions `TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, PICKUP, DROP, TOGGLE, DONE`. Nothing in the image_only prompt names a direction or coordinate. |
| Episode loop | `interface/runner.py`, `interface/episode_step.py` | `EpisodeStepper` drives queries, steps the backend, records `state_before`/`state_after` snapshots, frames, feedback (`interface/feedback.py`, square `forward_cell`), facing names (`interface/coords.py`, 4 names). |
| Watchdog | `interface/progress_watchdog.py` | Progress signature over `GridState` (position, carrying, keys, doors, switches, gates, blocks, key positions). Tiling-neutral. |
| Canonical paths / scoring | `gridworld/baselines.py`, `gridworld/task_validator.py`, `scorer/` | BFS (`plan_bfs_path`) and greedy over `world_model`; Stage 2 static score, difficulty, fragility, mechanism necessity; the 3x BFS step cap (`scripts/run_pipeline._runtime_capped_spec`); runtime score from positions (tiling-neutral). |
| Artifacts | `interface/episode_log.py`, `scripts/run_pipeline.py` | `episode.json` (transcript with `state_before`/`state_after`), `frames/`, `run_inputs.json`, `run_score.json`, reports. |

## 2. Gap list (MultiGrid / hex vs the above)

1. **No tiling in the task spec.** A spec cannot say it is hex; `MultiGridBackend(tiling="hex")` silently reinterprets any square spec.
2. **Hex direction labels are wrong for the geometry.** `multigrid/tilings/hex.py` lays cells out pointy-top odd-r but labels the six axial vectors with flat-top names: "north" `(0, -1)` actually moves up-left, there is no straight north. The legacy renderer and the view cone both draw "north" as straight up, so the arrow and the move disagree.
3. **Different step semantics.** `multigrid/world.py:execute_action` is a second rulebook: pickup from the facing cell, toggle switches from the facing cell, teleporter cooldowns, hazards terminate. None of PR #57 exists there (no frozen, rotating or death tiles); the converter drops them silently and paints every portal purple. Agent starts facing north (square) instead of east.
4. **Planner is square only.** `world_model` hard-codes 4 direction vectors, so canonical BFS, the 3x step cap, difficulty, fragility and planner-graded analyses cannot see a hex maze.
5. **GridState differs.** MultiGrid reports the held key's id (MiniGrid: its colour), no `key_positions`, `freeze_remaining` or `rotator_dirs`; `door_states` / `rotator_directions` / `freeze_remaining` are not overridden.
6. **Frames look nothing like R2.** `multigrid/rendering.py` uses its own palette (light floor, dark walls, blue agent), 640x640, and no icons for the PR #57 tiles.
7. **Pipeline cannot select it.** `RenderSettings` only knows `minigrid` and `mujoco3d`; nothing rejects a hex task on the MiniGrid or 3D backend.
8. **Interface text is square.** Facing names (`coords.direction_name`), `forward_cell` (feedback), rotating-tile direction names, the cardinal action space (`MOVE_NORTH`...) and the NL coordinate explanation all assume 4 directions.
9. **No hex mazes, manifest or run config.**

## 3. Plan

1. **Spec.** `maze.tiling` in `{"square", "hex"}`, default square, serialized only when not square (square spec hashes, Stage 2 caches and run hashes stay byte-identical). Hex cells are `(x, y) = (column, row)` of the odd-r offset layout (odd rows shifted half a cell east), the layout `HexTiling` already uses and the `(col, row)` mapping `MultiGridBackend` already applies. The border ring stays a wall ring. Validation: rotating directions must be in range for the tiling (0-3 square, 0-5 hex), `metadata.tiling` must agree with `maze.tiling` when present, unknown tilings are rejected.
2. **One direction convention.** Index 0 is east and directions count clockwise on screen on both tilings: square E, S, W, N (MiniGrid); hex E, SE, SW, W, NW, NE. TURN_RIGHT adds one, TURN_LEFT subtracts one, the agent starts facing east. Shared helper `gridworld/tilings.py` (direction count, names, neighbour of `(x, y)`). `HexTiling` is relabelled to the same order with geometry-correct names; the view cone and legacy renderer angles follow.
3. **Planner.** `TaskPlanningContext` takes the tiling from the spec; `world_model` asks it for the neighbour in a direction and the direction count (turns and rotator spins are mod 6 on hex). No other rule changes, so every PR #57 semantic (freeze swallows actions and spins rotators, rides, death resets the maze, portals send you on, illegal actions still spin) carries over unchanged.
4. **MultiGrid backend on the rulebook.** For square and hex the backend steps through `world_model` exactly as `CustomMiniGridEnv.step` does: read a `PlannerState` off the multigrid `WorldState`, `apply`, write it back, `classify_mechanic`, MiniGrid's reward/termination/truncation. New multigrid objects carry the PR #57 tiles (frozen, rotating with a live direction, death). `GridState` is built the MiniGrid way and gains a `tiling` field (serialized only when not square). Square through MultiGrid becomes step-for-step identical to MiniGrid, which is the parity harness for the bridge. Triangle / 3464 / 488 keep the legacy `execute_action` path and reject specs that need the rulebook (hex specs, PR #57 tiles).
5. **Frames.** `gridworld/render2d.py`: a polygon renderer in the MiniGrid look (same palette, 3x supersampling, MiniGrid's own icon renders for key / switch / portal / skull / ice, door and gate rings drawn as scaled cells, agent and rotating arrows rotated by the direction angle, frost overlay, grid lines). It renders square cells too, and the square output is checked against MiniGrid's frames; the hex output then differs only by cell shape. Hex cell size 20 px (centre to vertex): a cell's area is 1039 px^2 against MiniGrid's 1024, so an 18x18 maze is 641x550 against R2's 576x576.
6. **Pipeline.** `render: {"backend": "multigrid"}` selects the MultiGrid backend (2D frames from `render2d`); label `multigrid_r<version>`, provenance in the run hash. Fail fast before any paid call: hex tasks need the multigrid backend, the minigrid and 3D backends reject hex, and the cardinal action space is rejected on hex (see decisions).
7. **Interface.** Facing / rotating direction names and `forward_cell` follow `GridState.tiling` / the spec's tiling. The minimal image_only prompt is unchanged.
8. **Mazes.** A script that produces a hex mechanics-15-style set: same red key → red door → yellow switch → gate → blue key → blue door → goal chain, one skeleton shared by five mechanic overlays (portals, death, rotating, frozen, all four), BFS checked against R2's range, each verified solvable by the planner. Manifest + run config mirroring `run_config.mechanics_15_2d.json`.
9. **Proof without API spend.** A scripted agent that replays the planner's optimal actions through `run_from_config` (`agent_factory` hook, as `tests/test_run_pipeline.py` does) on the five hex mazes: success in exactly BFS steps with `episode.json`, `frames/`, `run_score.json`.

## 4. Decisions

- **D1 Tiling lives on `maze.tiling`.** The maze geometry owns the tiling; `metadata` is informational (but must agree when it names one). Default square and omitted from `to_dict` for square, so no existing hash moves.
- **D2 Hex coordinates are odd-r offset `(column, row)`.** Existing multigrid convention (`HexTiling` cells carry `col`/`row`; `test_multigrid_hex_adapter_preserves_grid_coordinates`).
- **D3 Direction 0 = east, clockwise.** MiniGrid's convention extended to six directions, so `agent_dir`, rotator directions and `GridState.agent_direction` mean the same thing on both tilings. This fixes the hex mislabel (gap 2) rather than preserving it.
- **D4 One rulebook.** Hex runs `world_model`, not multigrid's `execute_action`, so env and planner cannot drift (the PR #57 principle). The legacy path survives only for tilings the planner does not cover.
- **D5 A spec must match its backend.** `MultiGridBackend(tiling=None)` takes the spec's tiling; an explicit square/hex tiling that disagrees with the spec is an error (a square spec no longer silently runs on hex). `test_multigrid_hex_adapter_preserves_grid_coordinates` now declares `maze.tiling: "hex"`.
- **D6 Cardinal actions are unsupported on hex.** Pointy-top hexes have no north/south neighbour, so `MOVE_NORTH`/`MOVE_SOUTH` have no faithful meaning and a six-move vocabulary would change the prompt. `build_runner` and `run_from_config` reject `action_space="cardinal"` on a hex task with an explicit error.
