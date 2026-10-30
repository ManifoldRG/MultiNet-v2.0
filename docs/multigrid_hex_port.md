# MultiGrid hex port: R2 mechanics on a hexagonal tiling

| Field | Value |
|---|---|
| Status | Implemented on the branch; not yet run on a live model |
| Date | 2026-10-30 |
| Branch | `feat/multigrid-hex-port` (from `main` at 4b8b54d) |
| Goal | Run an R2-style mechanics-15 maze on a hex tiling through the same run pipeline, harness and scoring as the R2 2D run, so the only intended difference from R2 2D is the tiling |

This is the working record for the port: what the MiniGrid path does today, what the
MultiGrid path lacks, the plan, and the decisions taken on the way. Sections 1-3 were written
before any code; section 5 records how it turned out.

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
- **D7 The multigrid render label carries a render version.** `multigrid_r<RENDER2D_VERSION>`,
  and the render provenance (tiling, frame shape, version) enters the run hash, as for 3D: a
  visual change writes new episodes instead of reusing cached ones. MiniGrid 2D hashes and paths
  are untouched.
- **D8 Hex draws narrower triangles.** MiniGrid's agent triangle (apex 45 degrees) and turntable
  arrow (50) are close to equilateral, so turned by 120 degrees they look like themselves shifted:
  fine in 90-degree steps, ambiguous in 60-degree ones (the agent facing northwest read as east;
  a southwest arrow drew as an east one). Hex uses the same shapes with a 30-37 degree apex.
  This is the one deliberate visual difference beyond the cell shape. Square keeps MiniGrid's
  shapes and stays pixel-identical.
- **D9 Hex mazes are conversions of the R2 mazes, not new ones.** Re-reading a square maze as
  odd-r hexes keeps the walls, chain and every mechanic cell, so a hex maze differs from its R2
  original only in adjacency: square neighbours stay neighbours (north/south become the diagonal
  into the next row) and each cell gains two diagonals, so "vertical" corridors zigzag and 23
  wall corners open up. A fresh hex generator would give more natural three-axis corridors but
  would change the layout too; the conversion keeps the comparison to R2 maze-for-maze. The
  converter refuses a maze whose diagonals free a key or switch the square maze needs (none of
  the five do; the portals already bypass part of the chain in M7-portals and M10, on square too).
- **D10 Turntables keep their first target.** A square arrow pointing south (north) becomes the
  hex direction that reaches the same cell: southeast (northeast) from an even row, southwest
  (northwest) from an odd one. The arrow then turns through six directions per cycle, not four.
- **D11 Carried keys report their spec colour.** The MultiGrid `GridState` reports the key's
  colour from the spec; MiniGrid reports its own palette name, which folds cyan into blue and
  brown into grey. They agree on every colour MiniGrid has (all R2 keys are red or blue).
- **D12 Partial observability keeps multigrid's look.** Under `view_cone` / `fog_of_war` the
  MultiGrid frame dims cells out of view (explored 40%, never seen dark grey), as multigrid's
  renderer did; MiniGrid's frame does not dim. Full observability (the R2 cell) is unaffected.
- **D13 Hex static scores are 2.3-2.5x square.** `search_space_size` (BFS states explored) grows
  with six facings and six-phase turntables (M10: 18,811 states against 7,428), so the hex panel
  needs `--difficulty-max-static-score 20000` (largest: hex M10, 19,230) and its composite run
  scores are not comparable to R2's (normalized by 8000). Compare success, steps and optimality
  ratio instead.

## 5. Outcome

### What runs end to end

- `maze.tiling: "hex"` specs validate, plan (BFS, greedy, validator, fragility, difficulty) and
  score through Stage 2 unchanged; `TaskPlanningContext` reads the tiling.
- `MultiGridBackend` steps square and hex through `world_model` (read, apply, write back, as
  `CustomMiniGridEnv.step`). On square it is MiniGrid step for step and pixel for pixel: while
  developing, all 15 mechanics-15 BFS routes matched MiniGrid in every `GridState`, step result
  and frame; the tests check M10, random walks and a serpentine through every mechanic.
- On hex, the env steps exactly as the planner's `apply` (random-walk parity tests in the style
  of PR #57) and every mechanic has its own test: keys/doors, switch/gate, portals, ice (actions
  swallowed while turntables keep turning), turntables (six-direction spin, rides, blocked rides),
  death (full maze reset).
- Frames: `gridworld/render2d.py`, about 28 ms per 18x18 hex frame (641x550).
- Pipeline: `render: {"backend": "multigrid"}`; the fail-fast checks (hex on minigrid/3D, cardinal
  on hex) run in `run_from_config`, `run_pipeline` and the fleet's coordinator-prepare, which
  builds multigrid units unchanged (checked: 10 units for the two-model hex config).
- Scripted proof (`tests/test_hex_pipeline_e2e.py`, M10 under `-m slow`): an agent replaying the
  planner's optimal actions through `run_from_config` solves each hex maze in exactly its BFS
  steps (step ratio 1.0, BFS cell overlap 1.0) with `episode.json` (`state_before`/`state_after`,
  tiling hex), `frames/`, `queries/`, `run_inputs.json` (3x step cap) and `run_score.json`. The
  model-facing prompt is R2's minimal one word for word.

### The hex panel (`manifest.mechanics_15_hex.json`)

| Task id | Mechanics | BFS hex / square | Static score |
|---|---|---|---|
| `mech15hex_M7_18x18_dense_kr_sg_kb_portals_0` | portals | 100 / 90 | 3,610 |
| `mech15hex_M7_18x18_dense_kr_sg_kb_death_0` | death | 130 / 121 | 2,834 |
| `mech15hex_M7_18x18_dense_kr_sg_kb_rotating_0` | rotating | 134 / 128 | 14,809 |
| `mech15hex_M7_18x18_dense_kr_sg_kb_frozen_0` | frozen | 140 / 131 | 3,156 |
| `mech15hex_M10_18x18_dense_kr_sg_kb_portals_death_rotating_frozen_0` | all four | 83 / 75 | 19,230 |

Regenerate or verify with `python -m scripts.make_hex_mechanics [--check]`; `--tasks` converts
any other mechanics-15 row.

### Launch (not run)

```bash
python -m scripts.run_pipeline \
  --run-config gridworld/fixtures/run_config.mechanics_15_hex_2d_astra.json \
  --manifest gridworld/fixtures/manifest.mechanics_15_hex.json \
  --artifacts-root artifacts/mechanics_15_hex --run-set-id mech15_hex_astra_smoke \
  --difficulty-max-static-score 20000
```

`run_config.mechanics_15_hex_2d.json` is the same with Opus 5.5 added (Opus runs on the sync API
locally, Message Batches on the fleet as in R2). Projected Astra cost from the 2D note's
measurements: ~$13 if every episode runs its BFS length, ~$41 at 2x, ~$82 at the 3x cap; the
smoke config's `spend_cap_usd` is 100.

## 6. NL and 3D

**NL (ported).** The text observation was generic enough: positions are `(row, column)` and the
only square assumptions were the compass names, `forward_cell` and the coordinate explanation.
Facing and turntable names now come from the tiling (EAST ... NORTHEAST), and on a hex task the
coords and json formats explain hex coordinates and list each cell's six neighbours by row
parity; the ascii map indents odd rows, shows the agent as `A` (its facing in the legend) and a
turntable by its arrow's compass abbreviation. A hex NL run is `run_config.mechanics_15_nl.json`'s
cell with the hex manifest and `render: {"backend": "multigrid"}` (text_only on the multigrid
backend is allowed). Not yet done: no hex NL run config, and nobody has checked whether models
read the six-neighbour explanation the way it is meant.

**3D (out of scope; refused).** `render3d/scene.check_supported` refuses `tiling='hex'`. A hex 3D
view would need: hexagonal floor tiles and wall prisms in `render3d/scene.py` (cell centres from
the odd-r layout, as `render2d._hex_geometry` computes them); door/gate/turntable meshes and the
barrier-orientation logic (`wall_cells`, `_across_is_x`), which assume square passages with
one wall axis; camera yaw for six facings (`cameras.DIRECTION_YAW` has four) and first-person
framing at 60-degree turns; the HUD compass; and `RenderSettings` letting `mujoco3d` take a
multigrid state backend (`state_backend="multigrid"` already plugs in through `get_backend`).

## 7. Open decisions

- **Converted vs generated mazes (D9).** The conversion keeps R2's layouts; a generator of native
  hex mazes (three corridor axes, no zigzag) would test hex navigation more naturally but makes
  the comparison maze-to-maze impossible. Worth a second panel if the smoke shows a gap.
- **Narrower hex triangles (D8)** are a visual difference from R2 beyond the tiling. The
  alternative, MiniGrid's exact triangles, leaves the agent's facing ambiguous at 60 degrees.
- **Static-score normalization (D13).** Hex composite scores use a 20,000 cap; if hex and square
  are ever reported on one scale, the cap (or the search-space dimension) needs a decision.
- **Cardinal actions on hex (D6)** are refused; a six-move vocabulary (MOVE_EAST,
  MOVE_NORTHEAST, ...) would be the hex equivalent if that axis is ever run on hex.
- **Triangle tiling fix.** Relabelling the hex directions exposed that the triangle tiling linked
  each triangle to a non-adjacent one across hexes; it now follows the geometry (a regression
  test checks every link shares an edge). Anyone with saved triangle-tiling results should know
  the graph changed.
