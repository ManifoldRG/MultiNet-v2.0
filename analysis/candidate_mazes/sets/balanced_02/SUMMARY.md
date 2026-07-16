# Balanced 02: illustrated 50-maze candidate set

The balanced candidate is designed as the general-purpose benchmark panel. It repeats controlled single-stage key-door and switch-gate comparisons across several spatial scales, includes ordered depth-2 and depth-3 chains, and still allocates substantial mass to the 80+ action tail. It is the closest thing to a single panel that can support difficulty-curve, mechanism, distractor, and long-horizon analyses together.

This is the second retained alternative, chosen to differ by at least four mazes from variant 01. Its realized allocation contains 14 mazes at 80+ actions, 11 mazes with at least one distractor option, and 9 depth-3 mazes. The exact choices below should therefore be treated as a designed experimental panel, not merely a random sample from the corpus.

![Candidate optimal-length curve](candidate_curve.png)

## Headline composition

| measure | value |
|---|---:|
| mazes | 50 |
| optimal-action range | 9–106 |
| median optimal actions | 50.5 |
| largest adjacent length gap | 12 |
| B1 mazes | 8 |
| depth-2 mazes | 12 |
| depth-3 mazes | 9 |
| distractor mazes | 11 |
| five-option mazes | 3 |
| 80+ action mazes | 14 |
| D3 length-risk mazes | 5 |

**Length clusters:** <10: 1, 10-15: 1, 16-19: 0, 20-25: 2, 26-39: 11, 40-59: 15, 60-79: 6, 80+: 14.

**Path signatures:** none: 5, K: 8, S: 16, K→K: 3, K→S: 4, S→K: 5, K→S→K: 9.

**Families:** B1: 8, D1: 3, D2: 3, D3: 5, M1: 8, M2: 8, M3: 3, M4: 3, M5: 3, M6: 3, S1: 2, S4: 1.

The two earliest clusters are intentionally navigation controls. Mechanism comparisons start in the 26–39 band and are carried through medium, long, and tail regimes. The candidate curve above shows mechanism signature by color, distractor-bearing mazes with X markers, and B1 fixtures with a `B` label.

## Controlled comparisons

### Single-stage key versus switch

M1/M2 pairs below share size, topology, and filename variant. Their one-action length differences largely reflect the executable pickup/toggle semantics rather than different spatial geometry.

| matched cell | M1 key maze | steps | M2 switch maze | steps |
|---|---|---:|---|---:|
| `10x10:dense:1` | `M1/10x10_dense_kr_1` | 37 | `M2/10x10_dense_sg_1` | 36 |
| `10x10:dense:0` | `M1/10x10_dense_kr_0` | 44 | `M2/10x10_dense_sg_0` | 43 |
| `10x10:corridor:1` | `M1/10x10_corridor_kr_1` | 45 | `M2/10x10_corridor_sg_1` | 44 |
| `14x14:dense:1` | `M1/14x14_dense_kr_1` | 69 | `M2/14x14_dense_sg_1` | 68 |
| `14x14:dense:0` | `M1/14x14_dense_kr_0` | 80 | `M2/14x14_dense_sg_0` | 79 |
| `14x14:corridor:0` | `M1/14x14_corridor_kr_0` | 85 | `M2/14x14_corridor_sg_0` | 84 |
| `14x14:corridor:1` | `M1/14x14_corridor_kr_1` | 91 | `M2/14x14_corridor_sg_1` | 90 |

### Depth-2 ordering triplets

M3/M4/M5 triplets hold geometry fixed while changing the required ordering among K→S, S→K, and K→K. These are the cleanest fixtures for separating mechanism type from dependency depth.

| matched cell | M3 K→S | steps | M4 S→K | steps | M5 K→K | steps |
|---|---|---:|---|---:|---|---:|
| `8x8:corridor:1` | `M3/8x8_corridor_kr_sg_1` | 29 | `M4/8x8_corridor_sg_kr_1` | 29 | `M5/8x8_corridor_kr_kb_1` | 30 |
| `10x10:dense:1` | `M3/10x10_dense_kr_sg_1` | 42 | `M4/10x10_dense_sg_kr_1` | 42 | `M5/10x10_dense_kr_kb_1` | 43 |
| `10x10:dense:0` | `M3/10x10_dense_kr_sg_0` | 53 | `M4/10x10_dense_sg_kr_0` | 53 | `M5/10x10_dense_kr_kb_0` | 54 |

## Distractors, five-option cases, and B1 structure

This panel contains 11 distractor mazes. D1 adds a wrong key, D2 adds both a wrong key and an inactive switch, and D3 places a yellow key-door option in a dead-end branch. Distractor counts are kept separate from path depth: a D2/M6 maze is depth 3 but has five total options.

### Five-option mazes

| rank | steps | maze | composition |
|---:|---:|---|---|
| 42 | 88 | `D2/14x14_corridor_wrong_ky_kr_inactive_sb_sg_kb_0` | K→S→K required + wrong key + inactive switch |
| 47 | 95 | `D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_0` | K→S→K required + wrong key + inactive switch |
| 49 | 106 | `D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_1` | K→S→K required + wrong key + inactive switch |

### B1 selection

B1 is treated as a protected structural stratum rather than interchangeable with M2. The chosen B1 mazes are:

| rank | steps | size/topology | maze |
|---:|---:|---|---|
| 5 | 26 | 8x8 corridor | `B1/8x8_corridor_swg_0` |
| 6 | 27 | 8x8 corridor | `B1/8x8_corridor_swg_1` |
| 13 | 36 | 10x10 dense | `B1/10x10_dense_swg_1` |
| 19 | 43 | 10x10 corridor | `B1/10x10_corridor_swg_0` |
| 31 | 68 | 14x14 dense | `B1/14x14_dense_swg_1` |
| 35 | 79 | 14x14 dense | `B1/14x14_dense_swg_0` |
| 38 | 84 | 14x14 corridor | `B1/14x14_corridor_swg_0` |
| 44 | 90 | 14x14 corridor | `B1/14x14_corridor_swg_1` |

## Long-tail allocation

The panel contains 14 mazes at 80+ estimated optimal actions. These fixtures test whether errors grow with execution horizon, whether mechanism costs compound with navigation length, and whether corridor versus dense topology changes the model's ability to preserve a plan.

| rank | steps | depth | options | signature | topology | maze |
|---:|---:|---:|---:|---|---|---|
| 37 | 80 | 1 | 1 | K | dense | `M1/14x14_dense_kr_0` |
| 38 | 84 | 1 | 1 | S | corridor | `B1/14x14_corridor_swg_0` |
| 39 | 84 | 1 | 1 | S | corridor | `M2/14x14_corridor_sg_0` |
| 40 | 85 | 1 | 1 | K | corridor | `M1/14x14_corridor_kr_0` |
| 41 | 87 | 2 | 3 | S→K | dense | `D3/14x14_dense_sg_kr_deadend_ky_dy_0` |
| 42 | 88 | 3 | 5 | K→S→K | corridor | `D2/14x14_corridor_wrong_ky_kr_inactive_sb_sg_kb_0` |
| 43 | 88 | 3 | 3 | K→S→K | corridor | `M6/14x14_corridor_kr_sg_kb_0` |
| 44 | 90 | 1 | 1 | S | corridor | `B1/14x14_corridor_swg_1` |
| 45 | 90 | 1 | 1 | S | corridor | `M2/14x14_corridor_sg_1` |
| 46 | 91 | 1 | 1 | K | corridor | `M1/14x14_corridor_kr_1` |
| 47 | 95 | 3 | 5 | K→S→K | dense | `D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_0` |
| 48 | 106 | 3 | 4 | K→S→K | dense | `D1/14x14_dense_wrong_ky_kr_sg_kb_1` |
| 49 | 106 | 3 | 5 | K→S→K | dense | `D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_1` |
| 50 | 106 | 3 | 4 | K→S→K | dense | `D3/14x14_dense_kr_sg_kb_deadend_ky_dy_1` |

## Interpretation risks and gaps

1. The corpus does not contain replicated mechanism mazes at the requested earliest lengths. Only the 9/11-action S1 anchors occupy the first cluster, there are no 16–19-action mazes, and the 20–25-action fixtures require no on-path mechanism. Controlled mechanism comparisons begin at 26 actions.
2. This panel contains 5 D3 maze(s). Prior analysis documents at least one D3 case where the BFS estimate is 31 actions but a legal 23-step model solve exists. D3 values should be treated as estimated lengths pending planner/runtime reconciliation.
3. Length, topology, and mechanism density are not fully factorial in the source corpus. The matched-pair and matched-triplet analyses should be preferred when making causal mechanism claims.
4. Two source D2 files share one task ID. The candidate generator enforces unique task IDs, so this panel does not include both, but downstream manifests should still use the packaged source path as the audit key.

## Illustrated maze-by-maze walkthrough

The catalog is ordered by BFS-estimated optimal action count. Marker terminology matches the candidate-curve plot: path depth counts required stages, while options include off-path distractors.

### <10 actions

The absolute shortest navigation anchor.

#### 01. `S1/8x8_empty_room_1` — 9 actions

![S1/8x8_empty_room_1 maze render](images/001_S1__8x8_empty_room_1.png)

- **Structure:** 8x8 empty, family S1, variant 1.
- **Mechanism path:** none; depth 0, 0 total option(s), 0 distractor option(s).
- **Experimental role:** This is a minimal empty-room spatial baseline. Its optimal path is navigation only; no mechanism stage is required on the optimal path.
- **Why selected:** It represents the absolute shortest navigation anchor and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/S1/8x8_empty_room_1.json)

### 10-15 actions

The requested early navigation anchor.

#### 02. `S1/8x8_empty_room_0` — 11 actions

![S1/8x8_empty_room_0 maze render](images/002_S1__8x8_empty_room_0.png)

- **Structure:** 8x8 empty, family S1, variant 0.
- **Mechanism path:** none; depth 0, 0 total option(s), 0 distractor option(s).
- **Experimental role:** This is a minimal empty-room spatial baseline. Its optimal path is navigation only; no mechanism stage is required on the optimal path.
- **Why selected:** It represents the requested early navigation anchor and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/S1/8x8_empty_room_0.json)

### 20-25 actions

The second early cluster, still before mechanism comparisons begin.

#### 03. `D3/10x10_dense_deadend_ky_dy_1` — 23 actions

![D3/10x10_dense_deadend_ky_dy_1 maze render](images/003_D3__10x10_dense_deadend_ky_dy_1.png)

- **Structure:** 10x10 dense, family D3, variant 1.
- **Mechanism path:** none; depth 0, 1 total option(s), 1 distractor option(s).
- **Experimental role:** This is a dead-end yellow key-door distractor condition. Its optimal path is navigation only; no mechanism stage is required on the optimal path.
- **Why selected:** It represents the second early cluster, still before mechanism comparisons begin and helps maintain the intended smooth difficulty curve.
- **Length caveat:** D3 dead-end fixture; the BFS estimate may be inflated relative to legal runtime execution.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D3/10x10_dense_deadend_ky_dy_1.json)

#### 04. `S4/10x10_dense_1` — 23 actions

![S4/10x10_dense_1 maze render](images/004_S4__10x10_dense_1.png)

- **Structure:** 10x10 dense, family S4, variant 1.
- **Mechanism path:** none; depth 0, 0 total option(s), 0 distractor option(s).
- **Experimental role:** This is a medium dense-layout navigation baseline. Its optimal path is navigation only; no mechanism stage is required on the optimal path.
- **Why selected:** It represents the second early cluster, still before mechanism comparisons begin and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/S4/10x10_dense_1.json)

### 26-39 actions

Short mechanism-capable mazes.

#### 05. `B1/8x8_corridor_swg_0` — 26 actions

![B1/8x8_corridor_swg_0 maze render](images/005_B1__8x8_corridor_swg_0.png)

- **Structure:** 8x8 corridor, family B1, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents short mechanism-capable mazes while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/8x8_corridor_swg_0.json)

#### 06. `B1/8x8_corridor_swg_1` — 27 actions

![B1/8x8_corridor_swg_1 maze render](images/006_B1__8x8_corridor_swg_1.png)

- **Structure:** 8x8 corridor, family B1, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents short mechanism-capable mazes while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/8x8_corridor_swg_1.json)

#### 07. `M1/8x8_corridor_kr_0` — 27 actions

![M1/8x8_corridor_kr_0 maze render](images/007_M1__8x8_corridor_kr_0.png)

- **Structure:** 8x8 corridor, family M1, variant 0.
- **Mechanism path:** K; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single key-door condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents short mechanism-capable mazes and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M1/8x8_corridor_kr_0.json)

#### 08. `M2/8x8_corridor_sg_1` — 27 actions

![M2/8x8_corridor_sg_1 maze render](images/008_M2__8x8_corridor_sg_1.png)

- **Structure:** 8x8 corridor, family M2, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single switch-gate condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents short mechanism-capable mazes and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M2/8x8_corridor_sg_1.json)

#### 09. `M3/8x8_corridor_kr_sg_1` — 29 actions

![M3/8x8_corridor_kr_sg_1 maze render](images/009_M3__8x8_corridor_kr_sg_1.png)

- **Structure:** 8x8 corridor, family M3, variant 1.
- **Mechanism path:** K→S; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->S depth-2 condition. Its optimal path is a key-door stage followed by a switch-gate stage.
- **Why selected:** It represents short mechanism-capable mazes and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M3/8x8_corridor_kr_sg_1.json)

#### 10. `M4/8x8_corridor_sg_kr_1` — 29 actions

![M4/8x8_corridor_sg_kr_1 maze render](images/010_M4__8x8_corridor_sg_kr_1.png)

- **Structure:** 8x8 corridor, family M4, variant 1.
- **Mechanism path:** S→K; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled S->K depth-2 condition. Its optimal path is a switch-gate stage followed by a key-door stage.
- **Why selected:** It represents short mechanism-capable mazes and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M4/8x8_corridor_sg_kr_1.json)

#### 11. `M5/8x8_corridor_kr_kb_1` — 30 actions

![M5/8x8_corridor_kr_kb_1 maze render](images/011_M5__8x8_corridor_kr_kb_1.png)

- **Structure:** 8x8 corridor, family M5, variant 1.
- **Mechanism path:** K→K; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->K depth-2 condition. Its optimal path is two sequential key-door stages.
- **Why selected:** It represents short mechanism-capable mazes and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M5/8x8_corridor_kr_kb_1.json)

#### 12. `D1/8x8_corridor_wrong_ky_kr_sg_kb_1` — 31 actions

![D1/8x8_corridor_wrong_ky_kr_sg_kb_1 maze render](images/012_D1__8x8_corridor_wrong_ky_kr_sg_kb_1.png)

- **Structure:** 8x8 corridor, family D1, variant 1.
- **Mechanism path:** K→S→K; depth 3, 4 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents short mechanism-capable mazes and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/8x8_corridor_wrong_ky_kr_sg_kb_1.json)

#### 13. `B1/10x10_dense_swg_1` — 36 actions

![B1/10x10_dense_swg_1 maze render](images/013_B1__10x10_dense_swg_1.png)

- **Structure:** 10x10 dense, family B1, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents short mechanism-capable mazes while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/10x10_dense_swg_1.json)

#### 14. `M2/10x10_dense_sg_1` — 36 actions

![M2/10x10_dense_sg_1 maze render](images/014_M2__10x10_dense_sg_1.png)

- **Structure:** 10x10 dense, family M2, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single switch-gate condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents short mechanism-capable mazes and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M2/10x10_dense_sg_1.json)

#### 15. `M1/10x10_dense_kr_1` — 37 actions

![M1/10x10_dense_kr_1 maze render](images/015_M1__10x10_dense_kr_1.png)

- **Structure:** 10x10 dense, family M1, variant 1.
- **Mechanism path:** K; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single key-door condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents short mechanism-capable mazes and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M1/10x10_dense_kr_1.json)

### 40-59 actions

Medium-length mazes where ordering and distractor effects become separable.

#### 16. `D1/10x10_dense_wrong_ky_sg_kr_1` — 42 actions

![D1/10x10_dense_wrong_ky_sg_kr_1 maze render](images/016_D1__10x10_dense_wrong_ky_sg_kr_1.png)

- **Structure:** 10x10 dense, family D1, variant 1.
- **Mechanism path:** S→K; depth 2, 3 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is a switch-gate stage followed by a key-door stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/10x10_dense_wrong_ky_sg_kr_1.json)

#### 17. `M3/10x10_dense_kr_sg_1` — 42 actions

![M3/10x10_dense_kr_sg_1 maze render](images/017_M3__10x10_dense_kr_sg_1.png)

- **Structure:** 10x10 dense, family M3, variant 1.
- **Mechanism path:** K→S; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->S depth-2 condition. Its optimal path is a key-door stage followed by a switch-gate stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M3/10x10_dense_kr_sg_1.json)

#### 18. `M4/10x10_dense_sg_kr_1` — 42 actions

![M4/10x10_dense_sg_kr_1 maze render](images/018_M4__10x10_dense_sg_kr_1.png)

- **Structure:** 10x10 dense, family M4, variant 1.
- **Mechanism path:** S→K; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled S->K depth-2 condition. Its optimal path is a switch-gate stage followed by a key-door stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M4/10x10_dense_sg_kr_1.json)

#### 19. `B1/10x10_corridor_swg_0` — 43 actions

![B1/10x10_corridor_swg_0 maze render](images/019_B1__10x10_corridor_swg_0.png)

- **Structure:** 10x10 corridor, family B1, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/10x10_corridor_swg_0.json)

#### 20. `M2/10x10_dense_sg_0` — 43 actions

![M2/10x10_dense_sg_0 maze render](images/020_M2__10x10_dense_sg_0.png)

- **Structure:** 10x10 dense, family M2, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single switch-gate condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M2/10x10_dense_sg_0.json)

#### 21. `M5/10x10_dense_kr_kb_1` — 43 actions

![M5/10x10_dense_kr_kb_1 maze render](images/021_M5__10x10_dense_kr_kb_1.png)

- **Structure:** 10x10 dense, family M5, variant 1.
- **Mechanism path:** K→K; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->K depth-2 condition. Its optimal path is two sequential key-door stages.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M5/10x10_dense_kr_kb_1.json)

#### 22. `M1/10x10_dense_kr_0` — 44 actions

![M1/10x10_dense_kr_0 maze render](images/022_M1__10x10_dense_kr_0.png)

- **Structure:** 10x10 dense, family M1, variant 0.
- **Mechanism path:** K; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single key-door condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M1/10x10_dense_kr_0.json)

#### 23. `M2/10x10_corridor_sg_1` — 44 actions

![M2/10x10_corridor_sg_1 maze render](images/023_M2__10x10_corridor_sg_1.png)

- **Structure:** 10x10 corridor, family M2, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single switch-gate condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M2/10x10_corridor_sg_1.json)

#### 24. `M1/10x10_corridor_kr_1` — 45 actions

![M1/10x10_corridor_kr_1 maze render](images/024_M1__10x10_corridor_kr_1.png)

- **Structure:** 10x10 corridor, family M1, variant 1.
- **Mechanism path:** K; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single key-door condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M1/10x10_corridor_kr_1.json)

#### 25. `M6/10x10_corridor_kr_sg_kb_1` — 48 actions

![M6/10x10_corridor_kr_sg_kb_1 maze render](images/025_M6__10x10_corridor_kr_sg_kb_1.png)

- **Structure:** 10x10 corridor, family M6, variant 1.
- **Mechanism path:** K→S→K; depth 3, 3 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->S->K depth-3 condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M6/10x10_corridor_kr_sg_kb_1.json)

#### 26. `D3/10x10_dense_kr_sg_deadend_ky_dy_0` — 53 actions

![D3/10x10_dense_kr_sg_deadend_ky_dy_0 maze render](images/026_D3__10x10_dense_kr_sg_deadend_ky_dy_0.png)

- **Structure:** 10x10 dense, family D3, variant 0.
- **Mechanism path:** K→S; depth 2, 3 total option(s), 1 distractor option(s).
- **Experimental role:** This is a dead-end yellow key-door distractor condition. Its optimal path is a key-door stage followed by a switch-gate stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Length caveat:** D3 dead-end fixture; the BFS estimate may be inflated relative to legal runtime execution.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D3/10x10_dense_kr_sg_deadend_ky_dy_0.json)

#### 27. `M3/10x10_dense_kr_sg_0` — 53 actions

![M3/10x10_dense_kr_sg_0 maze render](images/027_M3__10x10_dense_kr_sg_0.png)

- **Structure:** 10x10 dense, family M3, variant 0.
- **Mechanism path:** K→S; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->S depth-2 condition. Its optimal path is a key-door stage followed by a switch-gate stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M3/10x10_dense_kr_sg_0.json)

#### 28. `M4/10x10_dense_sg_kr_0` — 53 actions

![M4/10x10_dense_sg_kr_0 maze render](images/028_M4__10x10_dense_sg_kr_0.png)

- **Structure:** 10x10 dense, family M4, variant 0.
- **Mechanism path:** S→K; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled S->K depth-2 condition. Its optimal path is a switch-gate stage followed by a key-door stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M4/10x10_dense_sg_kr_0.json)

#### 29. `M5/10x10_dense_kr_kb_0` — 54 actions

![M5/10x10_dense_kr_kb_0 maze render](images/029_M5__10x10_dense_kr_kb_0.png)

- **Structure:** 10x10 dense, family M5, variant 0.
- **Mechanism path:** K→K; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->K depth-2 condition. Its optimal path is two sequential key-door stages.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M5/10x10_dense_kr_kb_0.json)

#### 30. `M6/10x10_dense_kr_sg_kb_1` — 58 actions

![M6/10x10_dense_kr_sg_kb_1 maze render](images/030_M6__10x10_dense_kr_sg_kb_1.png)

- **Structure:** 10x10 dense, family M6, variant 1.
- **Mechanism path:** K→S→K; depth 3, 3 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->S->K depth-3 condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M6/10x10_dense_kr_sg_kb_1.json)

### 60-79 actions

Long mazes bridging the medium regime to the extreme tail.

#### 31. `B1/14x14_dense_swg_1` — 68 actions

![B1/14x14_dense_swg_1 maze render](images/031_B1__14x14_dense_swg_1.png)

- **Structure:** 14x14 dense, family B1, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents long mazes bridging the medium regime to the extreme tail while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/14x14_dense_swg_1.json)

#### 32. `M2/14x14_dense_sg_1` — 68 actions

![M2/14x14_dense_sg_1 maze render](images/032_M2__14x14_dense_sg_1.png)

- **Structure:** 14x14 dense, family M2, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single switch-gate condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents long mazes bridging the medium regime to the extreme tail and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M2/14x14_dense_sg_1.json)

#### 33. `M1/14x14_dense_kr_1` — 69 actions

![M1/14x14_dense_kr_1 maze render](images/033_M1__14x14_dense_kr_1.png)

- **Structure:** 14x14 dense, family M1, variant 1.
- **Mechanism path:** K; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single key-door condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents long mazes bridging the medium regime to the extreme tail and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M1/14x14_dense_kr_1.json)

#### 34. `D3/14x14_dense_deadend_ky_dy_0` — 72 actions

![D3/14x14_dense_deadend_ky_dy_0 maze render](images/034_D3__14x14_dense_deadend_ky_dy_0.png)

- **Structure:** 14x14 dense, family D3, variant 0.
- **Mechanism path:** none; depth 0, 1 total option(s), 1 distractor option(s).
- **Experimental role:** This is a dead-end yellow key-door distractor condition. Its optimal path is navigation only; no mechanism stage is required on the optimal path.
- **Why selected:** It represents long mazes bridging the medium regime to the extreme tail and helps maintain the intended smooth difficulty curve.
- **Length caveat:** D3 dead-end fixture; the BFS estimate may be inflated relative to legal runtime execution.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D3/14x14_dense_deadend_ky_dy_0.json)

#### 35. `B1/14x14_dense_swg_0` — 79 actions

![B1/14x14_dense_swg_0 maze render](images/035_B1__14x14_dense_swg_0.png)

- **Structure:** 14x14 dense, family B1, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents long mazes bridging the medium regime to the extreme tail while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/14x14_dense_swg_0.json)

#### 36. `M2/14x14_dense_sg_0` — 79 actions

![M2/14x14_dense_sg_0 maze render](images/036_M2__14x14_dense_sg_0.png)

- **Structure:** 14x14 dense, family M2, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single switch-gate condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents long mazes bridging the medium regime to the extreme tail and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M2/14x14_dense_sg_0.json)

### 80+ actions

The extreme long-horizon tail dominated by 14x14 structures.

#### 37. `M1/14x14_dense_kr_0` — 80 actions

![M1/14x14_dense_kr_0 maze render](images/037_M1__14x14_dense_kr_0.png)

- **Structure:** 14x14 dense, family M1, variant 0.
- **Mechanism path:** K; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single key-door condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M1/14x14_dense_kr_0.json)

#### 38. `B1/14x14_corridor_swg_0` — 84 actions

![B1/14x14_corridor_swg_0 maze render](images/038_B1__14x14_corridor_swg_0.png)

- **Structure:** 14x14 corridor, family B1, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/14x14_corridor_swg_0.json)

#### 39. `M2/14x14_corridor_sg_0` — 84 actions

![M2/14x14_corridor_sg_0 maze render](images/039_M2__14x14_corridor_sg_0.png)

- **Structure:** 14x14 corridor, family M2, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single switch-gate condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M2/14x14_corridor_sg_0.json)

#### 40. `M1/14x14_corridor_kr_0` — 85 actions

![M1/14x14_corridor_kr_0 maze render](images/040_M1__14x14_corridor_kr_0.png)

- **Structure:** 14x14 corridor, family M1, variant 0.
- **Mechanism path:** K; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single key-door condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M1/14x14_corridor_kr_0.json)

#### 41. `D3/14x14_dense_sg_kr_deadend_ky_dy_0` — 87 actions

![D3/14x14_dense_sg_kr_deadend_ky_dy_0 maze render](images/041_D3__14x14_dense_sg_kr_deadend_ky_dy_0.png)

- **Structure:** 14x14 dense, family D3, variant 0.
- **Mechanism path:** S→K; depth 2, 3 total option(s), 1 distractor option(s).
- **Experimental role:** This is a dead-end yellow key-door distractor condition. Its optimal path is a switch-gate stage followed by a key-door stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Length caveat:** D3 dead-end fixture; the BFS estimate may be inflated relative to legal runtime execution.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D3/14x14_dense_sg_kr_deadend_ky_dy_0.json)

#### 42. `D2/14x14_corridor_wrong_ky_kr_inactive_sb_sg_kb_0` — 88 actions

![D2/14x14_corridor_wrong_ky_kr_inactive_sb_sg_kb_0 maze render](images/042_D2__14x14_corridor_wrong_ky_kr_inactive_sb_sg_kb_0.png)

- **Structure:** 14x14 corridor, family D2, variant 0.
- **Mechanism path:** K→S→K; depth 3, 5 total option(s), 2 distractor option(s).
- **Experimental role:** This is a wrong-key plus inactive-switch distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Five-option stress case:** three required stages plus a wrong key and an inactive switch.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D2/14x14_corridor_wrong_ky_kr_inactive_sb_sg_kb_0.json)

#### 43. `M6/14x14_corridor_kr_sg_kb_0` — 88 actions

![M6/14x14_corridor_kr_sg_kb_0 maze render](images/043_M6__14x14_corridor_kr_sg_kb_0.png)

- **Structure:** 14x14 corridor, family M6, variant 0.
- **Mechanism path:** K→S→K; depth 3, 3 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->S->K depth-3 condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M6/14x14_corridor_kr_sg_kb_0.json)

#### 44. `B1/14x14_corridor_swg_1` — 90 actions

![B1/14x14_corridor_swg_1 maze render](images/044_B1__14x14_corridor_swg_1.png)

- **Structure:** 14x14 corridor, family B1, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/14x14_corridor_swg_1.json)

#### 45. `M2/14x14_corridor_sg_1` — 90 actions

![M2/14x14_corridor_sg_1 maze render](images/045_M2__14x14_corridor_sg_1.png)

- **Structure:** 14x14 corridor, family M2, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single switch-gate condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M2/14x14_corridor_sg_1.json)

#### 46. `M1/14x14_corridor_kr_1` — 91 actions

![M1/14x14_corridor_kr_1 maze render](images/046_M1__14x14_corridor_kr_1.png)

- **Structure:** 14x14 corridor, family M1, variant 1.
- **Mechanism path:** K; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single key-door condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M1/14x14_corridor_kr_1.json)

#### 47. `D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_0` — 95 actions

![D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_0 maze render](images/047_D2__14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_0.png)

- **Structure:** 14x14 dense, family D2, variant 0.
- **Mechanism path:** K→S→K; depth 3, 5 total option(s), 2 distractor option(s).
- **Experimental role:** This is a wrong-key plus inactive-switch distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Five-option stress case:** three required stages plus a wrong key and an inactive switch.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_0.json)

#### 48. `D1/14x14_dense_wrong_ky_kr_sg_kb_1` — 106 actions

![D1/14x14_dense_wrong_ky_kr_sg_kb_1 maze render](images/048_D1__14x14_dense_wrong_ky_kr_sg_kb_1.png)

- **Structure:** 14x14 dense, family D1, variant 1.
- **Mechanism path:** K→S→K; depth 3, 4 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/14x14_dense_wrong_ky_kr_sg_kb_1.json)

#### 49. `D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_1` — 106 actions

![D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_1 maze render](images/049_D2__14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_1.png)

- **Structure:** 14x14 dense, family D2, variant 1.
- **Mechanism path:** K→S→K; depth 3, 5 total option(s), 2 distractor option(s).
- **Experimental role:** This is a wrong-key plus inactive-switch distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Five-option stress case:** three required stages plus a wrong key and an inactive switch.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_1.json)

#### 50. `D3/14x14_dense_kr_sg_kb_deadend_ky_dy_1` — 106 actions

![D3/14x14_dense_kr_sg_kb_deadend_ky_dy_1 maze render](images/050_D3__14x14_dense_kr_sg_kb_deadend_ky_dy_1.png)

- **Structure:** 14x14 dense, family D3, variant 1.
- **Mechanism path:** K→S→K; depth 3, 4 total option(s), 1 distractor option(s).
- **Experimental role:** This is a dead-end yellow key-door distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Length caveat:** D3 dead-end fixture; the BFS estimate may be inflated relative to legal runtime execution.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D3/14x14_dense_kr_sg_kb_deadend_ky_dy_1.json)
