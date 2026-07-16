# Mechanism Rich 01: illustrated 50-maze candidate set

The mechanism-rich candidate focuses on complexity that cannot be explained by length alone. It deliberately increases distractor-bearing D1/D2/D3 fixtures, depth-2 ordering problems, depth-3 K->S->K chains, and the five-option cases that combine three required stages with a wrong key and an inactive switch. It keeps one pure K/S control pair in each feasible length region, but uses the released slots for mechanism and distractor stress rather than maximum replication.

This is the lowest-loss retained variant for the profile. Its realized allocation contains 19 mazes at 80+ actions, 20 mazes with at least one distractor option, and 10 depth-3 mazes. The exact choices below should therefore be treated as a designed experimental panel, not merely a random sample from the corpus.

![Candidate optimal-length curve](candidate_curve.png)

## Headline composition

| measure | value |
|---|---:|
| mazes | 50 |
| optimal-action range | 23–106 |
| median optimal actions | 58.0 |
| largest adjacent length gap | 9 |
| B1 mazes | 8 |
| depth-2 mazes | 20 |
| depth-3 mazes | 10 |
| distractor mazes | 20 |
| five-option mazes | 6 |
| 80+ action mazes | 19 |
| D3 length-risk mazes | 5 |

**Length clusters:** <10: 0, 10-15: 0, 16-19: 0, 20-25: 1, 26-39: 12, 40-59: 13, 60-79: 5, 80+: 19.

**Path signatures:** none: 3, K: 5, S: 12, K→K: 6, K→S: 7, S→K: 7, K→S→K: 10.

**Families:** B1: 8, D1: 8, D2: 7, D3: 5, M1: 4, M2: 4, M3: 3, M4: 3, M5: 3, M6: 3, S5: 2.

This exploratory variant deliberately omits the 9/11-action S1 empty rooms. Their depth-0 control role is replaced by the 77/89-action S5 corridor pair, shifting two observations into long-horizon navigation without introducing a mechanism confound. Mechanism comparisons still begin in the 26–39 band. The candidate curve above shows mechanism signature by color, distractor-bearing mazes with X markers, and B1 fixtures with a `B` label.

## Controlled comparisons

### Single-stage key versus switch

M1/M2 pairs below share size, topology, and filename variant. Their one-action length differences largely reflect the executable pickup/toggle semantics rather than different spatial geometry.

| matched cell | M1 key maze | steps | M2 switch maze | steps |
|---|---|---:|---|---:|
| `8x8:corridor:1` | `M1/8x8_corridor_kr_1` | 28 | `M2/8x8_corridor_sg_1` | 27 |
| `10x10:corridor:1` | `M1/10x10_corridor_kr_1` | 45 | `M2/10x10_corridor_sg_1` | 44 |
| `14x14:dense:0` | `M1/14x14_dense_kr_0` | 80 | `M2/14x14_dense_sg_0` | 79 |
| `14x14:corridor:1` | `M1/14x14_corridor_kr_1` | 91 | `M2/14x14_corridor_sg_1` | 90 |

### Depth-2 ordering triplets

M3/M4/M5 triplets hold geometry fixed while changing the required ordering among K→S, S→K, and K→K. These are the cleanest fixtures for separating mechanism type from dependency depth.

| matched cell | M3 K→S | steps | M4 S→K | steps | M5 K→K | steps |
|---|---|---:|---|---:|---|---:|
| `10x10:dense:0` | `M3/10x10_dense_kr_sg_0` | 53 | `M4/10x10_dense_sg_kr_0` | 53 | `M5/10x10_dense_kr_kb_0` | 54 |
| `14x14:corridor:1` | `M3/14x14_corridor_kr_sg_1` | 92 | `M4/14x14_corridor_sg_kr_1` | 92 | `M5/14x14_corridor_kr_kb_1` | 93 |

## Distractors, five-option cases, and B1 structure

This panel contains 20 distractor mazes. D1 adds a wrong key, D2 adds both a wrong key and an inactive switch, and D3 places a yellow key-door option in a dead-end branch. Distractor counts are kept separate from path depth: a D2/M6 maze is depth 3 but has five total options.

### Five-option mazes

| rank | steps | maze | composition |
|---:|---:|---|---|
| 11 | 30 | `D2/8x8_corridor_wrong_ky_kr_inactive_sb_sg_kb_0` | K→S→K required + wrong key + inactive switch |
| 12 | 31 | `D2/8x8_corridor_wrong_ky_kr_inactive_sb_sg_kb_1` | K→S→K required + wrong key + inactive switch |
| 20 | 47 | `D2/10x10_corridor_wrong_ky_kr_inactive_sb_sg_kb_0` | K→S→K required + wrong key + inactive switch |
| 27 | 61 | `D2/10x10_dense_wrong_ky_kr_inactive_sb_sg_kb_0` | K→S→K required + wrong key + inactive switch |
| 36 | 88 | `D2/14x14_corridor_wrong_ky_kr_inactive_sb_sg_kb_0` | K→S→K required + wrong key + inactive switch |
| 47 | 95 | `D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_0` | K→S→K required + wrong key + inactive switch |

### B1 selection

B1 is treated as a protected structural stratum rather than interchangeable with M2. The chosen B1 mazes are:

| rank | steps | size/topology | maze |
|---:|---:|---|---|
| 2 | 26 | 8x8 corridor | `B1/8x8_corridor_swg_0` |
| 3 | 27 | 8x8 corridor | `B1/8x8_corridor_swg_1` |
| 13 | 36 | 10x10 dense | `B1/10x10_dense_swg_1` |
| 15 | 43 | 10x10 dense | `B1/10x10_dense_swg_0` |
| 16 | 44 | 10x10 corridor | `B1/10x10_corridor_swg_1` |
| 28 | 68 | 14x14 dense | `B1/14x14_dense_swg_1` |
| 30 | 79 | 14x14 dense | `B1/14x14_dense_swg_0` |
| 39 | 90 | 14x14 corridor | `B1/14x14_corridor_swg_1` |

## Long-tail allocation

The panel contains 19 mazes at 80+ estimated optimal actions. These fixtures test whether errors grow with execution horizon, whether mechanism costs compound with navigation length, and whether corridor versus dense topology changes the model's ability to preserve a plan.

| rank | steps | depth | options | signature | topology | maze |
|---:|---:|---:|---:|---|---|---|
| 32 | 80 | 1 | 1 | K | dense | `M1/14x14_dense_kr_0` |
| 33 | 87 | 2 | 3 | S→K | dense | `D3/14x14_dense_sg_kr_deadend_ky_dy_0` |
| 34 | 87 | 2 | 2 | K→S | dense | `M3/14x14_dense_kr_sg_0` |
| 35 | 87 | 2 | 2 | S→K | dense | `M4/14x14_dense_sg_kr_0` |
| 36 | 88 | 3 | 5 | K→S→K | corridor | `D2/14x14_corridor_wrong_ky_kr_inactive_sb_sg_kb_0` |
| 37 | 88 | 3 | 3 | K→S→K | corridor | `M6/14x14_corridor_kr_sg_kb_0` |
| 38 | 89 | 0 | 0 | none | corridor | `S5/14x14_corridor_1` |
| 39 | 90 | 1 | 1 | S | corridor | `B1/14x14_corridor_swg_1` |
| 40 | 90 | 1 | 1 | S | corridor | `M2/14x14_corridor_sg_1` |
| 41 | 91 | 1 | 1 | K | corridor | `M1/14x14_corridor_kr_1` |
| 42 | 92 | 2 | 3 | K→S | corridor | `D1/14x14_corridor_wrong_ky_kr_sg_1` |
| 43 | 92 | 2 | 4 | S→K | corridor | `D2/14x14_corridor_wrong_ky_inactive_sb_sg_kr_1` |
| 44 | 92 | 2 | 2 | K→S | corridor | `M3/14x14_corridor_kr_sg_1` |
| 45 | 92 | 2 | 2 | S→K | corridor | `M4/14x14_corridor_sg_kr_1` |
| 46 | 93 | 2 | 2 | K→K | corridor | `M5/14x14_corridor_kr_kb_1` |
| 47 | 95 | 3 | 5 | K→S→K | dense | `D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_0` |
| 48 | 96 | 2 | 3 | K→S | dense | `D1/14x14_dense_wrong_ky_kr_sg_1` |
| 49 | 97 | 2 | 2 | K→K | dense | `M5/14x14_dense_kr_kb_1` |
| 50 | 106 | 3 | 3 | K→S→K | dense | `M6/14x14_dense_kr_sg_kb_1` |

## Interpretation risks and gaps

1. The corpus does not contain replicated mechanism mazes at the requested earliest lengths. Only the 9/11-action S1 anchors occupy the first cluster, there are no 16–19-action mazes, and the 20–25-action fixtures require no on-path mechanism. Controlled mechanism comparisons begin at 26 actions.
2. This panel contains 5 D3 maze(s). Prior analysis documents at least one D3 case where the BFS estimate is 31 actions but a legal 23-step model solve exists. D3 values should be treated as estimated lengths pending planner/runtime reconciliation.
3. Length, topology, and mechanism density are not fully factorial in the source corpus. The matched-pair and matched-triplet analyses should be preferred when making causal mechanism claims.
4. Two source D2 files share one task ID. The candidate generator enforces unique task IDs, so this panel does not include both, but downstream manifests should still use the packaged source path as the audit key.

## Illustrated maze-by-maze walkthrough

The catalog is ordered by BFS-estimated optimal action count. Marker terminology matches the candidate-curve plot: path depth counts required stages, while options include off-path distractors.

### 20-25 actions

The second early cluster, still before mechanism comparisons begin.

#### 01. `D3/10x10_dense_deadend_ky_dy_1` — 23 actions

![D3/10x10_dense_deadend_ky_dy_1 maze render](images/001_D3__10x10_dense_deadend_ky_dy_1.png)

- **Structure:** 10x10 dense, family D3, variant 1.
- **Mechanism path:** none; depth 0, 1 total option(s), 1 distractor option(s).
- **Experimental role:** This is a dead-end yellow key-door distractor condition. Its optimal path is navigation only; no mechanism stage is required on the optimal path.
- **Why selected:** It represents the second early cluster, still before mechanism comparisons begin while increasing the panel's off-path choice and distractor load.
- **Length caveat:** D3 dead-end fixture; the BFS estimate may be inflated relative to legal runtime execution.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D3/10x10_dense_deadend_ky_dy_1.json)

### 26-39 actions

Short mechanism-capable mazes.

#### 02. `B1/8x8_corridor_swg_0` — 26 actions

![B1/8x8_corridor_swg_0 maze render](images/002_B1__8x8_corridor_swg_0.png)

- **Structure:** 8x8 corridor, family B1, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents short mechanism-capable mazes while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/8x8_corridor_swg_0.json)

#### 03. `B1/8x8_corridor_swg_1` — 27 actions

![B1/8x8_corridor_swg_1 maze render](images/003_B1__8x8_corridor_swg_1.png)

- **Structure:** 8x8 corridor, family B1, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents short mechanism-capable mazes while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/8x8_corridor_swg_1.json)

#### 04. `M2/8x8_corridor_sg_1` — 27 actions

![M2/8x8_corridor_sg_1 maze render](images/004_M2__8x8_corridor_sg_1.png)

- **Structure:** 8x8 corridor, family M2, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single switch-gate condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents short mechanism-capable mazes and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M2/8x8_corridor_sg_1.json)

#### 05. `D1/8x8_corridor_wrong_ky_kr_1` — 28 actions

![D1/8x8_corridor_wrong_ky_kr_1 maze render](images/005_D1__8x8_corridor_wrong_ky_kr_1.png)

- **Structure:** 8x8 corridor, family D1, variant 1.
- **Mechanism path:** K; depth 1, 2 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents short mechanism-capable mazes while increasing the panel's off-path choice and distractor load.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/8x8_corridor_wrong_ky_kr_1.json)

#### 06. `M1/8x8_corridor_kr_1` — 28 actions

![M1/8x8_corridor_kr_1 maze render](images/006_M1__8x8_corridor_kr_1.png)

- **Structure:** 8x8 corridor, family M1, variant 1.
- **Mechanism path:** K; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single key-door condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents short mechanism-capable mazes and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M1/8x8_corridor_kr_1.json)

#### 07. `D1/8x8_corridor_wrong_ky_kr_sg_0` — 28 actions

![D1/8x8_corridor_wrong_ky_kr_sg_0 maze render](images/007_D1__8x8_corridor_wrong_ky_kr_sg_0.png)

- **Structure:** 8x8 corridor, family D1, variant 0.
- **Mechanism path:** K→S; depth 2, 3 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is a key-door stage followed by a switch-gate stage.
- **Why selected:** It represents short mechanism-capable mazes while increasing the panel's off-path choice and distractor load.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/8x8_corridor_wrong_ky_kr_sg_0.json)

#### 08. `D1/8x8_corridor_wrong_ky_kr_sg_1` — 29 actions

![D1/8x8_corridor_wrong_ky_kr_sg_1 maze render](images/008_D1__8x8_corridor_wrong_ky_kr_sg_1.png)

- **Structure:** 8x8 corridor, family D1, variant 1.
- **Mechanism path:** K→S; depth 2, 3 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is a key-door stage followed by a switch-gate stage.
- **Why selected:** It represents short mechanism-capable mazes while increasing the panel's off-path choice and distractor load.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/8x8_corridor_wrong_ky_kr_sg_1.json)

#### 09. `D1/8x8_corridor_wrong_ky_sg_kr_1` — 29 actions

![D1/8x8_corridor_wrong_ky_sg_kr_1 maze render](images/009_D1__8x8_corridor_wrong_ky_sg_kr_1.png)

- **Structure:** 8x8 corridor, family D1, variant 1.
- **Mechanism path:** S→K; depth 2, 3 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is a switch-gate stage followed by a key-door stage.
- **Why selected:** It represents short mechanism-capable mazes while increasing the panel's off-path choice and distractor load.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/8x8_corridor_wrong_ky_sg_kr_1.json)

#### 10. `D1/8x8_corridor_wrong_ky_kr_kb_1` — 30 actions

![D1/8x8_corridor_wrong_ky_kr_kb_1 maze render](images/010_D1__8x8_corridor_wrong_ky_kr_kb_1.png)

- **Structure:** 8x8 corridor, family D1, variant 1.
- **Mechanism path:** K→K; depth 2, 3 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is two sequential key-door stages.
- **Why selected:** It represents short mechanism-capable mazes while increasing the panel's off-path choice and distractor load.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/8x8_corridor_wrong_ky_kr_kb_1.json)

#### 11. `D2/8x8_corridor_wrong_ky_kr_inactive_sb_sg_kb_0` — 30 actions

![D2/8x8_corridor_wrong_ky_kr_inactive_sb_sg_kb_0 maze render](images/011_D2__8x8_corridor_wrong_ky_kr_inactive_sb_sg_kb_0.png)

- **Structure:** 8x8 corridor, family D2, variant 0.
- **Mechanism path:** K→S→K; depth 3, 5 total option(s), 2 distractor option(s).
- **Experimental role:** This is a wrong-key plus inactive-switch distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents short mechanism-capable mazes while increasing the panel's off-path choice and distractor load.
- **Five-option stress case:** three required stages plus a wrong key and an inactive switch.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D2/8x8_corridor_wrong_ky_kr_inactive_sb_sg_kb_0.json)

#### 12. `D2/8x8_corridor_wrong_ky_kr_inactive_sb_sg_kb_1` — 31 actions

![D2/8x8_corridor_wrong_ky_kr_inactive_sb_sg_kb_1 maze render](images/012_D2__8x8_corridor_wrong_ky_kr_inactive_sb_sg_kb_1.png)

- **Structure:** 8x8 corridor, family D2, variant 1.
- **Mechanism path:** K→S→K; depth 3, 5 total option(s), 2 distractor option(s).
- **Experimental role:** This is a wrong-key plus inactive-switch distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents short mechanism-capable mazes while increasing the panel's off-path choice and distractor load.
- **Five-option stress case:** three required stages plus a wrong key and an inactive switch.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D2/8x8_corridor_wrong_ky_kr_inactive_sb_sg_kb_1.json)

#### 13. `B1/10x10_dense_swg_1` — 36 actions

![B1/10x10_dense_swg_1 maze render](images/013_B1__10x10_dense_swg_1.png)

- **Structure:** 10x10 dense, family B1, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents short mechanism-capable mazes while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/10x10_dense_swg_1.json)

### 40-59 actions

Medium-length mazes where ordering and distractor effects become separable.

#### 14. `D3/10x10_dense_sg_kr_deadend_ky_dy_1` — 42 actions

![D3/10x10_dense_sg_kr_deadend_ky_dy_1 maze render](images/014_D3__10x10_dense_sg_kr_deadend_ky_dy_1.png)

- **Structure:** 10x10 dense, family D3, variant 1.
- **Mechanism path:** S→K; depth 2, 3 total option(s), 1 distractor option(s).
- **Experimental role:** This is a dead-end yellow key-door distractor condition. Its optimal path is a switch-gate stage followed by a key-door stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable while increasing the panel's off-path choice and distractor load.
- **Length caveat:** D3 dead-end fixture; the BFS estimate may be inflated relative to legal runtime execution.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D3/10x10_dense_sg_kr_deadend_ky_dy_1.json)

#### 15. `B1/10x10_dense_swg_0` — 43 actions

![B1/10x10_dense_swg_0 maze render](images/015_B1__10x10_dense_swg_0.png)

- **Structure:** 10x10 dense, family B1, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/10x10_dense_swg_0.json)

#### 16. `B1/10x10_corridor_swg_1` — 44 actions

![B1/10x10_corridor_swg_1 maze render](images/016_B1__10x10_corridor_swg_1.png)

- **Structure:** 10x10 corridor, family B1, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/10x10_corridor_swg_1.json)

#### 17. `M2/10x10_corridor_sg_1` — 44 actions

![M2/10x10_corridor_sg_1 maze render](images/017_M2__10x10_corridor_sg_1.png)

- **Structure:** 10x10 corridor, family M2, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single switch-gate condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M2/10x10_corridor_sg_1.json)

#### 18. `M1/10x10_corridor_kr_1` — 45 actions

![M1/10x10_corridor_kr_1 maze render](images/018_M1__10x10_corridor_kr_1.png)

- **Structure:** 10x10 corridor, family M1, variant 1.
- **Mechanism path:** K; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single key-door condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M1/10x10_corridor_kr_1.json)

#### 19. `D1/10x10_corridor_wrong_ky_kr_kb_0` — 46 actions

![D1/10x10_corridor_wrong_ky_kr_kb_0 maze render](images/019_D1__10x10_corridor_wrong_ky_kr_kb_0.png)

- **Structure:** 10x10 corridor, family D1, variant 0.
- **Mechanism path:** K→K; depth 2, 3 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is two sequential key-door stages.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable while increasing the panel's off-path choice and distractor load.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/10x10_corridor_wrong_ky_kr_kb_0.json)

#### 20. `D2/10x10_corridor_wrong_ky_kr_inactive_sb_sg_kb_0` — 47 actions

![D2/10x10_corridor_wrong_ky_kr_inactive_sb_sg_kb_0 maze render](images/020_D2__10x10_corridor_wrong_ky_kr_inactive_sb_sg_kb_0.png)

- **Structure:** 10x10 corridor, family D2, variant 0.
- **Mechanism path:** K→S→K; depth 3, 5 total option(s), 2 distractor option(s).
- **Experimental role:** This is a wrong-key plus inactive-switch distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable while increasing the panel's off-path choice and distractor load.
- **Five-option stress case:** three required stages plus a wrong key and an inactive switch.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D2/10x10_corridor_wrong_ky_kr_inactive_sb_sg_kb_0.json)

#### 21. `M3/10x10_dense_kr_sg_0` — 53 actions

![M3/10x10_dense_kr_sg_0 maze render](images/021_M3__10x10_dense_kr_sg_0.png)

- **Structure:** 10x10 dense, family M3, variant 0.
- **Mechanism path:** K→S; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->S depth-2 condition. Its optimal path is a key-door stage followed by a switch-gate stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M3/10x10_dense_kr_sg_0.json)

#### 22. `M4/10x10_dense_sg_kr_0` — 53 actions

![M4/10x10_dense_sg_kr_0 maze render](images/022_M4__10x10_dense_sg_kr_0.png)

- **Structure:** 10x10 dense, family M4, variant 0.
- **Mechanism path:** S→K; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled S->K depth-2 condition. Its optimal path is a switch-gate stage followed by a key-door stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M4/10x10_dense_sg_kr_0.json)

#### 23. `D3/10x10_dense_kr_kb_deadend_ky_dy_0` — 54 actions

![D3/10x10_dense_kr_kb_deadend_ky_dy_0 maze render](images/023_D3__10x10_dense_kr_kb_deadend_ky_dy_0.png)

- **Structure:** 10x10 dense, family D3, variant 0.
- **Mechanism path:** K→K; depth 2, 3 total option(s), 1 distractor option(s).
- **Experimental role:** This is a dead-end yellow key-door distractor condition. Its optimal path is two sequential key-door stages.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable while increasing the panel's off-path choice and distractor load.
- **Length caveat:** D3 dead-end fixture; the BFS estimate may be inflated relative to legal runtime execution.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D3/10x10_dense_kr_kb_deadend_ky_dy_0.json)

#### 24. `M5/10x10_dense_kr_kb_0` — 54 actions

![M5/10x10_dense_kr_kb_0 maze render](images/024_M5__10x10_dense_kr_kb_0.png)

- **Structure:** 10x10 dense, family M5, variant 0.
- **Mechanism path:** K→K; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->K depth-2 condition. Its optimal path is two sequential key-door stages.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M5/10x10_dense_kr_kb_0.json)

#### 25. `D3/10x10_dense_kr_sg_kb_deadend_ky_dy_1` — 58 actions

![D3/10x10_dense_kr_sg_kb_deadend_ky_dy_1 maze render](images/025_D3__10x10_dense_kr_sg_kb_deadend_ky_dy_1.png)

- **Structure:** 10x10 dense, family D3, variant 1.
- **Mechanism path:** K→S→K; depth 3, 4 total option(s), 1 distractor option(s).
- **Experimental role:** This is a dead-end yellow key-door distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable while increasing the panel's off-path choice and distractor load.
- **Length caveat:** D3 dead-end fixture; the BFS estimate may be inflated relative to legal runtime execution.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D3/10x10_dense_kr_sg_kb_deadend_ky_dy_1.json)

#### 26. `M6/10x10_dense_kr_sg_kb_1` — 58 actions

![M6/10x10_dense_kr_sg_kb_1 maze render](images/026_M6__10x10_dense_kr_sg_kb_1.png)

- **Structure:** 10x10 dense, family M6, variant 1.
- **Mechanism path:** K→S→K; depth 3, 3 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->S->K depth-3 condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M6/10x10_dense_kr_sg_kb_1.json)

### 60-79 actions

Long mazes bridging the medium regime to the extreme tail.

#### 27. `D2/10x10_dense_wrong_ky_kr_inactive_sb_sg_kb_0` — 61 actions

![D2/10x10_dense_wrong_ky_kr_inactive_sb_sg_kb_0 maze render](images/027_D2__10x10_dense_wrong_ky_kr_inactive_sb_sg_kb_0.png)

- **Structure:** 10x10 dense, family D2, variant 0.
- **Mechanism path:** K→S→K; depth 3, 5 total option(s), 2 distractor option(s).
- **Experimental role:** This is a wrong-key plus inactive-switch distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents long mazes bridging the medium regime to the extreme tail while increasing the panel's off-path choice and distractor load.
- **Five-option stress case:** three required stages plus a wrong key and an inactive switch.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D2/10x10_dense_wrong_ky_kr_inactive_sb_sg_kb_0.json)

#### 28. `B1/14x14_dense_swg_1` — 68 actions

![B1/14x14_dense_swg_1 maze render](images/028_B1__14x14_dense_swg_1.png)

- **Structure:** 14x14 dense, family B1, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents long mazes bridging the medium regime to the extreme tail while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/14x14_dense_swg_1.json)

#### 29. `S5/14x14_corridor_0` — 77 actions

![S5/14x14_corridor_0 maze render](images/029_S5__14x14_corridor_0.png)

- **Structure:** 14x14 corridor, family S5, variant 0.
- **Mechanism path:** none; depth 0, 0 total option(s), 0 distractor option(s).
- **Experimental role:** This is a long 14x14 corridor navigation baseline. Its optimal path is navigation only; no mechanism stage is required on the optimal path.
- **Why selected:** It represents long mazes bridging the medium regime to the extreme tail and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/S5/14x14_corridor_0.json)

#### 30. `B1/14x14_dense_swg_0` — 79 actions

![B1/14x14_dense_swg_0 maze render](images/030_B1__14x14_dense_swg_0.png)

- **Structure:** 14x14 dense, family B1, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents long mazes bridging the medium regime to the extreme tail while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/14x14_dense_swg_0.json)

#### 31. `M2/14x14_dense_sg_0` — 79 actions

![M2/14x14_dense_sg_0 maze render](images/031_M2__14x14_dense_sg_0.png)

- **Structure:** 14x14 dense, family M2, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single switch-gate condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents long mazes bridging the medium regime to the extreme tail and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M2/14x14_dense_sg_0.json)

### 80+ actions

The extreme long-horizon tail dominated by 14x14 structures.

#### 32. `M1/14x14_dense_kr_0` — 80 actions

![M1/14x14_dense_kr_0 maze render](images/032_M1__14x14_dense_kr_0.png)

- **Structure:** 14x14 dense, family M1, variant 0.
- **Mechanism path:** K; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single key-door condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M1/14x14_dense_kr_0.json)

#### 33. `D3/14x14_dense_sg_kr_deadend_ky_dy_0` — 87 actions

![D3/14x14_dense_sg_kr_deadend_ky_dy_0 maze render](images/033_D3__14x14_dense_sg_kr_deadend_ky_dy_0.png)

- **Structure:** 14x14 dense, family D3, variant 0.
- **Mechanism path:** S→K; depth 2, 3 total option(s), 1 distractor option(s).
- **Experimental role:** This is a dead-end yellow key-door distractor condition. Its optimal path is a switch-gate stage followed by a key-door stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures while increasing the panel's off-path choice and distractor load.
- **Length caveat:** D3 dead-end fixture; the BFS estimate may be inflated relative to legal runtime execution.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D3/14x14_dense_sg_kr_deadend_ky_dy_0.json)

#### 34. `M3/14x14_dense_kr_sg_0` — 87 actions

![M3/14x14_dense_kr_sg_0 maze render](images/034_M3__14x14_dense_kr_sg_0.png)

- **Structure:** 14x14 dense, family M3, variant 0.
- **Mechanism path:** K→S; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->S depth-2 condition. Its optimal path is a key-door stage followed by a switch-gate stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M3/14x14_dense_kr_sg_0.json)

#### 35. `M4/14x14_dense_sg_kr_0` — 87 actions

![M4/14x14_dense_sg_kr_0 maze render](images/035_M4__14x14_dense_sg_kr_0.png)

- **Structure:** 14x14 dense, family M4, variant 0.
- **Mechanism path:** S→K; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled S->K depth-2 condition. Its optimal path is a switch-gate stage followed by a key-door stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M4/14x14_dense_sg_kr_0.json)

#### 36. `D2/14x14_corridor_wrong_ky_kr_inactive_sb_sg_kb_0` — 88 actions

![D2/14x14_corridor_wrong_ky_kr_inactive_sb_sg_kb_0 maze render](images/036_D2__14x14_corridor_wrong_ky_kr_inactive_sb_sg_kb_0.png)

- **Structure:** 14x14 corridor, family D2, variant 0.
- **Mechanism path:** K→S→K; depth 3, 5 total option(s), 2 distractor option(s).
- **Experimental role:** This is a wrong-key plus inactive-switch distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures while increasing the panel's off-path choice and distractor load.
- **Five-option stress case:** three required stages plus a wrong key and an inactive switch.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D2/14x14_corridor_wrong_ky_kr_inactive_sb_sg_kb_0.json)

#### 37. `M6/14x14_corridor_kr_sg_kb_0` — 88 actions

![M6/14x14_corridor_kr_sg_kb_0 maze render](images/037_M6__14x14_corridor_kr_sg_kb_0.png)

- **Structure:** 14x14 corridor, family M6, variant 0.
- **Mechanism path:** K→S→K; depth 3, 3 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->S->K depth-3 condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M6/14x14_corridor_kr_sg_kb_0.json)

#### 38. `S5/14x14_corridor_1` — 89 actions

![S5/14x14_corridor_1 maze render](images/038_S5__14x14_corridor_1.png)

- **Structure:** 14x14 corridor, family S5, variant 1.
- **Mechanism path:** none; depth 0, 0 total option(s), 0 distractor option(s).
- **Experimental role:** This is a long 14x14 corridor navigation baseline. Its optimal path is navigation only; no mechanism stage is required on the optimal path.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/S5/14x14_corridor_1.json)

#### 39. `B1/14x14_corridor_swg_1` — 90 actions

![B1/14x14_corridor_swg_1 maze render](images/039_B1__14x14_corridor_swg_1.png)

- **Structure:** 14x14 corridor, family B1, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/14x14_corridor_swg_1.json)

#### 40. `M2/14x14_corridor_sg_1` — 90 actions

![M2/14x14_corridor_sg_1 maze render](images/040_M2__14x14_corridor_sg_1.png)

- **Structure:** 14x14 corridor, family M2, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single switch-gate condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M2/14x14_corridor_sg_1.json)

#### 41. `M1/14x14_corridor_kr_1` — 91 actions

![M1/14x14_corridor_kr_1 maze render](images/041_M1__14x14_corridor_kr_1.png)

- **Structure:** 14x14 corridor, family M1, variant 1.
- **Mechanism path:** K; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single key-door condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M1/14x14_corridor_kr_1.json)

#### 42. `D1/14x14_corridor_wrong_ky_kr_sg_1` — 92 actions

![D1/14x14_corridor_wrong_ky_kr_sg_1 maze render](images/042_D1__14x14_corridor_wrong_ky_kr_sg_1.png)

- **Structure:** 14x14 corridor, family D1, variant 1.
- **Mechanism path:** K→S; depth 2, 3 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is a key-door stage followed by a switch-gate stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures while increasing the panel's off-path choice and distractor load.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/14x14_corridor_wrong_ky_kr_sg_1.json)

#### 43. `D2/14x14_corridor_wrong_ky_inactive_sb_sg_kr_1` — 92 actions

![D2/14x14_corridor_wrong_ky_inactive_sb_sg_kr_1 maze render](images/043_D2__14x14_corridor_wrong_ky_inactive_sb_sg_kr_1.png)

- **Structure:** 14x14 corridor, family D2, variant 1.
- **Mechanism path:** S→K; depth 2, 4 total option(s), 2 distractor option(s).
- **Experimental role:** This is a wrong-key plus inactive-switch distractor condition. Its optimal path is a switch-gate stage followed by a key-door stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures while increasing the panel's off-path choice and distractor load.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D2/14x14_corridor_wrong_ky_inactive_sb_sg_kr_1.json)

#### 44. `M3/14x14_corridor_kr_sg_1` — 92 actions

![M3/14x14_corridor_kr_sg_1 maze render](images/044_M3__14x14_corridor_kr_sg_1.png)

- **Structure:** 14x14 corridor, family M3, variant 1.
- **Mechanism path:** K→S; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->S depth-2 condition. Its optimal path is a key-door stage followed by a switch-gate stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M3/14x14_corridor_kr_sg_1.json)

#### 45. `M4/14x14_corridor_sg_kr_1` — 92 actions

![M4/14x14_corridor_sg_kr_1 maze render](images/045_M4__14x14_corridor_sg_kr_1.png)

- **Structure:** 14x14 corridor, family M4, variant 1.
- **Mechanism path:** S→K; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled S->K depth-2 condition. Its optimal path is a switch-gate stage followed by a key-door stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M4/14x14_corridor_sg_kr_1.json)

#### 46. `M5/14x14_corridor_kr_kb_1` — 93 actions

![M5/14x14_corridor_kr_kb_1 maze render](images/046_M5__14x14_corridor_kr_kb_1.png)

- **Structure:** 14x14 corridor, family M5, variant 1.
- **Mechanism path:** K→K; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->K depth-2 condition. Its optimal path is two sequential key-door stages.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M5/14x14_corridor_kr_kb_1.json)

#### 47. `D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_0` — 95 actions

![D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_0 maze render](images/047_D2__14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_0.png)

- **Structure:** 14x14 dense, family D2, variant 0.
- **Mechanism path:** K→S→K; depth 3, 5 total option(s), 2 distractor option(s).
- **Experimental role:** This is a wrong-key plus inactive-switch distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures while increasing the panel's off-path choice and distractor load.
- **Five-option stress case:** three required stages plus a wrong key and an inactive switch.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_0.json)

#### 48. `D1/14x14_dense_wrong_ky_kr_sg_1` — 96 actions

![D1/14x14_dense_wrong_ky_kr_sg_1 maze render](images/048_D1__14x14_dense_wrong_ky_kr_sg_1.png)

- **Structure:** 14x14 dense, family D1, variant 1.
- **Mechanism path:** K→S; depth 2, 3 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is a key-door stage followed by a switch-gate stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures while increasing the panel's off-path choice and distractor load.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/14x14_dense_wrong_ky_kr_sg_1.json)

#### 49. `M5/14x14_dense_kr_kb_1` — 97 actions

![M5/14x14_dense_kr_kb_1 maze render](images/049_M5__14x14_dense_kr_kb_1.png)

- **Structure:** 14x14 dense, family M5, variant 1.
- **Mechanism path:** K→K; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->K depth-2 condition. Its optimal path is two sequential key-door stages.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M5/14x14_dense_kr_kb_1.json)

#### 50. `M6/14x14_dense_kr_sg_kb_1` — 106 actions

![M6/14x14_dense_kr_sg_kb_1 maze render](images/050_M6__14x14_dense_kr_sg_kb_1.png)

- **Structure:** 14x14 dense, family M6, variant 1.
- **Mechanism path:** K→S→K; depth 3, 3 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->S->K depth-3 condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M6/14x14_dense_kr_sg_kb_1.json)
