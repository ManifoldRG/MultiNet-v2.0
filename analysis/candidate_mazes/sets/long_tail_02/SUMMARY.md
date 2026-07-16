# Long Tail 02: illustrated 50-maze candidate set

The long-tail candidate treats path length as the primary stress variable. It retains the controlled key-versus-switch comparisons of the balanced design, but moves many discretionary slots into 14x14 corridor and dense layouts above 80 actions. This makes it well suited to measuring planning degradation, context or memory pressure, compounding navigation errors, and the interaction between mechanisms and very long execution horizons.

This is the second retained alternative, chosen to differ by at least four mazes from variant 01. Its realized allocation contains 22 mazes at 80+ actions, 12 mazes with at least one distractor option, and 9 depth-3 mazes. The exact choices below should therefore be treated as a designed experimental panel, not merely a random sample from the corpus.

![Candidate optimal-length curve](candidate_curve.png)

## Headline composition

| measure | value |
|---|---:|
| mazes | 50 |
| optimal-action range | 23–106 |
| median optimal actions | 73.0 |
| largest adjacent length gap | 11 |
| B1 mazes | 8 |
| depth-2 mazes | 13 |
| depth-3 mazes | 9 |
| distractor mazes | 12 |
| five-option mazes | 3 |
| 80+ action mazes | 22 |
| D3 length-risk mazes | 2 |

**Length clusters:** <10: 0, 10-15: 0, 16-19: 0, 20-25: 2, 26-39: 8, 40-59: 12, 60-79: 6, 80+: 22.

**Path signatures:** none: 4, K: 8, S: 16, K→K: 3, K→S: 5, S→K: 5, K→S→K: 9.

**Families:** B1: 8, D1: 7, D2: 3, D3: 2, M1: 8, M2: 8, M3: 3, M4: 4, M5: 2, M6: 2, S2: 1, S5: 2.

This exploratory variant deliberately omits the 9/11-action S1 empty rooms. Their depth-0 control role is replaced by the 77/89-action S5 corridor pair, shifting two observations into long-horizon navigation without introducing a mechanism confound. Mechanism comparisons still begin in the 26–39 band. The candidate curve above shows mechanism signature by color, distractor-bearing mazes with X markers, and B1 fixtures with a `B` label.

## Controlled comparisons

### Single-stage key versus switch

M1/M2 pairs below share size, topology, and filename variant. Their one-action length differences largely reflect the executable pickup/toggle semantics rather than different spatial geometry.

| matched cell | M1 key maze | steps | M2 switch maze | steps |
|---|---|---:|---|---:|
| `8x8:corridor:0` | `M1/8x8_corridor_kr_0` | 27 | `M2/8x8_corridor_sg_0` | 26 |
| `10x10:dense:1` | `M1/10x10_dense_kr_1` | 37 | `M2/10x10_dense_sg_1` | 36 |
| `10x10:corridor:0` | `M1/10x10_corridor_kr_0` | 44 | `M2/10x10_corridor_sg_0` | 43 |
| `10x10:dense:0` | `M1/10x10_dense_kr_0` | 44 | `M2/10x10_dense_sg_0` | 43 |
| `14x14:dense:1` | `M1/14x14_dense_kr_1` | 69 | `M2/14x14_dense_sg_1` | 68 |
| `14x14:dense:0` | `M1/14x14_dense_kr_0` | 80 | `M2/14x14_dense_sg_0` | 79 |
| `14x14:corridor:0` | `M1/14x14_corridor_kr_0` | 85 | `M2/14x14_corridor_sg_0` | 84 |
| `14x14:corridor:1` | `M1/14x14_corridor_kr_1` | 91 | `M2/14x14_corridor_sg_1` | 90 |

### Depth-2 ordering triplets

M3/M4/M5 triplets hold geometry fixed while changing the required ordering among K→S, S→K, and K→K. These are the cleanest fixtures for separating mechanism type from dependency depth.

| matched cell | M3 K→S | steps | M4 S→K | steps | M5 K→K | steps |
|---|---|---:|---|---:|---|---:|
| `10x10:corridor:1` | `M3/10x10_corridor_kr_sg_1` | 46 | `M4/10x10_corridor_sg_kr_1` | 46 | `M5/10x10_corridor_kr_kb_1` | 47 |
| `14x14:dense:1` | `M3/14x14_dense_kr_sg_1` | 96 | `M4/14x14_dense_sg_kr_1` | 96 | `M5/14x14_dense_kr_kb_1` | 97 |

## Distractors, five-option cases, and B1 structure

This panel contains 12 distractor mazes. D1 adds a wrong key, D2 adds both a wrong key and an inactive switch, and D3 places a yellow key-door option in a dead-end branch. Distractor counts are kept separate from path depth: a D2/M6 maze is depth 3 but has five total options.

### Five-option mazes

| rank | steps | maze | composition |
|---:|---:|---|---|
| 6 | 30 | `D2/8x8_corridor_wrong_ky_kr_inactive_sb_sg_kb_0` | K→S→K required + wrong key + inactive switch |
| 22 | 58 | `D2/10x10_dense_wrong_ky_kr_inactive_sb_sg_kb_1` | K→S→K required + wrong key + inactive switch |
| 49 | 106 | `D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_1` | K→S→K required + wrong key + inactive switch |

### B1 selection

B1 is treated as a protected structural stratum rather than interchangeable with M2. The chosen B1 mazes are:

| rank | steps | size/topology | maze |
|---:|---:|---|---|
| 4 | 27 | 8x8 corridor | `B1/8x8_corridor_swg_1` |
| 8 | 36 | 10x10 dense | `B1/10x10_dense_swg_1` |
| 11 | 43 | 10x10 corridor | `B1/10x10_corridor_swg_0` |
| 12 | 43 | 10x10 dense | `B1/10x10_dense_swg_0` |
| 15 | 44 | 10x10 corridor | `B1/10x10_corridor_swg_1` |
| 23 | 68 | 14x14 dense | `B1/14x14_dense_swg_1` |
| 27 | 79 | 14x14 dense | `B1/14x14_dense_swg_0` |
| 30 | 84 | 14x14 corridor | `B1/14x14_corridor_swg_0` |

## Long-tail allocation

The panel contains 22 mazes at 80+ estimated optimal actions. These fixtures test whether errors grow with execution horizon, whether mechanism costs compound with navigation length, and whether corridor versus dense topology changes the model's ability to preserve a plan.

| rank | steps | depth | options | signature | topology | maze |
|---:|---:|---:|---:|---|---|---|
| 29 | 80 | 1 | 1 | K | dense | `M1/14x14_dense_kr_0` |
| 30 | 84 | 1 | 1 | S | corridor | `B1/14x14_corridor_swg_0` |
| 31 | 84 | 1 | 1 | S | corridor | `M2/14x14_corridor_sg_0` |
| 32 | 85 | 1 | 1 | K | corridor | `M1/14x14_corridor_kr_0` |
| 33 | 86 | 2 | 3 | S→K | corridor | `D1/14x14_corridor_wrong_ky_sg_kr_0` |
| 34 | 86 | 2 | 2 | K→S | corridor | `M3/14x14_corridor_kr_sg_0` |
| 35 | 87 | 2 | 3 | K→S | dense | `D1/14x14_dense_wrong_ky_kr_sg_0` |
| 36 | 87 | 2 | 2 | S→K | dense | `M4/14x14_dense_sg_kr_0` |
| 37 | 88 | 2 | 3 | K→K | dense | `D1/14x14_dense_wrong_ky_kr_kb_0` |
| 38 | 89 | 0 | 0 | none | corridor | `S5/14x14_corridor_1` |
| 39 | 90 | 1 | 1 | S | corridor | `M2/14x14_corridor_sg_1` |
| 40 | 91 | 1 | 1 | K | corridor | `M1/14x14_corridor_kr_1` |
| 41 | 92 | 2 | 3 | K→S | corridor | `D1/14x14_corridor_wrong_ky_kr_sg_1` |
| 42 | 92 | 2 | 2 | S→K | corridor | `M4/14x14_corridor_sg_kr_1` |
| 43 | 94 | 3 | 4 | K→S→K | corridor | `D1/14x14_corridor_wrong_ky_kr_sg_kb_1` |
| 44 | 95 | 3 | 4 | K→S→K | dense | `D3/14x14_dense_kr_sg_kb_deadend_ky_dy_0` |
| 45 | 96 | 2 | 2 | K→S | dense | `M3/14x14_dense_kr_sg_1` |
| 46 | 96 | 2 | 2 | S→K | dense | `M4/14x14_dense_sg_kr_1` |
| 47 | 97 | 2 | 2 | K→K | dense | `M5/14x14_dense_kr_kb_1` |
| 48 | 106 | 3 | 4 | K→S→K | dense | `D1/14x14_dense_wrong_ky_kr_sg_kb_1` |
| 49 | 106 | 3 | 5 | K→S→K | dense | `D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_1` |
| 50 | 106 | 3 | 3 | K→S→K | dense | `M6/14x14_dense_kr_sg_kb_1` |

## Interpretation risks and gaps

1. The corpus does not contain replicated mechanism mazes at the requested earliest lengths. Only the 9/11-action S1 anchors occupy the first cluster, there are no 16–19-action mazes, and the 20–25-action fixtures require no on-path mechanism. Controlled mechanism comparisons begin at 26 actions.
2. This panel contains 2 D3 maze(s). Prior analysis documents at least one D3 case where the BFS estimate is 31 actions but a legal 23-step model solve exists. D3 values should be treated as estimated lengths pending planner/runtime reconciliation.
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
- **Why selected:** It represents the second early cluster, still before mechanism comparisons begin and helps maintain the intended smooth difficulty curve.
- **Length caveat:** D3 dead-end fixture; the BFS estimate may be inflated relative to legal runtime execution.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D3/10x10_dense_deadend_ky_dy_1.json)

#### 02. `S2/8x8_corridor_0` — 25 actions

![S2/8x8_corridor_0 maze render](images/002_S2__8x8_corridor_0.png)

- **Structure:** 8x8 corridor, family S2, variant 0.
- **Mechanism path:** none; depth 0, 0 total option(s), 0 distractor option(s).
- **Experimental role:** This is a short winding-corridor navigation baseline. Its optimal path is navigation only; no mechanism stage is required on the optimal path.
- **Why selected:** It represents the second early cluster, still before mechanism comparisons begin and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/S2/8x8_corridor_0.json)

### 26-39 actions

Short mechanism-capable mazes.

#### 03. `M2/8x8_corridor_sg_0` — 26 actions

![M2/8x8_corridor_sg_0 maze render](images/003_M2__8x8_corridor_sg_0.png)

- **Structure:** 8x8 corridor, family M2, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single switch-gate condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents short mechanism-capable mazes and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M2/8x8_corridor_sg_0.json)

#### 04. `B1/8x8_corridor_swg_1` — 27 actions

![B1/8x8_corridor_swg_1 maze render](images/004_B1__8x8_corridor_swg_1.png)

- **Structure:** 8x8 corridor, family B1, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents short mechanism-capable mazes while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/8x8_corridor_swg_1.json)

#### 05. `M1/8x8_corridor_kr_0` — 27 actions

![M1/8x8_corridor_kr_0 maze render](images/005_M1__8x8_corridor_kr_0.png)

- **Structure:** 8x8 corridor, family M1, variant 0.
- **Mechanism path:** K; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single key-door condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents short mechanism-capable mazes and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M1/8x8_corridor_kr_0.json)

#### 06. `D2/8x8_corridor_wrong_ky_kr_inactive_sb_sg_kb_0` — 30 actions

![D2/8x8_corridor_wrong_ky_kr_inactive_sb_sg_kb_0 maze render](images/006_D2__8x8_corridor_wrong_ky_kr_inactive_sb_sg_kb_0.png)

- **Structure:** 8x8 corridor, family D2, variant 0.
- **Mechanism path:** K→S→K; depth 3, 5 total option(s), 2 distractor option(s).
- **Experimental role:** This is a wrong-key plus inactive-switch distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents short mechanism-capable mazes and helps maintain the intended smooth difficulty curve.
- **Five-option stress case:** three required stages plus a wrong key and an inactive switch.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D2/8x8_corridor_wrong_ky_kr_inactive_sb_sg_kb_0.json)

#### 07. `M6/8x8_corridor_kr_sg_kb_1` — 31 actions

![M6/8x8_corridor_kr_sg_kb_1 maze render](images/007_M6__8x8_corridor_kr_sg_kb_1.png)

- **Structure:** 8x8 corridor, family M6, variant 1.
- **Mechanism path:** K→S→K; depth 3, 3 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->S->K depth-3 condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents short mechanism-capable mazes and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M6/8x8_corridor_kr_sg_kb_1.json)

#### 08. `B1/10x10_dense_swg_1` — 36 actions

![B1/10x10_dense_swg_1 maze render](images/008_B1__10x10_dense_swg_1.png)

- **Structure:** 10x10 dense, family B1, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents short mechanism-capable mazes while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/10x10_dense_swg_1.json)

#### 09. `M2/10x10_dense_sg_1` — 36 actions

![M2/10x10_dense_sg_1 maze render](images/009_M2__10x10_dense_sg_1.png)

- **Structure:** 10x10 dense, family M2, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single switch-gate condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents short mechanism-capable mazes and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M2/10x10_dense_sg_1.json)

#### 10. `M1/10x10_dense_kr_1` — 37 actions

![M1/10x10_dense_kr_1 maze render](images/010_M1__10x10_dense_kr_1.png)

- **Structure:** 10x10 dense, family M1, variant 1.
- **Mechanism path:** K; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single key-door condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents short mechanism-capable mazes and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M1/10x10_dense_kr_1.json)

### 40-59 actions

Medium-length mazes where ordering and distractor effects become separable.

#### 11. `B1/10x10_corridor_swg_0` — 43 actions

![B1/10x10_corridor_swg_0 maze render](images/011_B1__10x10_corridor_swg_0.png)

- **Structure:** 10x10 corridor, family B1, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/10x10_corridor_swg_0.json)

#### 12. `B1/10x10_dense_swg_0` — 43 actions

![B1/10x10_dense_swg_0 maze render](images/012_B1__10x10_dense_swg_0.png)

- **Structure:** 10x10 dense, family B1, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/10x10_dense_swg_0.json)

#### 13. `M2/10x10_corridor_sg_0` — 43 actions

![M2/10x10_corridor_sg_0 maze render](images/013_M2__10x10_corridor_sg_0.png)

- **Structure:** 10x10 corridor, family M2, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single switch-gate condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M2/10x10_corridor_sg_0.json)

#### 14. `M2/10x10_dense_sg_0` — 43 actions

![M2/10x10_dense_sg_0 maze render](images/014_M2__10x10_dense_sg_0.png)

- **Structure:** 10x10 dense, family M2, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single switch-gate condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M2/10x10_dense_sg_0.json)

#### 15. `B1/10x10_corridor_swg_1` — 44 actions

![B1/10x10_corridor_swg_1 maze render](images/015_B1__10x10_corridor_swg_1.png)

- **Structure:** 10x10 corridor, family B1, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/10x10_corridor_swg_1.json)

#### 16. `M1/10x10_corridor_kr_0` — 44 actions

![M1/10x10_corridor_kr_0 maze render](images/016_M1__10x10_corridor_kr_0.png)

- **Structure:** 10x10 corridor, family M1, variant 0.
- **Mechanism path:** K; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single key-door condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M1/10x10_corridor_kr_0.json)

#### 17. `M1/10x10_dense_kr_0` — 44 actions

![M1/10x10_dense_kr_0 maze render](images/017_M1__10x10_dense_kr_0.png)

- **Structure:** 10x10 dense, family M1, variant 0.
- **Mechanism path:** K; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single key-door condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M1/10x10_dense_kr_0.json)

#### 18. `M3/10x10_corridor_kr_sg_1` — 46 actions

![M3/10x10_corridor_kr_sg_1 maze render](images/018_M3__10x10_corridor_kr_sg_1.png)

- **Structure:** 10x10 corridor, family M3, variant 1.
- **Mechanism path:** K→S; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->S depth-2 condition. Its optimal path is a key-door stage followed by a switch-gate stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M3/10x10_corridor_kr_sg_1.json)

#### 19. `M4/10x10_corridor_sg_kr_1` — 46 actions

![M4/10x10_corridor_sg_kr_1 maze render](images/019_M4__10x10_corridor_sg_kr_1.png)

- **Structure:** 10x10 corridor, family M4, variant 1.
- **Mechanism path:** S→K; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled S->K depth-2 condition. Its optimal path is a switch-gate stage followed by a key-door stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M4/10x10_corridor_sg_kr_1.json)

#### 20. `M5/10x10_corridor_kr_kb_1` — 47 actions

![M5/10x10_corridor_kr_kb_1 maze render](images/020_M5__10x10_corridor_kr_kb_1.png)

- **Structure:** 10x10 corridor, family M5, variant 1.
- **Mechanism path:** K→K; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->K depth-2 condition. Its optimal path is two sequential key-door stages.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M5/10x10_corridor_kr_kb_1.json)

#### 21. `D1/10x10_dense_wrong_ky_kr_sg_kb_1` — 58 actions

![D1/10x10_dense_wrong_ky_kr_sg_kb_1 maze render](images/021_D1__10x10_dense_wrong_ky_kr_sg_kb_1.png)

- **Structure:** 10x10 dense, family D1, variant 1.
- **Mechanism path:** K→S→K; depth 3, 4 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/10x10_dense_wrong_ky_kr_sg_kb_1.json)

#### 22. `D2/10x10_dense_wrong_ky_kr_inactive_sb_sg_kb_1` — 58 actions

![D2/10x10_dense_wrong_ky_kr_inactive_sb_sg_kb_1 maze render](images/022_D2__10x10_dense_wrong_ky_kr_inactive_sb_sg_kb_1.png)

- **Structure:** 10x10 dense, family D2, variant 1.
- **Mechanism path:** K→S→K; depth 3, 5 total option(s), 2 distractor option(s).
- **Experimental role:** This is a wrong-key plus inactive-switch distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Five-option stress case:** three required stages plus a wrong key and an inactive switch.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D2/10x10_dense_wrong_ky_kr_inactive_sb_sg_kb_1.json)

### 60-79 actions

Long mazes bridging the medium regime to the extreme tail.

#### 23. `B1/14x14_dense_swg_1` — 68 actions

![B1/14x14_dense_swg_1 maze render](images/023_B1__14x14_dense_swg_1.png)

- **Structure:** 14x14 dense, family B1, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents long mazes bridging the medium regime to the extreme tail while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/14x14_dense_swg_1.json)

#### 24. `M2/14x14_dense_sg_1` — 68 actions

![M2/14x14_dense_sg_1 maze render](images/024_M2__14x14_dense_sg_1.png)

- **Structure:** 14x14 dense, family M2, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single switch-gate condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents long mazes bridging the medium regime to the extreme tail and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M2/14x14_dense_sg_1.json)

#### 25. `M1/14x14_dense_kr_1` — 69 actions

![M1/14x14_dense_kr_1 maze render](images/025_M1__14x14_dense_kr_1.png)

- **Structure:** 14x14 dense, family M1, variant 1.
- **Mechanism path:** K; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single key-door condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents long mazes bridging the medium regime to the extreme tail and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M1/14x14_dense_kr_1.json)

#### 26. `S5/14x14_corridor_0` — 77 actions

![S5/14x14_corridor_0 maze render](images/026_S5__14x14_corridor_0.png)

- **Structure:** 14x14 corridor, family S5, variant 0.
- **Mechanism path:** none; depth 0, 0 total option(s), 0 distractor option(s).
- **Experimental role:** This is a long 14x14 corridor navigation baseline. Its optimal path is navigation only; no mechanism stage is required on the optimal path.
- **Why selected:** It represents long mazes bridging the medium regime to the extreme tail and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/S5/14x14_corridor_0.json)

#### 27. `B1/14x14_dense_swg_0` — 79 actions

![B1/14x14_dense_swg_0 maze render](images/027_B1__14x14_dense_swg_0.png)

- **Structure:** 14x14 dense, family B1, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents long mazes bridging the medium regime to the extreme tail while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/14x14_dense_swg_0.json)

#### 28. `M2/14x14_dense_sg_0` — 79 actions

![M2/14x14_dense_sg_0 maze render](images/028_M2__14x14_dense_sg_0.png)

- **Structure:** 14x14 dense, family M2, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single switch-gate condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents long mazes bridging the medium regime to the extreme tail and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M2/14x14_dense_sg_0.json)

### 80+ actions

The extreme long-horizon tail dominated by 14x14 structures.

#### 29. `M1/14x14_dense_kr_0` — 80 actions

![M1/14x14_dense_kr_0 maze render](images/029_M1__14x14_dense_kr_0.png)

- **Structure:** 14x14 dense, family M1, variant 0.
- **Mechanism path:** K; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single key-door condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and directly contributes to the panel's high-horizon emphasis.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M1/14x14_dense_kr_0.json)

#### 30. `B1/14x14_corridor_swg_0` — 84 actions

![B1/14x14_corridor_swg_0 maze render](images/030_B1__14x14_corridor_swg_0.png)

- **Structure:** 14x14 corridor, family B1, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and directly contributes to the panel's high-horizon emphasis.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/14x14_corridor_swg_0.json)

#### 31. `M2/14x14_corridor_sg_0` — 84 actions

![M2/14x14_corridor_sg_0 maze render](images/031_M2__14x14_corridor_sg_0.png)

- **Structure:** 14x14 corridor, family M2, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single switch-gate condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and directly contributes to the panel's high-horizon emphasis.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M2/14x14_corridor_sg_0.json)

#### 32. `M1/14x14_corridor_kr_0` — 85 actions

![M1/14x14_corridor_kr_0 maze render](images/032_M1__14x14_corridor_kr_0.png)

- **Structure:** 14x14 corridor, family M1, variant 0.
- **Mechanism path:** K; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single key-door condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and directly contributes to the panel's high-horizon emphasis.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M1/14x14_corridor_kr_0.json)

#### 33. `D1/14x14_corridor_wrong_ky_sg_kr_0` — 86 actions

![D1/14x14_corridor_wrong_ky_sg_kr_0 maze render](images/033_D1__14x14_corridor_wrong_ky_sg_kr_0.png)

- **Structure:** 14x14 corridor, family D1, variant 0.
- **Mechanism path:** S→K; depth 2, 3 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is a switch-gate stage followed by a key-door stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and directly contributes to the panel's high-horizon emphasis.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/14x14_corridor_wrong_ky_sg_kr_0.json)

#### 34. `M3/14x14_corridor_kr_sg_0` — 86 actions

![M3/14x14_corridor_kr_sg_0 maze render](images/034_M3__14x14_corridor_kr_sg_0.png)

- **Structure:** 14x14 corridor, family M3, variant 0.
- **Mechanism path:** K→S; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->S depth-2 condition. Its optimal path is a key-door stage followed by a switch-gate stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and directly contributes to the panel's high-horizon emphasis.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M3/14x14_corridor_kr_sg_0.json)

#### 35. `D1/14x14_dense_wrong_ky_kr_sg_0` — 87 actions

![D1/14x14_dense_wrong_ky_kr_sg_0 maze render](images/035_D1__14x14_dense_wrong_ky_kr_sg_0.png)

- **Structure:** 14x14 dense, family D1, variant 0.
- **Mechanism path:** K→S; depth 2, 3 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is a key-door stage followed by a switch-gate stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and directly contributes to the panel's high-horizon emphasis.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/14x14_dense_wrong_ky_kr_sg_0.json)

#### 36. `M4/14x14_dense_sg_kr_0` — 87 actions

![M4/14x14_dense_sg_kr_0 maze render](images/036_M4__14x14_dense_sg_kr_0.png)

- **Structure:** 14x14 dense, family M4, variant 0.
- **Mechanism path:** S→K; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled S->K depth-2 condition. Its optimal path is a switch-gate stage followed by a key-door stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and directly contributes to the panel's high-horizon emphasis.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M4/14x14_dense_sg_kr_0.json)

#### 37. `D1/14x14_dense_wrong_ky_kr_kb_0` — 88 actions

![D1/14x14_dense_wrong_ky_kr_kb_0 maze render](images/037_D1__14x14_dense_wrong_ky_kr_kb_0.png)

- **Structure:** 14x14 dense, family D1, variant 0.
- **Mechanism path:** K→K; depth 2, 3 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is two sequential key-door stages.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and directly contributes to the panel's high-horizon emphasis.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/14x14_dense_wrong_ky_kr_kb_0.json)

#### 38. `S5/14x14_corridor_1` — 89 actions

![S5/14x14_corridor_1 maze render](images/038_S5__14x14_corridor_1.png)

- **Structure:** 14x14 corridor, family S5, variant 1.
- **Mechanism path:** none; depth 0, 0 total option(s), 0 distractor option(s).
- **Experimental role:** This is a long 14x14 corridor navigation baseline. Its optimal path is navigation only; no mechanism stage is required on the optimal path.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and directly contributes to the panel's high-horizon emphasis.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/S5/14x14_corridor_1.json)

#### 39. `M2/14x14_corridor_sg_1` — 90 actions

![M2/14x14_corridor_sg_1 maze render](images/039_M2__14x14_corridor_sg_1.png)

- **Structure:** 14x14 corridor, family M2, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single switch-gate condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and directly contributes to the panel's high-horizon emphasis.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M2/14x14_corridor_sg_1.json)

#### 40. `M1/14x14_corridor_kr_1` — 91 actions

![M1/14x14_corridor_kr_1 maze render](images/040_M1__14x14_corridor_kr_1.png)

- **Structure:** 14x14 corridor, family M1, variant 1.
- **Mechanism path:** K; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single key-door condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and directly contributes to the panel's high-horizon emphasis.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M1/14x14_corridor_kr_1.json)

#### 41. `D1/14x14_corridor_wrong_ky_kr_sg_1` — 92 actions

![D1/14x14_corridor_wrong_ky_kr_sg_1 maze render](images/041_D1__14x14_corridor_wrong_ky_kr_sg_1.png)

- **Structure:** 14x14 corridor, family D1, variant 1.
- **Mechanism path:** K→S; depth 2, 3 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is a key-door stage followed by a switch-gate stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and directly contributes to the panel's high-horizon emphasis.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/14x14_corridor_wrong_ky_kr_sg_1.json)

#### 42. `M4/14x14_corridor_sg_kr_1` — 92 actions

![M4/14x14_corridor_sg_kr_1 maze render](images/042_M4__14x14_corridor_sg_kr_1.png)

- **Structure:** 14x14 corridor, family M4, variant 1.
- **Mechanism path:** S→K; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled S->K depth-2 condition. Its optimal path is a switch-gate stage followed by a key-door stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and directly contributes to the panel's high-horizon emphasis.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M4/14x14_corridor_sg_kr_1.json)

#### 43. `D1/14x14_corridor_wrong_ky_kr_sg_kb_1` — 94 actions

![D1/14x14_corridor_wrong_ky_kr_sg_kb_1 maze render](images/043_D1__14x14_corridor_wrong_ky_kr_sg_kb_1.png)

- **Structure:** 14x14 corridor, family D1, variant 1.
- **Mechanism path:** K→S→K; depth 3, 4 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and directly contributes to the panel's high-horizon emphasis.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/14x14_corridor_wrong_ky_kr_sg_kb_1.json)

#### 44. `D3/14x14_dense_kr_sg_kb_deadend_ky_dy_0` — 95 actions

![D3/14x14_dense_kr_sg_kb_deadend_ky_dy_0 maze render](images/044_D3__14x14_dense_kr_sg_kb_deadend_ky_dy_0.png)

- **Structure:** 14x14 dense, family D3, variant 0.
- **Mechanism path:** K→S→K; depth 3, 4 total option(s), 1 distractor option(s).
- **Experimental role:** This is a dead-end yellow key-door distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and directly contributes to the panel's high-horizon emphasis.
- **Length caveat:** D3 dead-end fixture; the BFS estimate may be inflated relative to legal runtime execution.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D3/14x14_dense_kr_sg_kb_deadend_ky_dy_0.json)

#### 45. `M3/14x14_dense_kr_sg_1` — 96 actions

![M3/14x14_dense_kr_sg_1 maze render](images/045_M3__14x14_dense_kr_sg_1.png)

- **Structure:** 14x14 dense, family M3, variant 1.
- **Mechanism path:** K→S; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->S depth-2 condition. Its optimal path is a key-door stage followed by a switch-gate stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and directly contributes to the panel's high-horizon emphasis.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M3/14x14_dense_kr_sg_1.json)

#### 46. `M4/14x14_dense_sg_kr_1` — 96 actions

![M4/14x14_dense_sg_kr_1 maze render](images/046_M4__14x14_dense_sg_kr_1.png)

- **Structure:** 14x14 dense, family M4, variant 1.
- **Mechanism path:** S→K; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled S->K depth-2 condition. Its optimal path is a switch-gate stage followed by a key-door stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and directly contributes to the panel's high-horizon emphasis.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M4/14x14_dense_sg_kr_1.json)

#### 47. `M5/14x14_dense_kr_kb_1` — 97 actions

![M5/14x14_dense_kr_kb_1 maze render](images/047_M5__14x14_dense_kr_kb_1.png)

- **Structure:** 14x14 dense, family M5, variant 1.
- **Mechanism path:** K→K; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->K depth-2 condition. Its optimal path is two sequential key-door stages.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and directly contributes to the panel's high-horizon emphasis.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M5/14x14_dense_kr_kb_1.json)

#### 48. `D1/14x14_dense_wrong_ky_kr_sg_kb_1` — 106 actions

![D1/14x14_dense_wrong_ky_kr_sg_kb_1 maze render](images/048_D1__14x14_dense_wrong_ky_kr_sg_kb_1.png)

- **Structure:** 14x14 dense, family D1, variant 1.
- **Mechanism path:** K→S→K; depth 3, 4 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and directly contributes to the panel's high-horizon emphasis.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/14x14_dense_wrong_ky_kr_sg_kb_1.json)

#### 49. `D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_1` — 106 actions

![D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_1 maze render](images/049_D2__14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_1.png)

- **Structure:** 14x14 dense, family D2, variant 1.
- **Mechanism path:** K→S→K; depth 3, 5 total option(s), 2 distractor option(s).
- **Experimental role:** This is a wrong-key plus inactive-switch distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and directly contributes to the panel's high-horizon emphasis.
- **Five-option stress case:** three required stages plus a wrong key and an inactive switch.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_1.json)

#### 50. `M6/14x14_dense_kr_sg_kb_1` — 106 actions

![M6/14x14_dense_kr_sg_kb_1 maze render](images/050_M6__14x14_dense_kr_sg_kb_1.png)

- **Structure:** 14x14 dense, family M6, variant 1.
- **Mechanism path:** K→S→K; depth 3, 3 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->S->K depth-3 condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and directly contributes to the panel's high-horizon emphasis.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M6/14x14_dense_kr_sg_kb_1.json)
