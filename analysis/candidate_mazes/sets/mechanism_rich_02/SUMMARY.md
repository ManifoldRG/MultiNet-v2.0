# Mechanism Rich 02: illustrated 50-maze candidate set

The mechanism-rich candidate focuses on complexity that cannot be explained by length alone. It deliberately increases distractor-bearing D1/D2/D3 fixtures, depth-2 ordering problems, depth-3 K->S->K chains, and the five-option cases that combine three required stages with a wrong key and an inactive switch. It keeps one pure K/S control pair in each feasible length region, but uses the released slots for mechanism and distractor stress rather than maximum replication.

This is the second retained alternative, chosen to differ by at least four mazes from variant 01. Its realized allocation contains 18 mazes at 80+ actions, 19 mazes with at least one distractor option, and 11 depth-3 mazes. The exact choices below should therefore be treated as a designed experimental panel, not merely a random sample from the corpus.

![Candidate optimal-length curve](candidate_curve.png)

## Headline composition

| measure | value |
|---|---:|
| mazes | 50 |
| optimal-action range | 9–106 |
| median optimal actions | 53.5 |
| largest adjacent length gap | 15 |
| B1 mazes | 8 |
| depth-2 mazes | 19 |
| depth-3 mazes | 11 |
| distractor mazes | 19 |
| five-option mazes | 5 |
| 80+ action mazes | 18 |
| D3 length-risk mazes | 3 |

**Length clusters:** <10: 1, 10-15: 1, 16-19: 0, 20-25: 1, 26-39: 11, 40-59: 13, 60-79: 5, 80+: 18.

**Path signatures:** none: 3, K: 5, S: 12, K→K: 6, K→S: 6, S→K: 7, K→S→K: 11.

**Families:** B1: 8, D1: 7, D2: 9, D3: 3, M1: 4, M2: 4, M3: 3, M4: 3, M5: 3, M6: 3, S1: 2, S2: 1.

The two earliest clusters are intentionally navigation controls. Mechanism comparisons start in the 26–39 band and are carried through medium, long, and tail regimes. The candidate curve above shows mechanism signature by color, distractor-bearing mazes with X markers, and B1 fixtures with a `B` label.

## Controlled comparisons

### Single-stage key versus switch

M1/M2 pairs below share size, topology, and filename variant. Their one-action length differences largely reflect the executable pickup/toggle semantics rather than different spatial geometry.

| matched cell | M1 key maze | steps | M2 switch maze | steps |
|---|---|---:|---|---:|
| `10x10:dense:1` | `M1/10x10_dense_kr_1` | 37 | `M2/10x10_dense_sg_1` | 36 |
| `10x10:dense:0` | `M1/10x10_dense_kr_0` | 44 | `M2/10x10_dense_sg_0` | 43 |
| `14x14:dense:1` | `M1/14x14_dense_kr_1` | 69 | `M2/14x14_dense_sg_1` | 68 |
| `14x14:corridor:0` | `M1/14x14_corridor_kr_0` | 85 | `M2/14x14_corridor_sg_0` | 84 |

### Depth-2 ordering triplets

M3/M4/M5 triplets hold geometry fixed while changing the required ordering among K→S, S→K, and K→K. These are the cleanest fixtures for separating mechanism type from dependency depth.

| matched cell | M3 K→S | steps | M4 S→K | steps | M5 K→K | steps |
|---|---|---:|---|---:|---|---:|
| `10x10:dense:0` | `M3/10x10_dense_kr_sg_0` | 53 | `M4/10x10_dense_sg_kr_0` | 53 | `M5/10x10_dense_kr_kb_0` | 54 |
| `14x14:corridor:0` | `M3/14x14_corridor_kr_sg_0` | 86 | `M4/14x14_corridor_sg_kr_0` | 86 | `M5/14x14_corridor_kr_kb_0` | 87 |
| `14x14:dense:0` | `M3/14x14_dense_kr_sg_0` | 87 | `M4/14x14_dense_sg_kr_0` | 87 | `M5/14x14_dense_kr_kb_0` | 88 |

## Distractors, five-option cases, and B1 structure

This panel contains 19 distractor mazes. D1 adds a wrong key, D2 adds both a wrong key and an inactive switch, and D3 places a yellow key-door option in a dead-end branch. Distractor counts are kept separate from path depth: a D2/M6 maze is depth 3 but has five total options.

### Five-option mazes

| rank | steps | maze | composition |
|---:|---:|---|---|
| 10 | 31 | `D2/8x8_corridor_wrong_ky_kr_inactive_sb_sg_kb_1` | K→S→K required + wrong key + inactive switch |
| 22 | 48 | `D2/10x10_corridor_wrong_ky_kr_inactive_sb_sg_kb_1` | K→S→K required + wrong key + inactive switch |
| 29 | 61 | `D2/10x10_dense_wrong_ky_kr_inactive_sb_sg_kb_0` | K→S→K required + wrong key + inactive switch |
| 46 | 95 | `D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_0` | K→S→K required + wrong key + inactive switch |
| 49 | 106 | `D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_1` | K→S→K required + wrong key + inactive switch |

### B1 selection

B1 is treated as a protected structural stratum rather than interchangeable with M2. The chosen B1 mazes are:

| rank | steps | size/topology | maze |
|---:|---:|---|---|
| 4 | 26 | 8x8 corridor | `B1/8x8_corridor_swg_0` |
| 5 | 27 | 8x8 corridor | `B1/8x8_corridor_swg_1` |
| 12 | 36 | 10x10 dense | `B1/10x10_dense_swg_1` |
| 16 | 43 | 10x10 corridor | `B1/10x10_corridor_swg_0` |
| 17 | 43 | 10x10 dense | `B1/10x10_dense_swg_0` |
| 19 | 44 | 10x10 corridor | `B1/10x10_corridor_swg_1` |
| 33 | 84 | 14x14 corridor | `B1/14x14_corridor_swg_0` |
| 43 | 90 | 14x14 corridor | `B1/14x14_corridor_swg_1` |

## Long-tail allocation

The panel contains 18 mazes at 80+ estimated optimal actions. These fixtures test whether errors grow with execution horizon, whether mechanism costs compound with navigation length, and whether corridor versus dense topology changes the model's ability to preserve a plan.

| rank | steps | depth | options | signature | topology | maze |
|---:|---:|---:|---:|---|---|---|
| 33 | 84 | 1 | 1 | S | corridor | `B1/14x14_corridor_swg_0` |
| 34 | 84 | 1 | 1 | S | corridor | `M2/14x14_corridor_sg_0` |
| 35 | 85 | 1 | 1 | K | corridor | `M1/14x14_corridor_kr_0` |
| 36 | 86 | 2 | 3 | K→S | corridor | `D1/14x14_corridor_wrong_ky_kr_sg_0` |
| 37 | 86 | 2 | 2 | K→S | corridor | `M3/14x14_corridor_kr_sg_0` |
| 38 | 86 | 2 | 2 | S→K | corridor | `M4/14x14_corridor_sg_kr_0` |
| 39 | 87 | 2 | 2 | K→S | dense | `M3/14x14_dense_kr_sg_0` |
| 40 | 87 | 2 | 2 | S→K | dense | `M4/14x14_dense_sg_kr_0` |
| 41 | 87 | 2 | 2 | K→K | corridor | `M5/14x14_corridor_kr_kb_0` |
| 42 | 88 | 2 | 2 | K→K | dense | `M5/14x14_dense_kr_kb_0` |
| 43 | 90 | 1 | 1 | S | corridor | `B1/14x14_corridor_swg_1` |
| 44 | 92 | 2 | 4 | K→S | corridor | `D2/14x14_corridor_wrong_ky_kr_inactive_sb_sg_1` |
| 45 | 94 | 3 | 3 | K→S→K | corridor | `M6/14x14_corridor_kr_sg_kb_1` |
| 46 | 95 | 3 | 5 | K→S→K | dense | `D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_0` |
| 47 | 96 | 2 | 3 | S→K | dense | `D1/14x14_dense_wrong_ky_sg_kr_1` |
| 48 | 97 | 2 | 3 | K→K | dense | `D1/14x14_dense_wrong_ky_kr_kb_1` |
| 49 | 106 | 3 | 5 | K→S→K | dense | `D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_1` |
| 50 | 106 | 3 | 4 | K→S→K | dense | `D3/14x14_dense_kr_sg_kb_deadend_ky_dy_1` |

## Interpretation risks and gaps

1. The corpus does not contain replicated mechanism mazes at the requested earliest lengths. Only the 9/11-action S1 anchors occupy the first cluster, there are no 16–19-action mazes, and the 20–25-action fixtures require no on-path mechanism. Controlled mechanism comparisons begin at 26 actions.
2. This panel contains 3 D3 maze(s). Prior analysis documents at least one D3 case where the BFS estimate is 31 actions but a legal 23-step model solve exists. D3 values should be treated as estimated lengths pending planner/runtime reconciliation.
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

#### 03. `S2/8x8_corridor_0` — 25 actions

![S2/8x8_corridor_0 maze render](images/003_S2__8x8_corridor_0.png)

- **Structure:** 8x8 corridor, family S2, variant 0.
- **Mechanism path:** none; depth 0, 0 total option(s), 0 distractor option(s).
- **Experimental role:** This is a short winding-corridor navigation baseline. Its optimal path is navigation only; no mechanism stage is required on the optimal path.
- **Why selected:** It represents the second early cluster, still before mechanism comparisons begin and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/S2/8x8_corridor_0.json)

### 26-39 actions

Short mechanism-capable mazes.

#### 04. `B1/8x8_corridor_swg_0` — 26 actions

![B1/8x8_corridor_swg_0 maze render](images/004_B1__8x8_corridor_swg_0.png)

- **Structure:** 8x8 corridor, family B1, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents short mechanism-capable mazes while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/8x8_corridor_swg_0.json)

#### 05. `B1/8x8_corridor_swg_1` — 27 actions

![B1/8x8_corridor_swg_1 maze render](images/005_B1__8x8_corridor_swg_1.png)

- **Structure:** 8x8 corridor, family B1, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents short mechanism-capable mazes while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/8x8_corridor_swg_1.json)

#### 06. `D2/8x8_corridor_wrong_ky_inactive_sb_sg_kr_0` — 28 actions

![D2/8x8_corridor_wrong_ky_inactive_sb_sg_kr_0 maze render](images/006_D2__8x8_corridor_wrong_ky_inactive_sb_sg_kr_0.png)

- **Structure:** 8x8 corridor, family D2, variant 0.
- **Mechanism path:** S→K; depth 2, 4 total option(s), 2 distractor option(s).
- **Experimental role:** This is a wrong-key plus inactive-switch distractor condition. Its optimal path is a switch-gate stage followed by a key-door stage.
- **Why selected:** It represents short mechanism-capable mazes while increasing the panel's off-path choice and distractor load.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D2/8x8_corridor_wrong_ky_inactive_sb_sg_kr_0.json)

#### 07. `D1/8x8_corridor_wrong_ky_kr_kb_0` — 29 actions

![D1/8x8_corridor_wrong_ky_kr_kb_0 maze render](images/007_D1__8x8_corridor_wrong_ky_kr_kb_0.png)

- **Structure:** 8x8 corridor, family D1, variant 0.
- **Mechanism path:** K→K; depth 2, 3 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is two sequential key-door stages.
- **Why selected:** It represents short mechanism-capable mazes while increasing the panel's off-path choice and distractor load.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/8x8_corridor_wrong_ky_kr_kb_0.json)

#### 08. `D1/8x8_corridor_wrong_ky_kr_sg_kb_0` — 30 actions

![D1/8x8_corridor_wrong_ky_kr_sg_kb_0 maze render](images/008_D1__8x8_corridor_wrong_ky_kr_sg_kb_0.png)

- **Structure:** 8x8 corridor, family D1, variant 0.
- **Mechanism path:** K→S→K; depth 3, 4 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents short mechanism-capable mazes while increasing the panel's off-path choice and distractor load.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/8x8_corridor_wrong_ky_kr_sg_kb_0.json)

#### 09. `M6/8x8_corridor_kr_sg_kb_0` — 30 actions

![M6/8x8_corridor_kr_sg_kb_0 maze render](images/009_M6__8x8_corridor_kr_sg_kb_0.png)

- **Structure:** 8x8 corridor, family M6, variant 0.
- **Mechanism path:** K→S→K; depth 3, 3 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->S->K depth-3 condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents short mechanism-capable mazes and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M6/8x8_corridor_kr_sg_kb_0.json)

#### 10. `D2/8x8_corridor_wrong_ky_kr_inactive_sb_sg_kb_1` — 31 actions

![D2/8x8_corridor_wrong_ky_kr_inactive_sb_sg_kb_1 maze render](images/010_D2__8x8_corridor_wrong_ky_kr_inactive_sb_sg_kb_1.png)

- **Structure:** 8x8 corridor, family D2, variant 1.
- **Mechanism path:** K→S→K; depth 3, 5 total option(s), 2 distractor option(s).
- **Experimental role:** This is a wrong-key plus inactive-switch distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents short mechanism-capable mazes while increasing the panel's off-path choice and distractor load.
- **Five-option stress case:** three required stages plus a wrong key and an inactive switch.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D2/8x8_corridor_wrong_ky_kr_inactive_sb_sg_kb_1.json)

#### 11. `M6/8x8_corridor_kr_sg_kb_1` — 31 actions

![M6/8x8_corridor_kr_sg_kb_1 maze render](images/011_M6__8x8_corridor_kr_sg_kb_1.png)

- **Structure:** 8x8 corridor, family M6, variant 1.
- **Mechanism path:** K→S→K; depth 3, 3 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->S->K depth-3 condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents short mechanism-capable mazes and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M6/8x8_corridor_kr_sg_kb_1.json)

#### 12. `B1/10x10_dense_swg_1` — 36 actions

![B1/10x10_dense_swg_1 maze render](images/012_B1__10x10_dense_swg_1.png)

- **Structure:** 10x10 dense, family B1, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents short mechanism-capable mazes while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/10x10_dense_swg_1.json)

#### 13. `M2/10x10_dense_sg_1` — 36 actions

![M2/10x10_dense_sg_1 maze render](images/013_M2__10x10_dense_sg_1.png)

- **Structure:** 10x10 dense, family M2, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single switch-gate condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents short mechanism-capable mazes and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M2/10x10_dense_sg_1.json)

#### 14. `M1/10x10_dense_kr_1` — 37 actions

![M1/10x10_dense_kr_1 maze render](images/014_M1__10x10_dense_kr_1.png)

- **Structure:** 10x10 dense, family M1, variant 1.
- **Mechanism path:** K; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single key-door condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents short mechanism-capable mazes and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M1/10x10_dense_kr_1.json)

### 40-59 actions

Medium-length mazes where ordering and distractor effects become separable.

#### 15. `D2/10x10_dense_wrong_ky_inactive_sb_sg_kr_1` — 42 actions

![D2/10x10_dense_wrong_ky_inactive_sb_sg_kr_1 maze render](images/015_D2__10x10_dense_wrong_ky_inactive_sb_sg_kr_1.png)

- **Structure:** 10x10 dense, family D2, variant 1.
- **Mechanism path:** S→K; depth 2, 4 total option(s), 2 distractor option(s).
- **Experimental role:** This is a wrong-key plus inactive-switch distractor condition. Its optimal path is a switch-gate stage followed by a key-door stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable while increasing the panel's off-path choice and distractor load.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D2/10x10_dense_wrong_ky_inactive_sb_sg_kr_1.json)

#### 16. `B1/10x10_corridor_swg_0` — 43 actions

![B1/10x10_corridor_swg_0 maze render](images/016_B1__10x10_corridor_swg_0.png)

- **Structure:** 10x10 corridor, family B1, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/10x10_corridor_swg_0.json)

#### 17. `B1/10x10_dense_swg_0` — 43 actions

![B1/10x10_dense_swg_0 maze render](images/017_B1__10x10_dense_swg_0.png)

- **Structure:** 10x10 dense, family B1, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/10x10_dense_swg_0.json)

#### 18. `M2/10x10_dense_sg_0` — 43 actions

![M2/10x10_dense_sg_0 maze render](images/018_M2__10x10_dense_sg_0.png)

- **Structure:** 10x10 dense, family M2, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single switch-gate condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M2/10x10_dense_sg_0.json)

#### 19. `B1/10x10_corridor_swg_1` — 44 actions

![B1/10x10_corridor_swg_1 maze render](images/019_B1__10x10_corridor_swg_1.png)

- **Structure:** 10x10 corridor, family B1, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/10x10_corridor_swg_1.json)

#### 20. `M1/10x10_dense_kr_0` — 44 actions

![M1/10x10_dense_kr_0 maze render](images/020_M1__10x10_dense_kr_0.png)

- **Structure:** 10x10 dense, family M1, variant 0.
- **Mechanism path:** K; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single key-door condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M1/10x10_dense_kr_0.json)

#### 21. `D2/10x10_corridor_wrong_ky_inactive_sb_sg_kr_0` — 45 actions

![D2/10x10_corridor_wrong_ky_inactive_sb_sg_kr_0 maze render](images/021_D2__10x10_corridor_wrong_ky_inactive_sb_sg_kr_0.png)

- **Structure:** 10x10 corridor, family D2, variant 0.
- **Mechanism path:** S→K; depth 2, 4 total option(s), 2 distractor option(s).
- **Experimental role:** This is a wrong-key plus inactive-switch distractor condition. Its optimal path is a switch-gate stage followed by a key-door stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable while increasing the panel's off-path choice and distractor load.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D2/10x10_corridor_wrong_ky_inactive_sb_sg_kr_0.json)

#### 22. `D2/10x10_corridor_wrong_ky_kr_inactive_sb_sg_kb_1` — 48 actions

![D2/10x10_corridor_wrong_ky_kr_inactive_sb_sg_kb_1 maze render](images/022_D2__10x10_corridor_wrong_ky_kr_inactive_sb_sg_kb_1.png)

- **Structure:** 10x10 corridor, family D2, variant 1.
- **Mechanism path:** K→S→K; depth 3, 5 total option(s), 2 distractor option(s).
- **Experimental role:** This is a wrong-key plus inactive-switch distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable while increasing the panel's off-path choice and distractor load.
- **Five-option stress case:** three required stages plus a wrong key and an inactive switch.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D2/10x10_corridor_wrong_ky_kr_inactive_sb_sg_kb_1.json)

#### 23. `D3/10x10_dense_kr_sg_deadend_ky_dy_0` — 53 actions

![D3/10x10_dense_kr_sg_deadend_ky_dy_0 maze render](images/023_D3__10x10_dense_kr_sg_deadend_ky_dy_0.png)

- **Structure:** 10x10 dense, family D3, variant 0.
- **Mechanism path:** K→S; depth 2, 3 total option(s), 1 distractor option(s).
- **Experimental role:** This is a dead-end yellow key-door distractor condition. Its optimal path is a key-door stage followed by a switch-gate stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable while increasing the panel's off-path choice and distractor load.
- **Length caveat:** D3 dead-end fixture; the BFS estimate may be inflated relative to legal runtime execution.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D3/10x10_dense_kr_sg_deadend_ky_dy_0.json)

#### 24. `M3/10x10_dense_kr_sg_0` — 53 actions

![M3/10x10_dense_kr_sg_0 maze render](images/024_M3__10x10_dense_kr_sg_0.png)

- **Structure:** 10x10 dense, family M3, variant 0.
- **Mechanism path:** K→S; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->S depth-2 condition. Its optimal path is a key-door stage followed by a switch-gate stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M3/10x10_dense_kr_sg_0.json)

#### 25. `M4/10x10_dense_sg_kr_0` — 53 actions

![M4/10x10_dense_sg_kr_0 maze render](images/025_M4__10x10_dense_sg_kr_0.png)

- **Structure:** 10x10 dense, family M4, variant 0.
- **Mechanism path:** S→K; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled S->K depth-2 condition. Its optimal path is a switch-gate stage followed by a key-door stage.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M4/10x10_dense_sg_kr_0.json)

#### 26. `D1/10x10_dense_wrong_ky_kr_kb_0` — 54 actions

![D1/10x10_dense_wrong_ky_kr_kb_0 maze render](images/026_D1__10x10_dense_wrong_ky_kr_kb_0.png)

- **Structure:** 10x10 dense, family D1, variant 0.
- **Mechanism path:** K→K; depth 2, 3 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is two sequential key-door stages.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable while increasing the panel's off-path choice and distractor load.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/10x10_dense_wrong_ky_kr_kb_0.json)

#### 27. `M5/10x10_dense_kr_kb_0` — 54 actions

![M5/10x10_dense_kr_kb_0 maze render](images/027_M5__10x10_dense_kr_kb_0.png)

- **Structure:** 10x10 dense, family M5, variant 0.
- **Mechanism path:** K→K; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->K depth-2 condition. Its optimal path is two sequential key-door stages.
- **Why selected:** It represents medium-length mazes where ordering and distractor effects become separable and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M5/10x10_dense_kr_kb_0.json)

### 60-79 actions

Long mazes bridging the medium regime to the extreme tail.

#### 28. `D1/10x10_dense_wrong_ky_kr_sg_kb_0` — 61 actions

![D1/10x10_dense_wrong_ky_kr_sg_kb_0 maze render](images/028_D1__10x10_dense_wrong_ky_kr_sg_kb_0.png)

- **Structure:** 10x10 dense, family D1, variant 0.
- **Mechanism path:** K→S→K; depth 3, 4 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents long mazes bridging the medium regime to the extreme tail while increasing the panel's off-path choice and distractor load.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/10x10_dense_wrong_ky_kr_sg_kb_0.json)

#### 29. `D2/10x10_dense_wrong_ky_kr_inactive_sb_sg_kb_0` — 61 actions

![D2/10x10_dense_wrong_ky_kr_inactive_sb_sg_kb_0 maze render](images/029_D2__10x10_dense_wrong_ky_kr_inactive_sb_sg_kb_0.png)

- **Structure:** 10x10 dense, family D2, variant 0.
- **Mechanism path:** K→S→K; depth 3, 5 total option(s), 2 distractor option(s).
- **Experimental role:** This is a wrong-key plus inactive-switch distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents long mazes bridging the medium regime to the extreme tail while increasing the panel's off-path choice and distractor load.
- **Five-option stress case:** three required stages plus a wrong key and an inactive switch.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D2/10x10_dense_wrong_ky_kr_inactive_sb_sg_kb_0.json)

#### 30. `M2/14x14_dense_sg_1` — 68 actions

![M2/14x14_dense_sg_1 maze render](images/030_M2__14x14_dense_sg_1.png)

- **Structure:** 14x14 dense, family M2, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single switch-gate condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents long mazes bridging the medium regime to the extreme tail and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M2/14x14_dense_sg_1.json)

#### 31. `D3/14x14_dense_kr_deadend_ky_dy_1` — 69 actions

![D3/14x14_dense_kr_deadend_ky_dy_1 maze render](images/031_D3__14x14_dense_kr_deadend_ky_dy_1.png)

- **Structure:** 14x14 dense, family D3, variant 1.
- **Mechanism path:** K; depth 1, 2 total option(s), 1 distractor option(s).
- **Experimental role:** This is a dead-end yellow key-door distractor condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents long mazes bridging the medium regime to the extreme tail while increasing the panel's off-path choice and distractor load.
- **Length caveat:** D3 dead-end fixture; the BFS estimate may be inflated relative to legal runtime execution.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D3/14x14_dense_kr_deadend_ky_dy_1.json)

#### 32. `M1/14x14_dense_kr_1` — 69 actions

![M1/14x14_dense_kr_1 maze render](images/032_M1__14x14_dense_kr_1.png)

- **Structure:** 14x14 dense, family M1, variant 1.
- **Mechanism path:** K; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single key-door condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents long mazes bridging the medium regime to the extreme tail and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M1/14x14_dense_kr_1.json)

### 80+ actions

The extreme long-horizon tail dominated by 14x14 structures.

#### 33. `B1/14x14_corridor_swg_0` — 84 actions

![B1/14x14_corridor_swg_0 maze render](images/033_B1__14x14_corridor_swg_0.png)

- **Structure:** 14x14 corridor, family B1, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/14x14_corridor_swg_0.json)

#### 34. `M2/14x14_corridor_sg_0` — 84 actions

![M2/14x14_corridor_sg_0 maze render](images/034_M2__14x14_corridor_sg_0.png)

- **Structure:** 14x14 corridor, family M2, variant 0.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single switch-gate condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M2/14x14_corridor_sg_0.json)

#### 35. `M1/14x14_corridor_kr_0` — 85 actions

![M1/14x14_corridor_kr_0 maze render](images/035_M1__14x14_corridor_kr_0.png)

- **Structure:** 14x14 corridor, family M1, variant 0.
- **Mechanism path:** K; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled single key-door condition. Its optimal path is one required key-door stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M1/14x14_corridor_kr_0.json)

#### 36. `D1/14x14_corridor_wrong_ky_kr_sg_0` — 86 actions

![D1/14x14_corridor_wrong_ky_kr_sg_0 maze render](images/036_D1__14x14_corridor_wrong_ky_kr_sg_0.png)

- **Structure:** 14x14 corridor, family D1, variant 0.
- **Mechanism path:** K→S; depth 2, 3 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is a key-door stage followed by a switch-gate stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures while increasing the panel's off-path choice and distractor load.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/14x14_corridor_wrong_ky_kr_sg_0.json)

#### 37. `M3/14x14_corridor_kr_sg_0` — 86 actions

![M3/14x14_corridor_kr_sg_0 maze render](images/037_M3__14x14_corridor_kr_sg_0.png)

- **Structure:** 14x14 corridor, family M3, variant 0.
- **Mechanism path:** K→S; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->S depth-2 condition. Its optimal path is a key-door stage followed by a switch-gate stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M3/14x14_corridor_kr_sg_0.json)

#### 38. `M4/14x14_corridor_sg_kr_0` — 86 actions

![M4/14x14_corridor_sg_kr_0 maze render](images/038_M4__14x14_corridor_sg_kr_0.png)

- **Structure:** 14x14 corridor, family M4, variant 0.
- **Mechanism path:** S→K; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled S->K depth-2 condition. Its optimal path is a switch-gate stage followed by a key-door stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M4/14x14_corridor_sg_kr_0.json)

#### 39. `M3/14x14_dense_kr_sg_0` — 87 actions

![M3/14x14_dense_kr_sg_0 maze render](images/039_M3__14x14_dense_kr_sg_0.png)

- **Structure:** 14x14 dense, family M3, variant 0.
- **Mechanism path:** K→S; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->S depth-2 condition. Its optimal path is a key-door stage followed by a switch-gate stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M3/14x14_dense_kr_sg_0.json)

#### 40. `M4/14x14_dense_sg_kr_0` — 87 actions

![M4/14x14_dense_sg_kr_0 maze render](images/040_M4__14x14_dense_sg_kr_0.png)

- **Structure:** 14x14 dense, family M4, variant 0.
- **Mechanism path:** S→K; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled S->K depth-2 condition. Its optimal path is a switch-gate stage followed by a key-door stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M4/14x14_dense_sg_kr_0.json)

#### 41. `M5/14x14_corridor_kr_kb_0` — 87 actions

![M5/14x14_corridor_kr_kb_0 maze render](images/041_M5__14x14_corridor_kr_kb_0.png)

- **Structure:** 14x14 corridor, family M5, variant 0.
- **Mechanism path:** K→K; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->K depth-2 condition. Its optimal path is two sequential key-door stages.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M5/14x14_corridor_kr_kb_0.json)

#### 42. `M5/14x14_dense_kr_kb_0` — 88 actions

![M5/14x14_dense_kr_kb_0 maze render](images/042_M5__14x14_dense_kr_kb_0.png)

- **Structure:** 14x14 dense, family M5, variant 0.
- **Mechanism path:** K→K; depth 2, 2 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->K depth-2 condition. Its optimal path is two sequential key-door stages.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M5/14x14_dense_kr_kb_0.json)

#### 43. `B1/14x14_corridor_swg_1` — 90 actions

![B1/14x14_corridor_swg_1 maze render](images/043_B1__14x14_corridor_swg_1.png)

- **Structure:** 14x14 corridor, family B1, variant 1.
- **Mechanism path:** S; depth 1, 1 total option(s), 0 distractor option(s).
- **Experimental role:** This is a white-switch B1 structural variant retained as a distinct switch condition. Its optimal path is one required switch-gate stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures while preserving the distinct B1 switch structure.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/B1/14x14_corridor_swg_1.json)

#### 44. `D2/14x14_corridor_wrong_ky_kr_inactive_sb_sg_1` — 92 actions

![D2/14x14_corridor_wrong_ky_kr_inactive_sb_sg_1 maze render](images/044_D2__14x14_corridor_wrong_ky_kr_inactive_sb_sg_1.png)

- **Structure:** 14x14 corridor, family D2, variant 1.
- **Mechanism path:** K→S; depth 2, 4 total option(s), 2 distractor option(s).
- **Experimental role:** This is a wrong-key plus inactive-switch distractor condition. Its optimal path is a key-door stage followed by a switch-gate stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures while increasing the panel's off-path choice and distractor load.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D2/14x14_corridor_wrong_ky_kr_inactive_sb_sg_1.json)

#### 45. `M6/14x14_corridor_kr_sg_kb_1` — 94 actions

![M6/14x14_corridor_kr_sg_kb_1 maze render](images/045_M6__14x14_corridor_kr_sg_kb_1.png)

- **Structure:** 14x14 corridor, family M6, variant 1.
- **Mechanism path:** K→S→K; depth 3, 3 total option(s), 0 distractor option(s).
- **Experimental role:** This is a controlled K->S->K depth-3 condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures and helps maintain the intended smooth difficulty curve.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/M6/14x14_corridor_kr_sg_kb_1.json)

#### 46. `D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_0` — 95 actions

![D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_0 maze render](images/046_D2__14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_0.png)

- **Structure:** 14x14 dense, family D2, variant 0.
- **Mechanism path:** K→S→K; depth 3, 5 total option(s), 2 distractor option(s).
- **Experimental role:** This is a wrong-key plus inactive-switch distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures while increasing the panel's off-path choice and distractor load.
- **Five-option stress case:** three required stages plus a wrong key and an inactive switch.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_0.json)

#### 47. `D1/14x14_dense_wrong_ky_sg_kr_1` — 96 actions

![D1/14x14_dense_wrong_ky_sg_kr_1 maze render](images/047_D1__14x14_dense_wrong_ky_sg_kr_1.png)

- **Structure:** 14x14 dense, family D1, variant 1.
- **Mechanism path:** S→K; depth 2, 3 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is a switch-gate stage followed by a key-door stage.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures while increasing the panel's off-path choice and distractor load.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/14x14_dense_wrong_ky_sg_kr_1.json)

#### 48. `D1/14x14_dense_wrong_ky_kr_kb_1` — 97 actions

![D1/14x14_dense_wrong_ky_kr_kb_1 maze render](images/048_D1__14x14_dense_wrong_ky_kr_kb_1.png)

- **Structure:** 14x14 dense, family D1, variant 1.
- **Mechanism path:** K→K; depth 2, 3 total option(s), 1 distractor option(s).
- **Experimental role:** This is a wrong-key distractor condition. Its optimal path is two sequential key-door stages.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures while increasing the panel's off-path choice and distractor load.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D1/14x14_dense_wrong_ky_kr_kb_1.json)

#### 49. `D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_1` — 106 actions

![D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_1 maze render](images/049_D2__14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_1.png)

- **Structure:** 14x14 dense, family D2, variant 1.
- **Mechanism path:** K→S→K; depth 3, 5 total option(s), 2 distractor option(s).
- **Experimental role:** This is a wrong-key plus inactive-switch distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures while increasing the panel's off-path choice and distractor load.
- **Five-option stress case:** three required stages plus a wrong key and an inactive switch.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_1.json)

#### 50. `D3/14x14_dense_kr_sg_kb_deadend_ky_dy_1` — 106 actions

![D3/14x14_dense_kr_sg_kb_deadend_ky_dy_1 maze render](images/050_D3__14x14_dense_kr_sg_kb_deadend_ky_dy_1.png)

- **Structure:** 14x14 dense, family D3, variant 1.
- **Mechanism path:** K→S→K; depth 3, 4 total option(s), 1 distractor option(s).
- **Experimental role:** This is a dead-end yellow key-door distractor condition. Its optimal path is the deepest available chain: key-door, switch-gate, then key-door.
- **Why selected:** It represents the extreme long-horizon tail dominated by 14x14 structures while increasing the panel's off-path choice and distractor load.
- **Length caveat:** D3 dead-end fixture; the BFS estimate may be inflated relative to legal runtime execution.
- **Source:** [JSON](../../../../ogbench/ogbench/procgen/maze_jsons/D3/14x14_dense_kr_sg_kb_deadend_ky_dy_1.json)
