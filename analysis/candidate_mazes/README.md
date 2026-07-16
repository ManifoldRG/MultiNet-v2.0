# OGBench 50-maze candidate panels

Corpus: 214 JSON files, 214 BFS-beatable. Length means executable 
turn/move/pickup/toggle actions from the current BFS planner.

`path_depth` counts required key-door/switch-gate stages; `option_count` adds distractors. 
D3 lengths are flagged because `analysis/.cache/phase1_notes.md` documents a known 
BFS over-count on a dense dead-end decoy-key maze (31 reported versus a 23-step legal solve).

## Candidate overview

| profile | min | median | max | largest gap | 10-15 | 20-25 | 80+ | depth 2 | depth 3 | distractor | five-option | B1 | M1/M2 matched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | 9 | 48.5 | 106 | 12 | 1 | 2 | 14 | 13 | 9 | 11 | 3 | 8 | 8 |
| long_tail | 9 | 64.5 | 106 | 12 | 1 | 2 | 19 | 11 | 10 | 15 | 4 | 8 | 7 |
| mechanism_rich | 23 | 58.0 | 106 | 9 | 0 | 1 | 19 | 20 | 10 | 20 | 6 | 8 | 4 |
| pairwise | 23 | 46.0 | 106 | 11 | 0 | 2 | 15 | 16 | 6 | 8 | 2 | 8 | 8 |

## Balanced

Best overall compromise: two early clusters, broad mechanism coverage, and a long tail.

- Length clusters — <10: 1, 10-15: 1, 16-19: 0, 20-25: 2, 26-39: 11, 40-59: 14, 60-79: 7, 80+: 14.
- Path signatures — none: 4, K: 8, S: 16, K→K: 5, K→S: 3, S→K: 5, K→S→K: 9.
- Families — B1: 8, D1: 3, D2: 4, D3: 4, M1: 8, M2: 8, M3: 3, M4: 3, M5: 3, M6: 3, S1: 2, S4: 1.
- Matched geometry cells: M1-vs-M2 = 8; M3-vs-M4-vs-M5 = 3.
- Known-risk D3 lengths in panel: 4.
- Risks — length curve has a 12-action adjacent gap.

## Long Tail

Maximizes 80+ action coverage and preserves the corpus maximum while retaining both early clusters.

- Length clusters — <10: 1, 10-15: 1, 16-19: 0, 20-25: 2, 26-39: 10, 40-59: 10, 60-79: 7, 80+: 19.
- Path signatures — none: 5, K: 8, S: 16, K→K: 5, K→S: 4, S→K: 2, K→S→K: 10.
- Families — B1: 8, D1: 7, D2: 5, D3: 3, M1: 8, M2: 8, M3: 2, M4: 2, M5: 2, M6: 2, S1: 2, S4: 1.
- Matched geometry cells: M1-vs-M2 = 7; M3-vs-M4-vs-M5 = 2.
- Known-risk D3 lengths in panel: 3.
- Risks — length curve has a 12-action adjacent gap.

## Mechanism Rich

Favors depth-2/depth-3 chains, distractors, and five-option D2/M6 mazes.

- Length clusters — <10: 0, 10-15: 0, 16-19: 0, 20-25: 1, 26-39: 12, 40-59: 13, 60-79: 5, 80+: 19.
- Path signatures — none: 3, K: 5, S: 12, K→K: 6, K→S: 7, S→K: 7, K→S→K: 10.
- Families — B1: 8, D1: 8, D2: 7, D3: 5, M1: 4, M2: 4, M3: 3, M4: 3, M5: 3, M6: 3, S5: 2.
- Matched geometry cells: M1-vs-M2 = 4; M3-vs-M4-vs-M5 = 2.
- Known-risk D3 lengths in panel: 5.
- Risks — fewer than two controlled M1/M2 observations in short comparison (26-39), medium comparison (40-59), long comparison (60-81), tail comparison (82+).

## Pairwise

Prioritizes matched K-vs-S geometries and repeated depth-2 order comparisons.

- Length clusters — <10: 0, 10-15: 0, 16-19: 0, 20-25: 2, 26-39: 12, 40-59: 16, 60-79: 5, 80+: 15.
- Path signatures — none: 4, K: 8, S: 16, K→K: 5, K→S: 6, S→K: 5, K→S→K: 6.
- Families — B1: 8, D1: 2, D2: 4, D3: 2, M1: 8, M2: 8, M3: 4, M4: 4, M5: 5, M6: 2, S2: 1, S5: 2.
- Matched geometry cells: M1-vs-M2 = 8; M3-vs-M4-vs-M5 = 4.
- Known-risk D3 lengths in panel: 2.
- Risks — no automated coverage warning; D3 optimality caveat still applies.

## Corpus longest tail

| steps | depth | options | distractors | signature | maze |
|---:|---:|---:|---:|---|---|
| 106 | 3 | 5 | 2 | K→S→K | `D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_1` |
| 106 | 3 | 4 | 1 | K→S→K | `D1/14x14_dense_wrong_ky_kr_sg_kb_1` |
| 106 | 3 | 4 | 1 | K→S→K | `D3/14x14_dense_kr_sg_kb_deadend_ky_dy_1` |
| 106 | 3 | 3 | 0 | K→S→K | `M6/14x14_dense_kr_sg_kb_1` |
| 97 | 2 | 3 | 1 | K→K | `D1/14x14_dense_wrong_ky_kr_kb_1` |
| 97 | 2 | 3 | 1 | K→K | `D3/14x14_dense_kr_kb_deadend_ky_dy_1` |
| 97 | 2 | 2 | 0 | K→K | `M5/14x14_dense_kr_kb_1` |
| 96 | 2 | 4 | 2 | S→K | `D2/14x14_dense_wrong_ky_inactive_sb_sg_kr_1` |
| 96 | 2 | 4 | 2 | K→S | `D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_1` |
| 96 | 2 | 3 | 1 | K→S | `D1/14x14_dense_wrong_ky_kr_sg_1` |
| 96 | 2 | 3 | 1 | S→K | `D1/14x14_dense_wrong_ky_sg_kr_1` |
| 96 | 2 | 3 | 1 | K→S | `D3/14x14_dense_kr_sg_deadend_ky_dy_1` |
| 96 | 2 | 3 | 1 | S→K | `D3/14x14_dense_sg_kr_deadend_ky_dy_1` |
| 96 | 2 | 2 | 0 | K→S | `M3/14x14_dense_kr_sg_1` |
| 96 | 2 | 2 | 0 | S→K | `M4/14x14_dense_sg_kr_1` |

## Five-option corpus

These are the 3-on-path + 2-distractor cases.

| steps | maze |
|---:|---|
| 30 | `D2/8x8_corridor_wrong_ky_kr_inactive_sb_sg_kb_0` |
| 31 | `D2/8x8_corridor_wrong_ky_kr_inactive_sb_sg_kb_1` |
| 47 | `D2/10x10_corridor_wrong_ky_kr_inactive_sb_sg_kb_0` |
| 48 | `D2/10x10_corridor_wrong_ky_kr_inactive_sb_sg_kb_1` |
| 58 | `D2/10x10_dense_wrong_ky_kr_inactive_sb_sg_kb_1` |
| 61 | `D2/10x10_dense_wrong_ky_kr_inactive_sb_sg_kb_0` |
| 88 | `D2/14x14_corridor_wrong_ky_kr_inactive_sb_sg_kb_0` |
| 94 | `D2/14x14_corridor_wrong_ky_kr_inactive_sb_sg_kb_1` |
| 95 | `D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_0` |
| 106 | `D2/14x14_dense_wrong_ky_kr_inactive_sb_sg_kb_1` |

## Data-integrity warning

The following distinct files share a `task_id`; no generated panel includes both:

- `10x10_corridor_wrong_ky_inactive_sb_sg_kr_1`: `D2/10x10_corridor_wrong_ky_inactive_sb_sg_1`, `D2/10x10_corridor_wrong_ky_inactive_sb_sg_kr_1`
