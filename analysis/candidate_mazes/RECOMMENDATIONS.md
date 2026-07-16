# Recommended 50-maze panels

All four panels replace the 9/11-action S1 empty rooms with the two longest
mechanism-free navigation controls: the 77/89-action S5 corridor variants. Each
panel still contains eight of the ten B1 mazes, includes a 106-action endpoint,
and avoids the duplicated `task_id` collision. Lengths are current executable-BFS
estimates (turns, moves, pickup, and toggle).

| panel | median | 26-39 | 40-59 | 60-79 | 80+ | depth 2 | depth 3 | distractor mazes | five-option | exact M1/M2 pairs | exact M3/M4/M5 triplets |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Balanced 03 | 53.0 | 12 | 14 | 8 | 14 | 13 | 9 | 10 | 4 | 8 | 3 |
| Long-tail 02 | 73.0 | 8 | 12 | 6 | 22 | 13 | 9 | 12 | 3 | 8 | 2 |
| Mechanism-rich 01 | 58.0 | 12 | 13 | 5 | 19 | 20 | 10 | 20 | 6 | 4 | 2 |
| Pairwise 01 | 46.0 | 12 | 16 | 5 | 15 | 16 | 6 | 8 | 2 | 8 | 4 |

The recommended panels now begin at 23 actions. Balanced, long-tail, and
pairwise retain two 20-25-action controls; mechanism-rich retains one. The
source corpus still has no mazes at 16-19 actions.

## 1. Balanced 03 — recommended default

- Full list: [balanced_03.csv](balanced_03.csv)
- Metadata: [balanced_03.json](balanced_03.json)
- Diagnostic: [balanced_03.png](balanced_03.png)
- Strength: best general-purpose compromise. It has two M1 keys and two M2
  switches in each feasible comparison band (26-39, 40-59, 60-81, and 82+),
  eight exact M1/M2 layout matches, and three exact depth-2 triplets.
- Strength: four five-option mazes, nine depth-3 mazes, and only one D3 maze
  with the known BFS-length caveat.
- Strength: the 77/89-action S5 pair adds replicated mechanism-free controls in
  the long regime and reduces the largest adjacent length gap to 10 actions.
- Risk: distractor coverage (10/50) is intentionally moderate; use
  mechanism-rich if distractor effects are primary.

## 2. Long-tail 02 — maximum length emphasis

- Full list: [long_tail_02.csv](long_tail_02.csv)
- Metadata: [long_tail_02.json](long_tail_02.json)
- Diagnostic: [long_tail_02.png](long_tail_02.png)
- Strength: 22/50 mazes are 80+ actions, while the list still keeps two pure
  keys and two pure switches in every feasible comparison band.
- Strength: nine depth-3 mazes, three five-option mazes, and eight exact M1/M2
  pairs.
- Risk: the median rises to 73 actions and only eight mazes occupy 26-39, so
  estimates at the easy end will have less resolution.

## 3. Mechanism-rich 01 — distractor/depth stress panel

- Full list: [mechanism_rich_01.csv](mechanism_rich_01.csv)
- Metadata: [mechanism_rich_01.json](mechanism_rich_01.json)
- Diagnostic: [mechanism_rich_01.png](mechanism_rich_01.png)
- Strength: 20 distractor mazes, 20 depth-2 mazes, 10 depth-3 mazes, and six
  five-option `K->S->K + wrong key + inactive switch` mazes.
- Strength: still covers 19 mazes at 80+ actions and keeps one M1/M2 control
  pair in each feasible length region.
- Risk: pure key-vs-switch replication is only one per type per region, so this
  is a stress panel rather than the primary causal K/S comparison. Five D3
  mazes carry the known BFS-length caveat.

## 4. Pairwise 01 — strongest controlled comparisons

- Full list: [pairwise_01.csv](pairwise_01.csv)
- Metadata: [pairwise_01.json](pairwise_01.json)
- Diagnostic: [pairwise_01.png](pairwise_01.png)
- Strength: eight exact M1/M2 pairs plus four exact M3/M4/M5 triplets. It has
  at least two pure-key and two pure-switch observations in all four feasible
  length regions.
- Strength: retains 16 depth-2, six depth-3, 15 tail, and two five-option mazes.
- Risk: only eight distractor mazes; it is optimized for controlled mechanism
  contrasts rather than distractor prevalence.

## Corpus gaps and integrity risks

1. The requested early replication does not exist: only `S1` gives the 9/11
   anchors. These four exploratory panels now deliberately omit them in favor
   of S5 at 77/89 actions. The first mechanism comparisons begin at 26 actions.
2. `analysis/.cache/phase1_notes.md` documents a D3 example where BFS reports
   31 but a model produced a legal 23-step solve. Treat all D3 lengths as
   estimates pending planner/runtime reconciliation.
3. Two distinct D2 files share task ID
   `10x10_corridor_wrong_ky_inactive_sb_sg_kr_1`; the selector forbids both
   from entering the same panel.
4. The maximum current estimate is 106 actions. Four mazes share it: M6, D1,
   D2, and D3 variants of the 14x14 dense `K->S->K` layout.
5. Exactly ten five-option mazes exist, at 30, 31, 47, 48, 58, 61, 88, 94,
   95, and 106 actions.

## Reproduce or search alternatives

```bash
python -m analysis.propose_maze_candidates \
  --profiles all \
  --seed 20260713 \
  --proposals 30 \
  --iterations 500 \
  --keep 3
```

Change `--seed` for new panels or increase `--proposals`/`--iterations` for a
larger search. The output keeps alternatives that differ by at least four
mazes, so retained lists are materially distinct rather than reorderings.
