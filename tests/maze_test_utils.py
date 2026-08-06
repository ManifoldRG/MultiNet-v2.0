import json
from pathlib import Path

from BFS_solver import solve


# The experiment corpus lives in the ogbench submodule — the same files every
# R1 manifest resolves. (mazes/exp_maze_jsons/ was a stale duplicate, removed
# 2026-08: it had drifted on 9 files and carried a goal-invariant violation.)
MAZE_JSON_DIR = (
	Path(__file__).resolve().parent.parent
	/ 'ogbench' / 'ogbench' / 'procgen' / 'maze_jsons'
)
if not MAZE_JSON_DIR.exists():
	raise RuntimeError(
		f'Maze corpus missing at {MAZE_JSON_DIR} — the ogbench submodule is '
		'not initialized. Run: git submodule update --init'
	)
MECHANISM_KEYS = ('keys', 'doors', 'switches', 'gates')


# --- Known corpus defects: FROZEN, not fixed ---------------------------------
# The ogbench submodule above is the exact corpus every paid R1 episode
# resolved, including the files below — it must not be edited or re-pinned
# pre-release. None of these have runtime impact: chain_pattern and
# difficulty_tier are labels no runtime/scoring/prompting code reads
# (difficulty is computed via BFS), and every file remains self-consistent on
# maze.goal == goal.target (see test_maze_goal_invariant.py). Each entry names
# the known-bad value the affected test pins instead of the normal
# expectation, so any future change — including the eventual ogbench-fork
# fix — turns the suite red and forces the entry's removal.
KNOWN_CORPUS_DEFECTS = {
	'm_vs_s4_goal_desync': {
		'description': (
			"submodule commit 31a0549 fixed S4/10x10_dense_1.json's goal from "
			'[8, 1] to the correct [8, 8] but did not propagate the fix to '
			'these M-family structural counterparts, which still carry the '
			'pre-fix goal.'
		),
		'files': (
			'M1/10x10_dense_kr_1.json',
			'M2/10x10_dense_sg_1.json',
			'M3/10x10_dense_kr_sg_1.json',
			'M4/10x10_dense_sg_kr_1.json',
			'M5/10x10_dense_kr_kb_1.json',
			'M6/10x10_dense_kr_sg_kb_1.json',
		),
		'bad_goal': [8, 1],
		'counterpart_goal': [8, 8],  # S4/10x10_dense_1.json, fixed by 31a0549
	},
	'm2_chain_pattern_corruption': {
		'description': (
			'submodule commit 271bf47 mislabeled these pure switch/gate mazes '
			'as key_door (description text and metadata.chain_pattern); the '
			'retired mazes/exp_maze_jsons duplicate carried the correct '
			'single_switch_gate label for the same files.'
		),
		'files': (
			'M2/8x8_corridor_sg_0.json',
			'M2/8x8_corridor_sg_1.json',
		),
		'bad_chain_pattern': 'key_door',
	},
	'd1_m2_difficulty_tier_mismatch': {
		'description': (
			"D1/8x8_corridor_wrong_ky_sg_1.json's difficulty_tier drifted to "
			'4 in the submodule; its M2 counterpart kept the correct tier 3.'
		),
		'd_file': 'D1/8x8_corridor_wrong_ky_sg_1.json',
		'm_file': 'M2/8x8_corridor_sg_1.json',
		'bad_tiers': (4, 3),  # (D1 difficulty_tier, M2 difficulty_tier)
	},
	'd2_chain_pattern_corruption': {
		'description': (
			'Same submodule commit 271bf47 regression as '
			'm2_chain_pattern_corruption, on the D2 wrong-key overlays of the '
			'same base mazes. No existing D2 test reads metadata.chain_pattern '
			'today, so this entry has no matching test exemption — it is '
			'listed here for the record.'
		),
		'files': (
			'D2/8x8_corridor_wrong_ky_inactive_sb_sg_0.json',
			'D2/8x8_corridor_wrong_ky_inactive_sb_sg_1.json',
			'D2/10x10_corridor_wrong_ky_inactive_sb_sg_1.json',
		),
	},
}


def load_maze_specs(maze_type, *, include_file_name=False):
	specs = []
	for path in sorted((MAZE_JSON_DIR / maze_type).glob('*.json')):
		spec = json.loads(path.read_text(encoding='utf-8'))
		if include_file_name:
			specs.append((path.name, spec))
		else:
			specs.append(spec)
	return specs


def assert_navigation_contract(test_case, spec):
	maze = spec['maze']
	width, height = maze['dimensions']
	start = maze['start']
	goal = maze['goal']
	walls = {tuple(wall) for wall in maze['walls']}

	test_case.assertEqual(spec['goal']['type'], 'reach_position')
	if spec['goal'].get('target') is not None:
		test_case.assertEqual(spec['goal']['target'], goal)
	test_case.assertEqual(len(start), 2)
	test_case.assertEqual(len(goal), 2)
	test_case.assertNotEqual(start, goal)
	for label, point in (('start', start), ('goal', goal)):
		x, y = point
		test_case.assertGreaterEqual(x, 0, label)
		test_case.assertLess(x, width, label)
		test_case.assertGreaterEqual(y, 0, label)
		test_case.assertLess(y, height, label)
		test_case.assertNotIn(tuple(point), walls, label)


def assert_goal_target_matches_maze_goal(test_case, spec):
	test_case.assertEqual(spec['goal']['target'], spec['maze']['goal'])


def assert_bfs_solver_finds_path_to_goal(test_case, spec):
	result = solve(spec)
	test_case.assertTrue(result['is_solvable'])
	test_case.assertEqual(result['path'][0], tuple(spec['maze']['start']))
	test_case.assertEqual(result['path'][-1], tuple(spec['maze']['goal']))
	test_case.assertEqual(result['optimal_cost'], len(result['path']) - 1)

	walls = {tuple(wall) for wall in spec['maze']['walls']}
	for current, next_cell in zip(result['path'], result['path'][1:]):
		test_case.assertNotIn(current, walls)
		test_case.assertEqual(
			abs(current[0] - next_cell[0]) + abs(current[1] - next_cell[1]),
			1,
		)
	test_case.assertNotIn(result['path'][-1], walls)
	return result


def assert_no_mechanisms(test_case, spec):
	mechanisms = spec['mechanisms']
	for key in MECHANISM_KEYS:
		test_case.assertEqual(mechanisms[key], [], key)
	for key, value in mechanisms.items():
		test_case.assertEqual(value, [], key)
	test_case.assertEqual(spec['rules']['hidden_mechanisms'], [])
	test_case.assertEqual(spec['metadata']['chain_pattern'], 'none')


def assert_standard_mechanism_groups(test_case, spec):
	test_case.assertEqual(
		set(spec['mechanisms']),
		{'keys', 'doors', 'switches', 'gates'},
	)


def assert_no_hidden_or_auxiliary_mechanisms(test_case, spec):
	test_case.assertEqual(spec['rules']['hidden_mechanisms'], [])
	test_case.assertEqual(spec['goal']['auxiliary_conditions'], [])


def assert_key_door_chain_on_path(test_case, spec, *, key_id, door_id):
	mechanisms = spec['mechanisms']
	key = next(key for key in mechanisms['keys'] if key['id'] == key_id)
	door = next(door for door in mechanisms['doors'] if door['id'] == door_id)

	to_door = {
		**spec,
		'maze': {
			**spec['maze'],
			'goal': door['position'],
		},
		'goal': {
			**spec['goal'],
			'target': door['position'],
		},
	}
	to_door_result = solve(to_door)
	test_case.assertTrue(to_door_result['is_solvable'])
	test_case.assertEqual(to_door_result['path'][-1], tuple(door['position']))
	test_case.assertIn(f'pickup:{key_id}', to_door_result['interactions'])
	test_case.assertIn(f'open:{door_id}', to_door_result['interactions'])
	test_case.assertIn(tuple(key['position']), to_door_result['path'])

	from_door = {
		**spec,
		'maze': {
			**spec['maze'],
			'start': door['position'],
		},
		'mechanisms': {
			**mechanisms,
			'doors': [item for item in mechanisms['doors'] if item['id'] != door_id],
		},
	}
	from_door_result = solve(from_door)
	test_case.assertTrue(from_door_result['is_solvable'])
	test_case.assertEqual(from_door_result['path'][0], tuple(door['position']))
	test_case.assertEqual(from_door_result['path'][-1], tuple(spec['maze']['goal']))
