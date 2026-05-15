import json
from pathlib import Path

from BFS_solver import solve


MAZE_JSON_DIR = Path(__file__).resolve().parent.parent / 'mazes' / 'exp_maze_jsons'
MECHANISM_KEYS = ('keys', 'doors', 'switches', 'gates')


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
