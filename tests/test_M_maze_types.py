import sys
import unittest
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
	sys.path.insert(0, _REPO_ROOT)

from maze_test_utils import (
	MECHANISM_KEYS,
	assert_bfs_solver_finds_path_to_goal,
	assert_goal_target_matches_maze_goal,
	assert_navigation_contract,
	assert_no_hidden_or_auxiliary_mechanisms,
	assert_standard_mechanism_groups,
	load_maze_specs,
)
from BFS_solver import solve
from gridworld.baselines import plan_bfs_path
from gridworld.task_spec import TaskSpecification


S_MAZE_TYPE_BY_BASE_NAME = {
	'8x8_corridor': 'S2',
	'10x10_corridor': 'S3',
	'10x10_dense': 'S4',
	'14x14_corridor': 'S5',
	'14x14_dense': 'S6',
}

M_MAZE_EXPECTATIONS = {
	'M1': {
		'mechanisms': ('kr',),
		'barrier_order': ('DR',),
		'interaction_order': ('pickup:kR', 'open:DR'),
		'encounter_order': ('pickup:kR', 'open:DR'),
		'chain_patterns': {'key_door', 'single_key_door'},
	},
	'M2': {
		'mechanisms': ('sg',),
		'barrier_order': ('g1',),
		'interaction_order': ('toggle:s1',),
		'encounter_order': ('toggle:s1', 'gate:g1'),
		'chain_patterns': {'single_switch_gate', 'switch_gate', 'sg'},
	},
	'M3': {
		'mechanisms': ('kr', 'sg'),
		'barrier_order': ('DR', 'g1'),
		'interaction_order': ('pickup:kR', 'open:DR', 'toggle:s1'),
		'encounter_order': ('pickup:kR', 'open:DR', 'toggle:s1', 'gate:g1'),
		'chain_patterns': {'key_door_then_switch_gate', 'kr_sg'},
	},
	'M4': {
		'mechanisms': ('sg', 'kr'),
		'barrier_order': ('g1', 'DR'),
		'interaction_order': ('toggle:s1', 'pickup:kR', 'open:DR'),
		'encounter_order': ('toggle:s1', 'gate:g1', 'pickup:kR', 'open:DR'),
		'chain_patterns': {'switch_gate_then_key_door', 'sg_kr'},
	},
	'M5': {
		'mechanisms': ('kr', 'kb'),
		'barrier_order': ('DR', 'DB'),
		'interaction_order': ('pickup:kR', 'open:DR', 'pickup:kB', 'open:DB'),
		'encounter_order': ('pickup:kR', 'open:DR', 'pickup:kB', 'open:DB'),
		'chain_patterns': {'kk', 'kr_kb'},
	},
	'M6': {
		'mechanisms': ('kr', 'sg', 'kb'),
		'barrier_order': ('DR', 'g1', 'DB'),
		'interaction_order': ('pickup:kR', 'open:DR', 'toggle:s1', 'pickup:kB', 'open:DB'),
		'encounter_order': (
			'pickup:kR',
			'open:DR',
			'toggle:s1',
			'gate:g1',
			'pickup:kB',
			'open:DB',
		),
		'chain_patterns': {'kr_sg_kb'},
	},
}

KEY_BY_TOKEN = {
	'kr': {'id': 'kR', 'color': 'red'},
	'kb': {'id': 'kB', 'color': 'blue'},
}

DOOR_BY_TOKEN = {
	'kr': {'id': 'DR', 'color': 'red', 'requires_key': 'red'},
	'kb': {'id': 'DB', 'color': 'blue', 'requires_key': 'blue'},
}

SWITCH_BY_TOKEN = {
	'sg': {'id': 's1', 'controls': ['g1'], 'switch_type': 'toggle', 'initial_state': 'off'},
}

GATE_BY_TOKEN = {
	'sg': {'id': 'g1', 'initial_state': 'closed'},
}


def _parse_m_name(file_name):
	stem = Path(file_name).stem
	parts = stem.split('_')
	if len(parts) < 4:
		raise ValueError(f'Unexpected M filename: {file_name}')
	size, structure_type, *mechanism_and_variant = parts
	mechanisms = tuple(mechanism_and_variant[:-1])
	variant = mechanism_and_variant[-1]
	return size, structure_type, mechanisms, variant


def _load_s_counterparts():
	counterparts = {}
	for base_name, maze_type in S_MAZE_TYPE_BY_BASE_NAME.items():
		for spec in load_maze_specs(maze_type):
			variant = spec['task_id'].rsplit('_', 1)[-1]
			counterparts[f'{base_name}_{variant}'] = spec
	return counterparts


def _by_id(items):
	return {item['id']: item for item in items}


def _assert_subsequence(test_case, sequence, expected):
	cursor = 0
	for item in sequence:
		if cursor < len(expected) and item == expected[cursor]:
			cursor += 1
	test_case.assertEqual(cursor, len(expected), f'{expected} not in order within {sequence}')


def _planned_encounters(spec):
	task_spec = TaskSpecification.from_dict(spec)
	planned = plan_bfs_path(task_spec)
	if not planned.success:
		return []

	gate_positions = {
		tuple(gate['position']): f"gate:{gate['id']}"
		for gate in spec['mechanisms']['gates']
	}
	encounters = []
	for index, label in enumerate(planned.action_labels):
		if label.startswith('pickup:'):
			encounters.append(label)
		elif label.startswith('open_door:'):
			encounters.append(f"open:{label.split(':', 1)[1]}")
		elif label.startswith('toggle:'):
			encounters.append(label)

		if index + 1 >= len(planned.positions):
			continue
		gate = gate_positions.get(planned.positions[index + 1])
		if gate is not None:
			encounters.append(gate)
	return encounters


def _assert_switch_gate_chain_on_path(test_case, spec, *, switch_id, gate_id):
	mechanisms = spec['mechanisms']
	switch = next(switch for switch in mechanisms['switches'] if switch['id'] == switch_id)
	gate = next(gate for gate in mechanisms['gates'] if gate['id'] == gate_id)

	to_gate = {
		**spec,
		'maze': {
			**spec['maze'],
			'goal': gate['position'],
		},
		'goal': {
			**spec['goal'],
			'target': gate['position'],
		},
	}
	to_gate_result = solve(to_gate)
	test_case.assertTrue(to_gate_result['is_solvable'])
	test_case.assertEqual(to_gate_result['path'][-1], tuple(gate['position']))
	test_case.assertIn(f'toggle:{switch_id}', to_gate_result['interactions'])
	test_case.assertIn(tuple(switch['position']), to_gate_result['path'])

	from_gate = {
		**spec,
		'maze': {
			**spec['maze'],
			'start': gate['position'],
		},
		'mechanisms': {
			**mechanisms,
			'gates': [item for item in mechanisms['gates'] if item['id'] != gate_id],
		},
	}
	from_gate_result = solve(from_gate)
	test_case.assertTrue(from_gate_result['is_solvable'])
	test_case.assertEqual(from_gate_result['path'][0], tuple(gate['position']))
	test_case.assertEqual(from_gate_result['path'][-1], tuple(spec['maze']['goal']))


class TestMMazeTypes(unittest.TestCase):
	"""M mazes should preserve their S maze geometry while adding ordered mechanisms."""

	@classmethod
	def setUpClass(cls):
		cls.specs_by_type = {
			maze_type: load_maze_specs(maze_type, include_file_name=True)
			for maze_type in M_MAZE_EXPECTATIONS
		}
		cls.s_counterparts = _load_s_counterparts()

	def test_expected_number_of_variants(self):
		"""Tests that each M maze type contains the expected ten variants."""
		for maze_type, specs in self.specs_by_type.items():
			with self.subTest(maze_type=maze_type):
				self.assertEqual(len(specs), 10)

	def test_naming_scheme(self):
		"""Tests that each M filename matches its dimensions and mechanism sequence."""
		for maze_type, specs in self.specs_by_type.items():
			expected = M_MAZE_EXPECTATIONS[maze_type]['mechanisms']
			for file_name, spec in specs:
				with self.subTest(maze_type=maze_type, file_name=file_name):
					size, structure_type, mechanisms, variant = _parse_m_name(file_name)
					width, height = spec['maze']['dimensions']

					self.assertEqual(size, f'{width}x{height}')
					self.assertIn(structure_type, {'corridor', 'dense'})
					self.assertEqual(mechanisms, expected)
					self.assertIn(variant, {'0', '1'})

	def test_navigation_contract(self):
		"""Tests that each M maze has valid start and goal navigation fields."""
		for maze_type, specs in self.specs_by_type.items():
			for file_name, spec in specs:
				with self.subTest(maze_type=maze_type, file_name=file_name):
					assert_navigation_contract(self, spec)
					assert_goal_target_matches_maze_goal(self, spec)

	def test_structure_matches_s_maze_counterpart(self):
		"""Tests that M maze walls, start, and goal match the corresponding S maze."""
		for maze_type, specs in self.specs_by_type.items():
			for file_name, spec in specs:
				with self.subTest(maze_type=maze_type, file_name=file_name):
					size, structure_type, _, variant = _parse_m_name(file_name)
					counterpart_key = f'{size}_{structure_type}_{variant}'
					s_spec = self.s_counterparts[counterpart_key]

					self.assertEqual(spec['maze']['dimensions'], s_spec['maze']['dimensions'])
					self.assertEqual(spec['maze']['walls'], s_spec['maze']['walls'])
					self.assertEqual(spec['maze']['start'], s_spec['maze']['start'])
					self.assertEqual(spec['maze']['goal'], s_spec['maze']['goal'])

	def test_bfs_solver_finds_path_to_goal(self):
		"""Tests that each M maze is solvable by the BFS solver."""
		for maze_type, specs in self.specs_by_type.items():
			for file_name, spec in specs:
				with self.subTest(maze_type=maze_type, file_name=file_name):
					assert_bfs_solver_finds_path_to_goal(self, spec)

	def test_has_expected_mechanisms_only(self):
		"""Tests that each M maze has exactly the mechanisms required by its tier."""
		for maze_type, specs in self.specs_by_type.items():
			expected_tokens = M_MAZE_EXPECTATIONS[maze_type]['mechanisms']
			for file_name, spec in specs:
				with self.subTest(maze_type=maze_type, file_name=file_name):
					mechanisms = spec['mechanisms']
					keys = _by_id(mechanisms['keys'])
					doors = _by_id(mechanisms['doors'])
					switches = _by_id(mechanisms['switches'])
					gates = _by_id(mechanisms['gates'])

					self.assertEqual(sum(len(mechanisms[key]) for key in MECHANISM_KEYS), self._expected_mechanism_count(expected_tokens))
					self.assertEqual(set(keys), {KEY_BY_TOKEN[token]['id'] for token in expected_tokens if token in KEY_BY_TOKEN})
					self.assertEqual(set(doors), {DOOR_BY_TOKEN[token]['id'] for token in expected_tokens if token in DOOR_BY_TOKEN})
					self.assertEqual(set(switches), {SWITCH_BY_TOKEN[token]['id'] for token in expected_tokens if token in SWITCH_BY_TOKEN})
					self.assertEqual(set(gates), {GATE_BY_TOKEN[token]['id'] for token in expected_tokens if token in GATE_BY_TOKEN})

					for token in expected_tokens:
						if token in KEY_BY_TOKEN:
							expected_key = KEY_BY_TOKEN[token]
							key = keys[expected_key['id']]
							self.assertEqual(key['color'], expected_key['color'])
						if token in DOOR_BY_TOKEN:
							expected_door = DOOR_BY_TOKEN[token]
							door = doors[expected_door['id']]
							self.assertEqual(door['color'], expected_door['color'])
							self.assertEqual(door['requires_key'], expected_door['requires_key'])
							self.assertEqual(door['initial_state'], 'locked')
						if token in SWITCH_BY_TOKEN:
							expected_switch = SWITCH_BY_TOKEN[token]
							switch = switches[expected_switch['id']]
							self.assertEqual(switch['controls'], expected_switch['controls'])
							self.assertEqual(switch['switch_type'], expected_switch['switch_type'])
							self.assertEqual(switch['initial_state'], expected_switch['initial_state'])
						if token in GATE_BY_TOKEN:
							expected_gate = GATE_BY_TOKEN[token]
							gate = gates[expected_gate['id']]
							self.assertEqual(gate['initial_state'], expected_gate['initial_state'])

	def test_mechanism_chains_are_on_path_in_expected_order(self):
		"""Tests that required mechanism barriers occur on the path in tier order."""
		for maze_type, specs in self.specs_by_type.items():
			expectation = M_MAZE_EXPECTATIONS[maze_type]
			for file_name, spec in specs:
				with self.subTest(maze_type=maze_type, file_name=file_name):
					assert_no_hidden_or_auxiliary_mechanisms(self, spec)
					self.assertIn(spec['metadata']['chain_pattern'], expectation['chain_patterns'])

					result = assert_bfs_solver_finds_path_to_goal(self, spec)
					barrier_path = self._barrier_path(spec, expectation['barrier_order'])
					_assert_subsequence(self, result['path'], barrier_path)
					_assert_subsequence(self, result['interactions'], expectation['interaction_order'])

	def test_bfs_encounters_mechanisms_in_expected_order(self):
		"""Tests the chronological mechanism order encountered by the BFS plan."""
		for maze_type, specs in self.specs_by_type.items():
			expectation = M_MAZE_EXPECTATIONS[maze_type]
			for file_name, spec in specs:
				with self.subTest(maze_type=maze_type, file_name=file_name):
					encounters = _planned_encounters(spec)
					_assert_subsequence(self, encounters, expectation['encounter_order'])

	def test_switch_gate_chains_are_required(self):
		"""Tests that each switch-gate chain is required to reach its gate and continue to goal."""
		for maze_type, specs in self.specs_by_type.items():
			expected_tokens = M_MAZE_EXPECTATIONS[maze_type]['mechanisms']
			for file_name, spec in specs:
				for token in expected_tokens:
					if token not in SWITCH_BY_TOKEN:
						continue
					switch_id = SWITCH_BY_TOKEN[token]['id']
					gate_id = GATE_BY_TOKEN[token]['id']
					with self.subTest(maze_type=maze_type, file_name=file_name, gate_id=gate_id):
						_assert_switch_gate_chain_on_path(self, spec, switch_id=switch_id, gate_id=gate_id)

	def test_has_no_extra_mechanism_groups(self):
		"""Tests that M mechanisms only use the standard mechanism groups."""
		for maze_type, specs in self.specs_by_type.items():
			for file_name, spec in specs:
				with self.subTest(maze_type=maze_type, file_name=file_name):
					assert_standard_mechanism_groups(self, spec)

	@staticmethod
	def _expected_mechanism_count(tokens):
		return sum(2 if token in {'kr', 'kb', 'sg'} else 0 for token in tokens)

	@staticmethod
	def _barrier_path(spec, barrier_order):
		doors = _by_id(spec['mechanisms']['doors'])
		gates = _by_id(spec['mechanisms']['gates'])
		positions = []
		for barrier_id in barrier_order:
			if barrier_id in doors:
				positions.append(tuple(doors[barrier_id]['position']))
			else:
				positions.append(tuple(gates[barrier_id]['position']))
		return positions
