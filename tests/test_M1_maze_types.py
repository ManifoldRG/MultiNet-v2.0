import unittest
from pathlib import Path

from maze_test_utils import (
	MECHANISM_KEYS,
	assert_bfs_solver_finds_path_to_goal,
	assert_goal_target_matches_maze_goal,
	assert_key_door_chain_on_path,
	assert_navigation_contract,
	assert_no_hidden_or_auxiliary_mechanisms,
	assert_standard_mechanism_groups,
	load_maze_specs,
)


S_MAZE_TYPE_BY_BASE_NAME = {
	'8x8_corridor': 'S2',
	'10x10_corridor': 'S3',
	'10x10_dense': 'S4',
	'14x14_corridor': 'S5',
	'14x14_dense': 'S6',
}


def _parse_m1_name(file_name):
	stem = Path(file_name).stem
	parts = stem.split('_')
	if len(parts) < 4:
		raise ValueError(f'Unexpected M1 filename: {file_name}')
	size, structure_type, mechanism, variant = parts
	return size, structure_type, mechanism, variant


def _load_s_counterparts():
	counterparts = {}
	for base_name, maze_type in S_MAZE_TYPE_BY_BASE_NAME.items():
		for spec in load_maze_specs(maze_type):
			variant = spec['task_id'].rsplit('_', 1)[-1]
			counterparts[f'{base_name}_{variant}'] = spec
	return counterparts


class TestM1MazeTypes(unittest.TestCase):
	"""In addition to passing the S maze tests, M1 mazes should also have one and only one mechanism chain.

	The chain must be on the path to reaching the goal. i.e. it's impossible to solve the maze without going through the mechanism chain.
	"""

	@classmethod
	def setUpClass(cls):
		cls.specs = load_maze_specs('M1', include_file_name=True)
		cls.s_counterparts = _load_s_counterparts()

	def test_expected_number_of_variants(self):
		"""Tests that M1 contains the expected number of maze variants."""
		self.assertEqual(len(self.specs), 10)

	def test_naming_scheme(self):
		"""Tests that each M1 filename matches its dimensions and kr mechanism type."""
		for file_name, spec in self.specs:
			with self.subTest(file_name=file_name):
				size, structure_type, mechanism, variant = _parse_m1_name(file_name)
				width, height = spec['maze']['dimensions']

				self.assertEqual(size, f'{width}x{height}')
				self.assertIn(structure_type, {'corridor', 'dense'})
				self.assertEqual(mechanism, 'kr')
				self.assertIn(variant, {'0', '1'})

	def test_navigation_contract(self):
		"""Tests that each M1 maze has valid start and goal navigation fields."""
		for file_name, spec in self.specs:
			with self.subTest(file_name=file_name):
				assert_navigation_contract(self, spec)
				assert_goal_target_matches_maze_goal(self, spec)

	def test_structure_matches_s_maze_counterpart(self):
		"""Tests that M1 walls, start, and goal match the corresponding S maze."""
		for file_name, spec in self.specs:
			with self.subTest(file_name=file_name):
				size, structure_type, _, variant = _parse_m1_name(file_name)
				counterpart_key = f'{size}_{structure_type}_{variant}'
				s_spec = self.s_counterparts[counterpart_key]

				self.assertEqual(spec['maze']['dimensions'], s_spec['maze']['dimensions'])
				self.assertEqual(spec['maze']['walls'], s_spec['maze']['walls'])
				self.assertEqual(spec['maze']['start'], s_spec['maze']['start'])
				self.assertEqual(spec['maze']['goal'], s_spec['maze']['goal'])

	def test_bfs_solver_finds_path_to_goal(self):
		"""Tests that each M1 maze is solvable by the BFS solver."""
		for file_name, spec in self.specs:
			with self.subTest(file_name=file_name):
				assert_bfs_solver_finds_path_to_goal(self, spec)

	def test_has_one_red_key_and_one_red_door(self):
		"""Tests that each M1 maze has exactly one red key and one red door."""
		for file_name, spec in self.specs:
			with self.subTest(file_name=file_name):
				mechanisms = spec['mechanisms']
				self.assertEqual(mechanisms['switches'], [])
				self.assertEqual(mechanisms['gates'], [])
				self.assertEqual(len(mechanisms['keys']), 1)
				self.assertEqual(len(mechanisms['doors']), 1)

				self.assertEqual(
					mechanisms['keys'],
					[
						{
							'id': 'kR',
							'position': mechanisms['keys'][0]['position'],
							'color': 'red',
						}
					],
				)
				self.assertEqual(
					mechanisms['doors'],
					[
						{
							'id': 'DR',
							'position': mechanisms['doors'][0]['position'],
							'color': 'red',
							'requires_key': 'red',
							'initial_state': 'locked',
						}
					],
				)

	def test_has_one_and_only_one_mechanism_chain(self):
		"""Tests that M1 mazes contain one key-door chain with the door on a valid path to the goal."""
		for file_name, spec in self.specs:
			with self.subTest(file_name=file_name):
				mechanisms = spec['mechanisms']
				key = mechanisms['keys'][0]
				door = mechanisms['doors'][0]
				chain = [key['id'], door['id']]

				self.assertEqual(chain, ['kR', 'DR'])
				self.assertIn(spec['metadata']['chain_pattern'], {'key_door', 'single_key_door'})
				assert_no_hidden_or_auxiliary_mechanisms(self, spec)
				self.assertEqual(sum(len(mechanisms[key]) for key in MECHANISM_KEYS), 2)
				assert_key_door_chain_on_path(self, spec, key_id=key['id'], door_id=door['id'])

	def test_has_no_extra_mechanism_groups(self):
		"""Tests that M1 mechanisms only use the standard mechanism groups."""
		for file_name, spec in self.specs:
			with self.subTest(file_name=file_name):
				assert_standard_mechanism_groups(self, spec)
