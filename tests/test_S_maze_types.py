import unittest

from maze_test_utils import (
	assert_bfs_solver_finds_path_to_goal,
	assert_navigation_contract,
	assert_no_mechanisms,
	load_maze_specs,
)


class SimpleMazeAssertions:
	maze_type = None
	expected_dimensions = None
	expected_difficulty_tier = None
	expected_wall_topology = None

	@classmethod
	def setUpClass(cls):
		cls.specs = load_maze_specs(cls.maze_type)

	def test_expected_number_of_variants(self):
		"""Tests that each simple maze type has exactly two JSON variants."""
		self.assertEqual(len(self.specs), 2)

	def test_navigation_contract(self):
		"""Tests that each simple maze has valid start and goal navigation fields."""
		for spec in self.specs:
			with self.subTest(task_id=spec['task_id']):
				assert_navigation_contract(self, spec)

	def test_has_no_mechanisms(self):
		"""Tests that simple mazes do not define keys, doors, switches, or gates."""
		for spec in self.specs:
			with self.subTest(task_id=spec['task_id']):
				assert_no_mechanisms(self, spec)

	def test_matches_maze_type_metadata(self):
		"""Tests that each simple maze matches its expected dimensions and metadata."""
		for spec in self.specs:
			with self.subTest(task_id=spec['task_id']):
				self.assertEqual(spec['maze']['dimensions'], self.expected_dimensions)
				self.assertEqual(spec['difficulty_tier'], self.expected_difficulty_tier)
				self.assertEqual(spec['metadata']['wall_topology'], self.expected_wall_topology)

	def test_bfs_solver_finds_path_to_goal(self):
		"""Tests that each simple maze is solvable by the BFS solver."""
		for spec in self.specs:
			with self.subTest(task_id=spec['task_id']):
				assert_bfs_solver_finds_path_to_goal(self, spec)


class TestS1DSMazes(SimpleMazeAssertions, unittest.TestCase):
	maze_type = 'S1'
	expected_dimensions = [8, 8]
	expected_difficulty_tier = 1
	expected_wall_topology = 'open'

	def test_has_no_interior_walls(self):
		"""Tests that S1 open-room mazes have no interior wall cells."""
		for spec in self.specs:
			with self.subTest(task_id=spec['task_id']):
				self.assertEqual(spec['maze']['walls'], [])


class TestS2SmallCorridorMazes(SimpleMazeAssertions, unittest.TestCase):
	maze_type = 'S2'
	expected_dimensions = [8, 8]
	expected_difficulty_tier = 2
	expected_wall_topology = 'winding'

	def test_has_corridor_walls(self):
		"""Tests that S2 small corridor mazes define interior wall cells."""
		for spec in self.specs:
			with self.subTest(task_id=spec['task_id']):
				self.assertGreater(len(spec['maze']['walls']), 0)


class TestS3MediumCorridorMazes(SimpleMazeAssertions, unittest.TestCase):
	maze_type = 'S3'
	expected_dimensions = [10, 10]
	expected_difficulty_tier = 3
	expected_wall_topology = 'winding'

	def test_has_corridor_walls(self):
		"""Tests that S3 medium corridor mazes define interior wall cells."""
		for spec in self.specs:
			with self.subTest(task_id=spec['task_id']):
				self.assertGreater(len(spec['maze']['walls']), 0)


class TestS4MediumDenseMazes(SimpleMazeAssertions, unittest.TestCase):
	maze_type = 'S4'
	expected_dimensions = [10, 10]
	expected_difficulty_tier = 4
	expected_wall_topology = 'dense_dead_ends'

	def test_has_dense_walls(self):
		"""Tests that S4 medium dense mazes have the expected dense wall count."""
		for spec in self.specs:
			with self.subTest(task_id=spec['task_id']):
				self.assertGreaterEqual(len(spec['maze']['walls']), 30)


class TestS5LargeCorridorMazes(SimpleMazeAssertions, unittest.TestCase):
	maze_type = 'S5'
	expected_dimensions = [14, 14]
	expected_difficulty_tier = 5
	expected_wall_topology = 'winding'

	def test_has_corridor_walls(self):
		"""Tests that S5 large corridor mazes define interior wall cells."""
		for spec in self.specs:
			with self.subTest(task_id=spec['task_id']):
				self.assertGreater(len(spec['maze']['walls']), 0)


class TestS6LargeDenseMazes(SimpleMazeAssertions, unittest.TestCase):
	maze_type = 'S6'
	expected_dimensions = [14, 14]
	expected_difficulty_tier = 6
	expected_wall_topology = 'dense_dead_ends'

	def test_has_dense_walls(self):
		"""Tests that S6 large dense mazes have the expected dense wall count."""
		for spec in self.specs:
			with self.subTest(task_id=spec['task_id']):
				self.assertGreaterEqual(len(spec['maze']['walls']), 60)
