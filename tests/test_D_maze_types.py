import sys
import unittest
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
	sys.path.insert(0, _REPO_ROOT)

from maze_test_utils import (
	MAZE_JSON_DIR,
	assert_goal_target_matches_maze_goal,
	assert_navigation_contract,
	assert_no_hidden_or_auxiliary_mechanisms,
	load_maze_specs,
)


MECHANISM_TOKENS_TO_M_TYPE = {
	('kr',): 'M1',
	('sg',): 'M2',
	('kr', 'sg'): 'M3',
	('sg', 'kr'): 'M4',
	('kr', 'kb'): 'M5',
	('kr', 'sg', 'kb'): 'M6',
}


def _parse_d1_name(file_name):
	stem = Path(file_name).stem
	parts = stem.split('_')
	if len(parts) < 6:
		raise ValueError(f'Unexpected D1 filename: {file_name}')
	size, structure_type, wrong, key_token, *mechanism_and_variant = parts
	if (wrong, key_token) != ('wrong', 'ky'):
		raise ValueError(f'Expected wrong_ky marker in D1 filename: {file_name}')
	mechanisms = tuple(mechanism_and_variant[:-1])
	variant = mechanism_and_variant[-1]
	return size, structure_type, mechanisms, variant


def _m_counterpart_name(file_name):
	size, structure_type, mechanisms, variant = _parse_d1_name(file_name)
	return f"{size}_{structure_type}_{'_'.join(mechanisms)}_{variant}.json"


def _m_counterpart_type(file_name):
	_, _, mechanisms, _ = _parse_d1_name(file_name)
	return MECHANISM_TOKENS_TO_M_TYPE[mechanisms]


def _by_counterpart_name(specs_by_type):
	counterparts = {}
	for maze_type, specs in specs_by_type.items():
		for file_name, spec in specs:
			counterparts[(maze_type, file_name)] = spec
	return counterparts


def _yellow_keys(spec):
	return [
		key
		for key in spec['mechanisms']['keys']
		if key['id'] == 'kY' or key['color'] == 'yellow'
	]


def _keys_without_yellow(spec):
	return [
		key
		for key in spec['mechanisms']['keys']
		if key['id'] != 'kY' and key['color'] != 'yellow'
	]


def _load_all_d_maze_specs():
	specs = []
	for maze_dir in sorted(MAZE_JSON_DIR.glob('D*')):
		if maze_dir.is_dir():
			specs.extend(load_maze_specs(maze_dir.name, include_file_name=True))
	return specs


class TestD1MazeTypes(unittest.TestCase):
	"""D1 mazes should be M mazes with exactly one distractor yellow key."""

	@classmethod
	def setUpClass(cls):
		cls.specs = load_maze_specs('D1', include_file_name=True)
		cls.m_counterparts = _by_counterpart_name(
			{
				maze_type: load_maze_specs(maze_type, include_file_name=True)
				for maze_type in MECHANISM_TOKENS_TO_M_TYPE.values()
			}
		)

	def test_expected_number_of_variants(self):
		"""Tests that D1 has one counterpart for every M maze."""
		self.assertEqual(len(self.specs), 60)
		self.assertEqual(len(self.m_counterparts), 60)

	def test_naming_scheme_maps_to_m_counterpart(self):
		"""Tests that removing wrong_ky from the D1 filename identifies an M maze."""
		for file_name, _ in self.specs:
			with self.subTest(file_name=file_name):
				maze_type = _m_counterpart_type(file_name)
				counterpart_name = _m_counterpart_name(file_name)
				self.assertIn((maze_type, counterpart_name), self.m_counterparts)

	def test_navigation_contract(self):
		"""Tests that each D1 maze has valid start and goal navigation fields."""
		for file_name, spec in self.specs:
			with self.subTest(file_name=file_name):
				assert_navigation_contract(self, spec)
				assert_goal_target_matches_maze_goal(self, spec)

	def test_has_one_and_only_one_yellow_key_distractor(self):
		"""Tests that each D1 maze adds exactly one yellow key and no yellow door."""
		for file_name, spec in self.specs:
			with self.subTest(file_name=file_name):
				yellow_keys = _yellow_keys(spec)
				self.assertEqual(
					yellow_keys,
					[
						{
							'id': 'kY',
							'position': yellow_keys[0]['position'],
							'color': 'yellow',
						}
					],
				)
				self.assertNotIn(tuple(yellow_keys[0]['position']), {tuple(wall) for wall in spec['maze']['walls']})
				self.assertNotEqual(yellow_keys[0]['position'], spec['maze']['start'])
				self.assertNotEqual(yellow_keys[0]['position'], spec['maze']['goal'])
				self.assertFalse(
					[
						door
						for door in spec['mechanisms']['doors']
						if door.get('requires_key') == 'yellow' or door.get('color') == 'yellow'
					]
				)

	def test_matches_m_counterpart_except_yellow_key(self):
		"""Tests that D1 gameplay fields match the M counterpart after removing kY."""
		for file_name, d_spec in self.specs:
			with self.subTest(file_name=file_name):
				maze_type = _m_counterpart_type(file_name)
				counterpart_name = _m_counterpart_name(file_name)
				m_spec = self.m_counterparts[(maze_type, counterpart_name)]

				self.assertEqual(d_spec['version'], m_spec['version'])
				self.assertEqual(d_spec['seed'], m_spec['seed'])
				self.assertEqual(d_spec['difficulty_tier'], m_spec['difficulty_tier'])
				self.assertEqual(d_spec['maze'], m_spec['maze'])
				self.assertEqual(d_spec['rules'], m_spec['rules'])
				self.assertEqual(d_spec['goal'], m_spec['goal'])
				self.assertEqual(d_spec['max_steps'], m_spec['max_steps'])
				self.assertEqual(
					{
						key: value
						for key, value in d_spec['metadata'].items()
						if key != 'chain_pattern'
					},
					{
						key: value
						for key, value in m_spec['metadata'].items()
						if key != 'chain_pattern'
					},
				)

				self.assertEqual(_keys_without_yellow(d_spec), m_spec['mechanisms']['keys'])
				self.assertEqual(d_spec['mechanisms']['doors'], m_spec['mechanisms']['doors'])
				self.assertEqual(d_spec['mechanisms']['switches'], m_spec['mechanisms']['switches'])
				self.assertEqual(d_spec['mechanisms']['gates'], m_spec['mechanisms']['gates'])

	def test_metadata_and_task_id_mark_wrong_key_variant(self):
		"""Tests that D1-specific metadata is limited to the wrong-key overlay."""
		for file_name, spec in self.specs:
			with self.subTest(file_name=file_name):
				counterpart_name = _m_counterpart_name(file_name)
				self.assertEqual(spec['task_id'], Path(file_name).stem)
				self.assertEqual(counterpart_name, f"{spec['task_id'].replace('_wrong_ky', '')}.json")
				self.assertTrue(spec['metadata']['chain_pattern'].startswith('wrong_key_then_'))
				assert_no_hidden_or_auxiliary_mechanisms(self, spec)


class TestDMazeDistractors(unittest.TestCase):
	"""D maze filenames should match their distractor mechanisms."""

	@classmethod
	def setUpClass(cls):
		cls.specs = _load_all_d_maze_specs()

	def test_inactive_sb_files_have_inactive_switch(self):
		"""Tests that inactive_sb maze names include an inactive switch mechanism."""
		for file_name, spec in self.specs:
			if 'inactive_sb' not in file_name:
				continue
			with self.subTest(file_name=file_name):
				inactive_switches = [
					switch
					for switch in spec['mechanisms']['switches']
					if switch['id'] == 'inactive_sb'
				]
				self.assertEqual(len(inactive_switches), 1)
				self.assertEqual(inactive_switches[0]['color'], 'blue')
				self.assertEqual(inactive_switches[0]['controls'], [])
				self.assertEqual(inactive_switches[0]['initial_state'], 'off')
