import json
import sys
import unittest
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
	sys.path.insert(0, _REPO_ROOT)

from maze_test_utils import MAZE_JSON_DIR


class TestGoalInvariantAllFamilies(unittest.TestCase):
	"""CLAUDE.md 'one goal source': goal.target must equal maze.goal for every
	maze in the experiment corpus. The per-family suites assert this only for
	D1 and M mazes, which let a defective D3 spec ship undetected."""

	def test_goal_target_matches_maze_goal_everywhere(self):
		maze_files = sorted(MAZE_JSON_DIR.rglob('*.json'))
		self.assertTrue(maze_files, f'no maze JSONs under {MAZE_JSON_DIR}')
		violations = []
		for path in maze_files:
			spec = json.loads(path.read_text(encoding='utf-8'))
			target = spec.get('goal', {}).get('target')
			if target is not None and target != spec['maze']['goal']:
				violations.append(
					f'{path.relative_to(MAZE_JSON_DIR)}: '
					f'goal.target={target} != maze.goal={spec["maze"]["goal"]}'
				)
		self.assertEqual(violations, [])
