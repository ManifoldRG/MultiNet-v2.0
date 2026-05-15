"""BFS test helper for experimental maze JSON specs."""

from collections import deque


def _point(value):
	return tuple(value)


def _mechanism_maps(spec):
	mechanisms = spec.get('mechanisms', {})
	keys_by_position = {
		_point(key['position']): key
		for key in mechanisms.get('keys', [])
	}
	doors_by_position = {
		_point(door['position']): door
		for door in mechanisms.get('doors', [])
	}
	return keys_by_position, doors_by_position


def _neighbors(point):
	x, y = point
	return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _in_bounds(point, width, height):
	x, y = point
	return 0 <= x < width and 0 <= y < height


def solve(spec):
	"""Return a shortest path result for a maze JSON spec."""
	maze = spec['maze']
	width, height = maze['dimensions']
	start = _point(maze['start'])
	goal = _point(maze['goal'])
	walls = {_point(wall) for wall in maze.get('walls', [])}
	keys_by_position, doors_by_position = _mechanism_maps(spec)

	start_state = (start, frozenset(), frozenset())
	queue = deque([(start_state, [start], [])])
	visited = {start_state}

	while queue:
		(position, held_keys, opened_doors), path, interactions = queue.popleft()
		if position == goal:
			return {
				'is_solvable': True,
				'path': path,
				'interactions': interactions,
				'optimal_cost': len(path) - 1,
			}

		for next_position in _neighbors(position):
			if (
				not _in_bounds(next_position, width, height)
				or next_position in walls
			):
				continue

			next_keys = held_keys
			next_opened_doors = opened_doors
			next_interactions = interactions

			key = keys_by_position.get(next_position)
			if key is not None and key['id'] not in held_keys:
				next_keys = frozenset((*held_keys, key['id']))
				next_interactions = [*interactions, f"pickup:{key['id']}"]

			door = doors_by_position.get(next_position)
			if door is not None and door['id'] not in opened_doors:
				required_key = door.get('requires_key', door.get('color'))
				has_required_key = any(
					key['color'] == required_key and key['id'] in next_keys
					for key in keys_by_position.values()
				)
				if not has_required_key:
					continue
				next_opened_doors = frozenset((*opened_doors, door['id']))
				next_interactions = [*next_interactions, f"open:{door['id']}"]

			next_state = (next_position, next_keys, next_opened_doors)
			if next_state in visited:
				continue

			visited.add(next_state)
			queue.append((next_state, [*path, next_position], next_interactions))

	return {
		'is_solvable': False,
		'path': [],
		'interactions': [],
		'optimal_cost': None,
	}


def find_all_paths(spec):
	"""Return the BFS solver path in the legacy list-of-paths test-helper shape."""
	result = solve(spec)
	if not result['is_solvable']:
		return []
	return [result['path']]
