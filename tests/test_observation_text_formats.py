"""json/ascii alternatives to the default coordinate-prose observation."""

from __future__ import annotations

import dataclasses
import json

import pytest

from interface.config import ExperimentConfig
from interface.coords import agent_row_col, goal_row_col, maze_rows_cols, wall_cells
from interface.loader import load_task
from interface.observation import current_observation_text
from interface.renderer import render_initial_maze_text, render_user_observation_text

MAZE = "gridworld/tasks/tier3/complex_deps_003.json"


@pytest.fixture(scope="module")
def spec_state():
    backend, spec = load_task(MAZE)
    _, state, _ = backend.reset(seed=spec.seed)
    return spec, state


@pytest.fixture(scope="module")
def mid(spec_state):
    spec, state = spec_state
    return spec, dataclasses.replace(
        state,
        agent_position=(5, 4),
        agent_direction=1,
        agent_carrying="red",
        collected_keys={"key_red"},
        open_doors={"door_blue"},
        open_gates={"gate_final"},
        active_switches={"switch_main"},
    )


def _grid(text):
    lines = text.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith("Map.")) + 1
    end = next(i for i, line in enumerate(lines) if line.startswith("Legend:"))
    return [line.split() for line in lines[start:end] if line.strip()]


def _legend(text):
    lines = text.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith("Legend:")) + 1
    end = next(i for i, line in enumerate(lines) if line.startswith("Status:"))
    out = {}
    for line in lines[start:end]:
        token, _, desc = line.strip().partition(" = ")
        if token:
            out[token] = desc
    return out


def _status(text):
    lines = text.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith("Status:")) + 1
    out = {}
    for line in lines[start:]:
        label, _, value = line.strip().partition(": ")
        if label:
            out[label] = value
    return out


def _ascii(spec, state=None, include_facing=True):
    if state is None:
        return render_initial_maze_text(
            spec, observation_text_format="ascii", include_facing=include_facing
        )
    return render_user_observation_text(
        spec, state, include_facing=include_facing, observation_text_format="ascii"
    )


def _json(spec, state, **kw):
    return json.loads(render_user_observation_text(
        spec, state, include_facing=True, observation_text_format="json", **kw
    ))


def test_default_format_is_coords():
    assert ExperimentConfig().observation_text_format == "coords"


def test_coords_default_and_collected_key(spec_state, mid):
    spec, start = spec_state
    _, state = mid
    assert render_user_observation_text(spec, start, include_facing=True) == (
        render_user_observation_text(
            spec, start, include_facing=True, observation_text_format="coords"
        )
    )
    assert "Keys collected" not in render_user_observation_text(spec, start, include_facing=True)
    coords = render_user_observation_text(spec, state, include_facing=True)
    assert "Your inventory: red." in coords
    assert "Keys collected" not in coords
    assert "Keys used up" not in coords
    spent = dataclasses.replace(state, agent_carrying=None)
    spent_coords = render_user_observation_text(spec, spent, include_facing=True)
    assert "Keys used up: red." in spent_coords
    assert "Your inventory: empty." in spent_coords
    assert {k["color"]: k["status"] for k in _json(spec, state)["map_contents"]["keys"]}["red"] == "carried"
    assert _status(_ascii(spec, state))["Carrying"] == "red key"


def test_ascii_initial_and_current(spec_state, mid):
    spec, start = spec_state
    _, state = mid
    rows, cols = maze_rows_cols(spec)
    text = _ascii(spec, start)
    grid = _grid(text)
    legend = _legend(text)
    assert text.startswith("Map.")
    assert len(grid) == rows and {len(r) for r in grid} == {cols}
    for r, c in wall_cells(spec):
        assert grid[r - 1][c - 1] == "#"
    gr, gc = goal_row_col(spec)
    assert grid[gr - 1][gc - 1] == "G"
    assert grid[7][1] == "kB" and grid[4][4] == "kR"
    assert grid[2][3] == "dB" and grid[3][6] == "dR"
    assert grid[7][7] == "s1" and grid[4][9] == "g1"
    assert legend["kB"] == "blue key"
    assert legend["s1"] == "closed switch"
    assert legend["g1"] == "closed gate"
    assert "dO" not in legend and "gO" not in legend
    assert _status(text) == {
        "Carrying": "nothing",
        "Moves remaining": str(start.max_steps - start.step_count),
        "Switches on": "none",
    }

    init = _ascii(spec)
    assert _grid(init)[0][0] == ">"
    assert _legend(init)[">"] == "you, facing EAST"
    assert "S" not in _legend(init)
    assert init.startswith("Coordinates are ")
    assert "The following cells are walls" not in init

    mid_text = _ascii(spec, state)
    g, lg = _grid(mid_text), _legend(mid_text)
    assert g[2][3] == "dO" and g[4][9] == "g1" and g[3][6] == "dR"
    assert g[3][4] == "v" and g[4][4] == "." and g[7][1] == "kB"
    assert lg["dO"] == "unlocked door"
    assert lg["s1"] == "open switch"
    assert lg["g1"] == "open gate"
    assert _grid(_ascii(spec, state, include_facing=False))[3][4] == "A"
    assert _status(mid_text) == {
        "Carrying": "red key",
        "Moves remaining": str(state.max_steps - state.step_count),
        "Switches on": "s1",
    }
    assert "Doors open" not in _status(mid_text)
    spent = dataclasses.replace(state, agent_carrying=None)
    assert _status(_ascii(spec, spent))["Keys used up"] == "red"
    standing = dataclasses.replace(start, agent_position=(2, 8))
    standing_text = _ascii(spec, standing)
    assert _grid(standing_text)[7][1] == ">"
    assert _legend(standing_text)[">"] == "you, facing EAST"
    assert _legend(standing_text)["kB"] == "blue key (you are standing on it)"


def test_json_payload(spec_state, mid):
    spec, start = spec_state
    _, state = mid
    payload = _json(spec, start)
    row, col = agent_row_col(start)
    assert payload["agent"] == {"row": row, "col": col, "facing": "EAST"}
    assert payload["inventory"] == []
    assert payload["moves_remaining"] == start.max_steps - start.step_count
    assert "stall" not in payload
    assert {k["color"] for k in payload["map_contents"]["keys"]} == {"blue", "red"}

    contents = _json(spec, state)["map_contents"]
    assert [d["status"] for d in contents["doors"]] == ["open", "locked"]
    assert contents["gates"][0]["status"] == "open"
    assert contents["gates"][0]["initial_status"] == "closed"
    assert contents["switches"][0]["status"] == "on"
    by_color = {k["color"]: k for k in contents["keys"]}
    assert by_color["blue"] == {"color": "blue", "status": "on_ground", "row": 8, "col": 2}
    assert by_color["red"] == {"color": "red", "status": "carried"}
    assert _json(spec, state)["inventory"] == ["red"]

    text = render_user_observation_text(
        spec, state, include_facing=True, observation_text_format="json"
    )
    assert '"status"' in text and '"state"' not in text and '"initial_state"' not in text
    spent = dataclasses.replace(state, agent_carrying=None)
    assert {k["color"]: k["status"] for k in _json(spec, spent)["map_contents"]["keys"]} == {
        "blue": "on_ground",
        "red": "used",
    }
    dropped = dataclasses.replace(
        state, agent_carrying=None, collected_keys=set(), key_positions={"key_red": (5, 6)}
    )
    red = next(k for k in _json(spec, dropped)["map_contents"]["keys"] if k["color"] == "red")
    assert red == {"color": "red", "status": "dropped", "row": 6, "col": 5}
    assert "facing" not in json.loads(render_user_observation_text(
        spec, start, include_facing=False, observation_text_format="json"
    ))["agent"]

    init = render_initial_maze_text(spec, observation_text_format="json")
    init_payload = json.loads(init)
    rows, cols = maze_rows_cols(spec)
    assert init_payload["world"] == {"rows": rows, "cols": cols}
    assert init_payload["goal"] == list(goal_row_col(spec))
    assert {tuple(c) for c in init_payload["walls"]} == wall_cells(spec)
    assert '"start": [1, 1]' in init
    walls_line = next(line for line in init.split("\n") if '"walls"' in line)
    assert walls_line.rstrip().endswith("],")


def test_observation_wiring(spec_state):
    spec, state = spec_state
    text = current_observation_text(
        "text_only", spec, state, include_description=True, include_facing=True,
        observation_text_format="json",
    )
    assert json.loads(text)["agent"]["facing"] == "EAST"
    for fmt in ("coords", "json", "ascii"):
        assert current_observation_text(
            "image_only", spec, state, include_description=True, include_facing=True,
            observation_text_format=fmt,
        ) == ""
