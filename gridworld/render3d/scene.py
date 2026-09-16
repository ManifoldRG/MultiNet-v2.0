"""Task spec -> MuJoCo scene (MJCF string) + an index of state-dependent geoms.

Built once per configure(); per-frame state changes only move geoms and mocap
bodies (sync.py), never rebuild the model. The wall set mirrors MiniGrid's
grid construction: border ring + spec walls, minus any cell an object is
placed on (objects overwrite walls there).
"""

from __future__ import annotations

from dataclasses import dataclass
from xml.sax.saxutils import quoteattr

from gridworld.task_spec import TaskSpecification

from . import palette
from .cameras import cell_center

KEY_HEIGHT = 0.10
CARRIED_KEY_HEIGHT = 0.62
AGENT_GROUP = 3  # agent geoms; hidden in first_person via MjvOption.geomgroup
HIDDEN_GROUP = 5  # geoms of the inactive state; the renderer never draws this group
KEY_PARTS = ("bow", "shaft", "tooth1", "tooth2")


class UnsupportedSpecError(ValueError):
    """The spec uses a feature the 3D renderer cannot draw faithfully."""


def check_supported(spec: TaskSpecification) -> None:
    problems = []
    mech = spec.mechanisms
    if mech.blocks:
        problems.append("blocks")
    if mech.hazards:
        problems.append("hazards")
    if mech.teleporters:
        problems.append("teleporters")
    if spec.goal.goal_type != "reach_position":
        problems.append(f"goal_type={spec.goal.goal_type!r}")
    if spec.rules.observability != "full":
        problems.append(f"observability={spec.rules.observability!r}")

    # Colours are pure-Python (no mujoco) and checked here too, so a bad
    # colour is rejected before configure() touches the state backend or
    # closes the current renderer -- the same completeness the other checks
    # give (build_scene would otherwise raise mid-build via palette.rgba).
    colour_names = (
        [key.color for key in mech.keys]
        + [door.requires_key for door in mech.doors]
        + [switch.color for switch in mech.switches]
        + [gate.color for gate in mech.gates]
    )
    bad_colours: list[str] = []
    for name in colour_names:
        try:
            palette.rgba(name)
        except ValueError:
            if name not in bad_colours:
                bad_colours.append(name)
    problems.extend(f"colour {name!r}" for name in bad_colours)

    if problems:
        raise UnsupportedSpecError(
            f"3D renderer does not support {', '.join(problems)} (task {spec.task_id!r})"
        )


# wall_cells() and _across_is_x() below encode MiniGrid-derived placement
# assumptions: objects overwrite the wall at their cell, and doors/gates/
# switches sit in 1-wide passages (so "the wall on one side" determines the
# barrier's orientation). A future non-MiniGrid state engine must preserve
# these assumptions -- or these two functions must be revisited alongside it.
def wall_cells(spec: TaskSpecification) -> frozenset[tuple[int, int]]:
    width, height = spec.maze.dimensions
    border = {
        (x, y)
        for x in range(width)
        for y in range(height)
        if x in (0, width - 1) or y in (0, height - 1)
    }
    walls = border | {(p.x, p.y) for p in spec.maze.walls}
    mech = spec.mechanisms
    occupied = {tuple(spec.resolved_goal())}
    for group in (mech.keys, mech.doors, mech.gates, mech.switches):
        occupied |= {(m.position.x, m.position.y) for m in group}
    return frozenset(walls - occupied)


@dataclass(frozen=True)
class SceneIndex:
    width: int
    height: int
    wall_height: float
    wall_cells: frozenset
    door_closed: dict[str, tuple[str, ...]]
    door_open: dict[str, tuple[str, ...]]
    switch_off: dict[str, tuple[str, ...]]
    switch_on: dict[str, tuple[str, ...]]
    gate_closed: dict[str, tuple[str, ...]]
    gate_open: dict[str, tuple[str, ...]]
    key_bodies: dict[str, str]
    carried: dict[str, tuple[str, ...]]
    agent_body: str = "agent"


def _fmt(values) -> str:
    return " ".join(f"{float(v):.5g}" for v in values)


def _geom(name, kind, pos, size, rgba, *, group: int = 0) -> str:
    # contype/conaffinity 0: purely visual; physics is never stepped.
    return (
        f'<geom name={quoteattr(name)} type="{kind}" pos="{_fmt(pos)}" size="{_fmt(size)}" '
        f'rgba="{_fmt(rgba)}" group="{group}" contype="0" conaffinity="0"/>'
    )


def _key_parts(prefix, rgba, *, scale: float = 1.0, z: float = 0.0, group: int = 0) -> list[str]:
    s = scale
    return [
        _geom(f"{prefix}:bow", "cylinder", (-0.14 * s, 0, z), (0.11 * s, 0.02 * s), rgba, group=group),
        _geom(f"{prefix}:shaft", "box", (0.05 * s, 0, z), (0.15 * s, 0.03 * s, 0.02 * s), rgba, group=group),
        _geom(f"{prefix}:tooth1", "box", (0.16 * s, -0.06 * s, z), (0.025 * s, 0.04 * s, 0.02 * s), rgba, group=group),
        _geom(f"{prefix}:tooth2", "box", (0.08 * s, -0.06 * s, z), (0.025 * s, 0.04 * s, 0.02 * s), rgba, group=group),
    ]


# See the MiniGrid-derived placement note above wall_cells(): this also
# assumes a 1-wide passage (exactly one axis has walls on both sides).
def _across_is_x(cell, walls) -> bool:
    """True when the passage through ``cell`` runs north-south (walls east and
    west), so a barrier spans the x axis; otherwise it spans y."""
    x, y = cell
    return (x - 1, y) in walls and (x + 1, y) in walls


def _size(across_x: bool, across: float, along: float, half_z: float) -> tuple[float, float, float]:
    return (across, along, half_z) if across_x else (along, across, half_z)


def _offset(across_x: bool, d: float) -> tuple[float, float]:
    return (d, 0.0) if across_x else (0.0, d)


def build_scene(spec: TaskSpecification, *, wall_height: float, resolution: int) -> tuple[str, SceneIndex]:
    check_supported(spec)
    width, height = spec.maze.dimensions
    walls = wall_cells(spec)
    mech = spec.mechanisms
    hz = wall_height / 2.0
    world: list[str] = []

    # Floor: a grid-line base under one inset tile per non-wall cell.
    world.append(_geom("floor:base", "box", (width / 2, -height / 2, -0.01), (width / 2, height / 2, 0.01), palette.GRID_LINE))
    for x in range(width):
        for y in range(height):
            cx, cy, _ = cell_center(x, y)
            if (x, y) in walls:
                world.append(_geom(f"wall:{x}:{y}", "box", (cx, cy, hz), (0.5, 0.5, hz), palette.WALL))
            else:
                world.append(_geom(f"floor:{x}:{y}", "box", (cx, cy, 0.0), (0.46, 0.46, 0.005), palette.FLOOR))
    gx, gy, _ = cell_center(*spec.resolved_goal())
    world.append(_geom("goal", "box", (gx, gy, 0.012), (0.40, 0.40, 0.012), palette.GOAL))

    door_closed: dict[str, tuple[str, ...]] = {}
    door_open: dict[str, tuple[str, ...]] = {}
    for door in mech.doors:
        cell = (door.position.x, door.position.y)
        cx, cy, _ = cell_center(*cell)
        across_x = _across_is_x(cell, walls)
        colour = palette.rgba(door.requires_key)
        slab = f"door:{door.id}:slab"
        world.append(_geom(slab, "box", (cx, cy, hz * 0.9), _size(across_x, 0.47, 0.06, hz * 0.9), colour))
        posts = []
        for i, d in enumerate((-0.41, 0.41)):
            name = f"door:{door.id}:post{i}"
            ox, oy = _offset(across_x, d)
            world.append(_geom(name, "box", (cx + ox, cy + oy, hz * 0.9), _size(across_x, 0.06, 0.06, hz * 0.9), colour))
            posts.append(name)
        door_closed[door.id] = (slab,)
        door_open[door.id] = tuple(posts)

    gate_closed: dict[str, tuple[str, ...]] = {}
    gate_open: dict[str, tuple[str, ...]] = {}
    for gate in mech.gates:
        cell = (gate.position.x, gate.position.y)
        cx, cy, _ = cell_center(*cell)
        across_x = _across_is_x(cell, walls)
        colour = palette.rgba(gate.color)
        closed, opened = [], []
        for i, d in enumerate((-0.3, 0.0, 0.3)):
            ox, oy = _offset(across_x, d)
            bar, stub = f"gate:{gate.id}:bar{i}", f"gate:{gate.id}:stub{i}"
            world.append(_geom(bar, "cylinder", (cx + ox, cy + oy, hz * 0.9), (0.045, hz * 0.9), colour))
            world.append(_geom(stub, "cylinder", (cx + ox, cy + oy, 0.02), (0.045, 0.02), colour))
            closed.append(bar)
            opened.append(stub)
        for i, z in enumerate((hz * 0.5, hz * 1.6)):
            rail = f"gate:{gate.id}:rail{i}"
            world.append(_geom(rail, "box", (cx, cy, z), _size(across_x, 0.40, 0.03, 0.03), colour))
            closed.append(rail)
        gate_closed[gate.id] = tuple(closed)
        gate_open[gate.id] = tuple(opened)

    switch_off: dict[str, tuple[str, ...]] = {}
    switch_on: dict[str, tuple[str, ...]] = {}
    for sw in mech.switches:
        cx, cy, _ = cell_center(sw.position.x, sw.position.y)
        colour = palette.rgba(sw.color)
        off, on, halo = f"switch:{sw.id}:button_off", f"switch:{sw.id}:button_on", f"switch:{sw.id}:halo"
        world.append(_geom(f"switch:{sw.id}:plate", "box", (cx, cy, 0.008), (0.3, 0.3, 0.008), palette.SWITCH_PLATE))
        world.append(_geom(off, "cylinder", (cx, cy, 0.078), (0.17, 0.07), palette.dim(colour, 0.35)))
        world.append(_geom(on, "cylinder", (cx, cy, 0.031), (0.17, 0.015), colour))
        world.append(_geom(halo, "cylinder", (cx, cy, 0.018), (0.27, 0.004), colour))
        switch_off[sw.id] = (off,)
        switch_on[sw.id] = (on, halo)

    bodies: list[str] = []
    key_bodies: dict[str, str] = {}
    for key in mech.keys:
        cx, cy, _ = cell_center(key.position.x, key.position.y)
        body = f"key:{key.id}"
        parts = "".join(_key_parts(body, palette.rgba(key.color)))
        bodies.append(f'<body name={quoteattr(body)} mocap="true" pos="{_fmt((cx, cy, KEY_HEIGHT))}">{parts}</body>')
        key_bodies[key.id] = body

    carried: dict[str, tuple[str, ...]] = {}
    carried_parts: list[str] = []
    for colour_name in sorted({palette.normalize(k.color) for k in mech.keys}):
        prefix = f"carried:{colour_name}"
        carried_parts += _key_parts(
            prefix, palette.rgba(colour_name), scale=0.7, z=CARRIED_KEY_HEIGHT, group=AGENT_GROUP
        )
        carried[colour_name] = tuple(f"{prefix}:{p}" for p in KEY_PARTS)

    sx, sy, _ = cell_center(spec.maze.start.x, spec.maze.start.y)
    agent = (
        f'<body name="agent" mocap="true" pos="{_fmt((sx, sy, 0.0))}">'
        + _geom("agent:body", "cylinder", (0, 0, 0.14), (0.28, 0.12), palette.AGENT, group=AGENT_GROUP)
        + f'<geom name="agent:nose" type="mesh" mesh="agent_nose" rgba="{_fmt(palette.dim(palette.AGENT, 0.45))}" '
        f'group="{AGENT_GROUP}" contype="0" conaffinity="0"/>'
        + "".join(carried_parts)
        + "</body>"
    )

    newline = "\n    "
    xml = f"""<mujoco model={quoteattr("maze_" + spec.task_id)}>
  <visual>
    <global offwidth="{resolution}" offheight="{resolution}"/>
    <quality shadowsize="0"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.35 0.35 0.35" specular="0 0 0"/>
  </visual>
  <asset>
    <mesh name="agent_nose" vertex="0.42 0 0.2  0.1 0.16 0.2  0.1 -0.16 0.2  0.1 0 0.34"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 10" dir="0.3 0.4 -1" diffuse="0.5 0.5 0.5" specular="0 0 0" castshadow="false"/>
    {newline.join(world)}
    {newline.join(bodies)}
    {agent}
  </worldbody>
</mujoco>"""

    index = SceneIndex(
        width=width,
        height=height,
        wall_height=wall_height,
        wall_cells=walls,
        door_closed=door_closed,
        door_open=door_open,
        switch_off=switch_off,
        switch_on=switch_on,
        gate_closed=gate_closed,
        gate_open=gate_open,
        key_bodies=key_bodies,
        carried=carried,
    )
    return xml, index
