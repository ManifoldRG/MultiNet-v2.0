"""Task spec -> MuJoCo scene (MJCF string) + an index of state-dependent geoms.

Built once per configure(); per-frame state changes only move geoms and mocap
bodies (sync.py), never rebuild the model. The wall set mirrors MiniGrid's
grid construction: border ring + spec walls, minus any cell an object is
placed on (objects overwrite walls there).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from xml.sax.saxutils import quoteattr

from gridworld.task_spec import TaskSpecification

from . import palette
from .cameras import cell_center

KEY_HEIGHT = 0.10
CARRIED_KEY_HEIGHT = 0.62
AGENT_GROUP = 3  # agent geoms; hidden in first_person via MjvOption.geomgroup
HIDDEN_GROUP = 5  # geoms of the inactive state; the renderer never draws this group
KEY_PARTS = ("bow", "shaft", "tooth1", "tooth2")

# PR #57 tiles. Glyph layout copies custom_env.py (2D), lifted off the floor so
# the first-person eye can see them: portal rings and rotator arrows are short
# solids, the skull and frost marks thin discs stacked in 2D's paint order.
PORTAL_RING_TOP = 0.14
ROTATOR_ARROW_BASE = 0.024  # top of the orange slab
ROTATOR_ARROW_TOP = 0.10
# Arrow outline per direction (0=E 1=S 2=W 3=N) in cell-relative world units,
# from the 2D triangles: image (u, v) -> (u - 0.5, 0.5 - v).
ROTATOR_ARROWS: tuple[tuple[tuple[float, float], ...], ...] = (
    ((-0.28, 0.28), (-0.28, -0.28), (0.32, 0.0)),
    ((-0.28, 0.28), (0.28, 0.28), (0.0, -0.32)),
    ((0.28, 0.28), (0.28, -0.28), (-0.32, 0.0)),
    ((-0.28, -0.28), (0.28, -0.28), (0.0, 0.32)),
)


class UnsupportedSpecError(ValueError):
    """The spec uses a feature the 3D renderer cannot draw faithfully."""


def check_supported(spec: TaskSpecification) -> None:
    problems = []
    mech = spec.mechanisms
    if mech.blocks:
        problems.append("blocks")
    if mech.hazards:
        problems.append("hazards")
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
        + [tp.color for tp in mech.teleporters]
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
    for tp in mech.teleporters:
        occupied |= {(tp.position_a.x, tp.position_a.y), (tp.position_b.x, tp.position_b.y)}
    for group in (mech.kill_cells, mech.frozen_tiles, mech.rotating_tiles):
        occupied |= {(p.x, p.y) for p in group}
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
    # rotating tile index (spec order) -> geom names of its arrow, per direction
    rotator_arrows: dict[int, tuple[tuple[str, ...], ...]] = field(default_factory=dict)
    rotator_initial: tuple[int, ...] = ()


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


def _disc(name, cx, cy, radius, z_bottom, z_top, rgba) -> str:
    half = (z_top - z_bottom) / 2.0
    return _geom(name, "cylinder", (cx, cy, z_bottom + half), (radius, half), rgba)


def _portal_end(prefix, cx, cy, colour) -> list[str]:
    # 2D: colour ring r.42, dark core r.26, colour dot r.12 -- each a step taller
    # so the top-down view shows the same concentric glyph.
    return [
        _disc(f"{prefix}:ring", cx, cy, 0.42, 0.0, PORTAL_RING_TOP, colour),
        _disc(f"{prefix}:core", cx, cy, 0.26, 0.0, PORTAL_RING_TOP + 0.01, palette.PORTAL_CORE),
        _disc(f"{prefix}:dot", cx, cy, 0.12, 0.0, PORTAL_RING_TOP + 0.02, colour),
    ]


def _kill_cell(prefix, cx, cy) -> list[str]:
    # 2D: dark-red disc r.48; white skull r.30 centred a little north; two eyes
    # and a nose. The skull has some height so it reads from the eye view.
    return [
        _disc(f"{prefix}:disc", cx, cy, 0.48, 0.0, 0.02, palette.KILL_DISC),
        _disc(f"{prefix}:skull", cx, cy + 0.08, 0.30, 0.0, 0.08, palette.SKULL),
        _disc(f"{prefix}:eye0", cx - 0.12, cy + 0.12, 0.08, 0.08, 0.09, palette.SKULL_DARK),
        _disc(f"{prefix}:eye1", cx + 0.12, cy + 0.12, 0.08, 0.08, 0.09, palette.SKULL_DARK),
        _disc(f"{prefix}:nose", cx, cy - 0.08, 0.06, 0.08, 0.09, palette.SKULL_DARK),
    ]


def _frozen_tile(prefix, cx, cy) -> list[str]:
    # 2D: pale-blue tile, white spot r.16, two flecks.
    return [
        _geom(f"{prefix}:slab", "box", (cx, cy, 0.012), (0.46, 0.46, 0.012), palette.FROZEN_TILE),
        _disc(f"{prefix}:spot", cx, cy, 0.16, 0.024, 0.034, palette.FROZEN_SPOT),
        _disc(f"{prefix}:fleck0", cx - 0.22, cy + 0.18, 0.07, 0.024, 0.034, palette.FROZEN_FLECK),
        _disc(f"{prefix}:fleck1", cx + 0.20, cy - 0.12, 0.06, 0.024, 0.034, palette.FROZEN_FLECK),
    ]


def _arrow_mesh(direction: int) -> str:
    """Vertices of the arrow prism for ``direction`` (convex hull of 6 points)."""
    return "  ".join(
        f"{x} {y} {z}"
        for z in (ROTATOR_ARROW_BASE, ROTATOR_ARROW_TOP)
        for x, y in ROTATOR_ARROWS[direction]
    )


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

    for tp in mech.teleporters:
        colour = palette.rgba(tp.color)
        for end, pos in (("a", tp.position_a), ("b", tp.position_b)):
            cx, cy, _ = cell_center(pos.x, pos.y)
            world += _portal_end(f"portal:{tp.id}:{end}", cx, cy, colour)
    for i, cell in enumerate(mech.kill_cells):
        cx, cy, _ = cell_center(cell.x, cell.y)
        world += _kill_cell(f"kill:{i}", cx, cy)
    for i, cell in enumerate(mech.frozen_tiles):
        cx, cy, _ = cell_center(cell.x, cell.y)
        world += _frozen_tile(f"frozen:{i}", cx, cy)

    rotator_arrows: dict[int, tuple[tuple[str, ...], ...]] = {}
    for i, cell in enumerate(mech.rotating_tiles):
        cx, cy, _ = cell_center(cell.x, cell.y)
        world.append(_geom(f"rotator:{i}:slab", "box", (cx, cy, 0.012), (0.46, 0.46, 0.012), palette.ROTATOR_TILE))
        arrows = []
        for d in range(len(ROTATOR_ARROWS)):
            name = f"rotator:{i}:arrow{d}"
            world.append(
                f'<geom name={quoteattr(name)} type="mesh" mesh="rotator_arrow_{d}" pos="{_fmt((cx, cy, 0.0))}" '
                f'rgba="{_fmt(palette.ROTATOR_ARROW)}" group="0" contype="0" conaffinity="0"/>'
            )
            arrows.append((name,))
        rotator_arrows[i] = tuple(arrows)
    arrow_meshes = "".join(
        f'<mesh name="rotator_arrow_{d}" vertex="{_arrow_mesh(d)}"/>' for d in range(len(ROTATOR_ARROWS))
    )

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
    # A wedge pointing along the body's +x (its facing), like MiniGrid's triangle.
    agent = (
        f'<body name="agent" mocap="true" pos="{_fmt((sx, sy, 0.0))}">'
        + f'<geom name="agent:body" type="mesh" mesh="agent_wedge" rgba="{_fmt(palette.AGENT)}" '
        f'group="{AGENT_GROUP}" contype="0" conaffinity="0"/>'
        + "".join(carried_parts)
        + "</body>"
    )
    wedge = "  ".join(
        f"{x} {y} {z}" for z in (0.02, 0.28) for x, y in ((0.42, 0), (-0.26, 0.3), (-0.26, -0.3))
    )

    newline = "\n    "
    xml = f"""<mujoco model={quoteattr("maze_" + spec.task_id)}>
  <visual>
    <global offwidth="{resolution}" offheight="{resolution}"/>
    <quality shadowsize="0"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.35 0.35 0.35" specular="0 0 0"/>
  </visual>
  <asset>
    <mesh name="agent_wedge" vertex="{wedge}"/>
    {arrow_meshes}
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
        rotator_arrows=rotator_arrows,
        rotator_initial=tuple(int(d) for d in mech.rotating_initial_directions),
    )
    return xml, index
