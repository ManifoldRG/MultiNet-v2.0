from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple, Optional

Pos = Tuple[int, int]

FACING_ORDER = ["NORTH", "EAST", "SOUTH", "WEST"]

FACING_TO_DELTA: Dict[str, Tuple[int, int]] = {
    "NORTH": (-1,  0),
    "EAST":  ( 0,  1),
    "SOUTH": ( 1,  0),
    "WEST":  ( 0, -1),
}


@dataclass
class GridState:
    rows: int
    cols: int
    walls: Set[Pos]
    start: Pos
    goal: Pos
    agent_pos: Pos
    facing: str = "NORTH"
    step_count: int = 0
    max_steps: int = 50
    inventory: List[str] = field(default_factory=list)   # collected key colors
    keys: List[Dict[str, Any]] = field(default_factory=list)
    doors: List[Dict[str, Any]] = field(default_factory=list)
    switches: List[Dict[str, Any]] = field(default_factory=list)
    gates: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StepEvent:
    type: str    # TURNED, MOVED, BLOCKED, DONE, PICKUP, TOGGLED, NOTHING, WRONG_DONE, INVALID
    message: str


class GridWorldEnv:

    def __init__(
        self,
        rows: int,
        cols: int,
        walls: Set[Pos],
        start: Pos,
        goal: Pos,
        max_steps: int = 50,
        mechanisms: Optional[Dict[str, Any]] = None,
    ):
        mechs = mechanisms or {}
        self.initial = GridState(
            rows=rows,
            cols=cols,
            walls=walls,
            start=start,
            goal=goal,
            agent_pos=start,
            max_steps=max_steps,
            keys=mechs.get("keys", []),
            doors=mechs.get("doors", []),
            switches=mechs.get("switches", []),
            gates=mechs.get("gates", []),
        )
        self.state: Optional[GridState] = None

    def reset(self) -> GridState:
        s = self.initial
        self.state = GridState(
            rows=s.rows,
            cols=s.cols,
            walls=set(s.walls),
            start=s.start,
            goal=s.goal,
            agent_pos=s.start,
            facing="NORTH",
            step_count=0,
            max_steps=s.max_steps,
            inventory=[],
            keys=[dict(k) for k in s.keys],
            doors=[dict(d) for d in s.doors],
            switches=[{**dict(sw), "on": bool(sw.get("on", False))} for sw in s.switches],
            gates=[GridWorldEnv._gate_state_from_switches(dict(g), s.switches) for g in s.gates],
        )
        return self.state

    @staticmethod
    def _gate_state_from_switches(gate: Dict, switches: List[Dict]) -> Dict:
        """Gates are open if any linked switch is on, else use initial/embedded state."""
        g = dict(gate)
        gid = g.get("id")
        if gid:
            if any(
                bool(sw.get("on")) and gid in sw.get("controls", [])
                for sw in switches
            ):
                g["state"] = "open"
            else:
                g["state"] = g.get("state", g.get("initial_state", "closed"))
        return g

    def step(self, action: str) -> tuple[GridState, StepEvent]:
        assert self.state is not None, "Call reset() first."

        verb = action.strip().upper()

        # --- Turns ---
        if verb in ("TURN_LEFT", "TURN_RIGHT"):
            idx = FACING_ORDER.index(self.state.facing)
            self.state.facing = FACING_ORDER[(idx + (-1 if verb == "TURN_LEFT" else 1)) % 4]
            self.state.step_count += 1
            return self.state, StepEvent("TURNED", f"Now facing {self.state.facing}.")

        # --- Move one step forward ---
        if verb == "MOVE_FORWARD":
            dr, dc = FACING_TO_DELTA[self.state.facing]
            r, c   = self.state.agent_pos
            nr, nc = r + dr, c + dc
            reason = self._blocked(nr, nc)
            if reason:
                return self.state, StepEvent("BLOCKED", f"MOVE_FORWARD blocked by {reason}.")
            self.state.agent_pos = (nr, nc)
            # With matching key in inventory, moving onto a door tile opens it (no TOGGLE on doors)
            door = self._door_at((nr, nc))
            if door and door["requires_key"] in self.state.inventory:
                self.state.doors = [
                    d for d in self.state.doors if tuple(d["position"]) != (nr, nc)
                ]
            self.state.step_count += 1
            if self.state.agent_pos == self.state.goal:
                return self.state, StepEvent("DONE", f"Reached goal at {self.state.goal}.")
            return self.state, StepEvent("MOVED", f"Moved to {self.state.agent_pos}.")

        # --- Pick up object at current position ---
        if verb == "PICKUP":
            pos = self.state.agent_pos
            key = self._key_at(pos)
            if key:
                self.state.inventory.append(key["color"])
                self.state.keys = [k for k in self.state.keys if tuple(k["position"]) != pos]
                self.state.step_count += 1
                return self.state, StepEvent("PICKUP", f"Picked up {key['color']} key.")
            self.state.step_count += 1
            return self.state, StepEvent("NOTHING", f"Nothing to pick up at {pos}.")

        # --- Toggle facing switch only (opens/closes linked gates; doors and gates are not toggled directly) ---
        if verb == "TOGGLE":
            dr, dc = FACING_TO_DELTA[self.state.facing]
            r, c   = self.state.agent_pos
            target = (r + dr, c + dc)
            sw = self._switch_at(target)
            if sw:
                self._toggle_switch(sw)
                self.state.step_count += 1
                st = "on" if sw.get("on") else "off"
                return self.state, StepEvent("TOGGLED", f"Switch at {target} is {st}.")
            self.state.step_count += 1
            if self._door_at(target) or self._gate_at(target):
                return self.state, StepEvent("NOTHING", "Use PICKUP to collect keys. Doors open when you have the right key. Only switches can be TOGGLED (gates follow switch on/off).")
            return self.state, StepEvent("NOTHING", f"No switch to toggle at {target}.")

        # --- Agent signals task complete ---
        if verb == "DONE":
            if self.state.agent_pos == self.state.goal:
                return self.state, StepEvent("DONE", f"Task complete at {self.state.goal}.")
            self.state.step_count += 1
            return self.state, StepEvent("WRONG_DONE", f"DONE called but not at goal {self.state.goal}.")

        return self.state, StepEvent("INVALID", f"Unknown action: {action}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _blocked(self, nr: int, nc: int) -> Optional[str]:
        """Return a reason string if (nr, nc) is impassable, else None."""
        if nr < 1 or nr > self.state.rows or nc < 1 or nc > self.state.cols:
            return "out of bounds"
        if (nr, nc) in self.state.walls:
            return "wall"
        door = self._door_at((nr, nc))
        if door and door["requires_key"] not in self.state.inventory:
            return f"locked {door['requires_key']} door"
        gate = self._gate_at((nr, nc))
        if gate and gate.get("state", gate.get("initial_state", "closed")) == "closed":
            return "closed gate"
        return None

    def _key_at(self, pos: Pos):
        return next((k for k in self.state.keys    if tuple(k["position"]) == pos), None)

    def _door_at(self, pos: Pos):
        return next((d for d in self.state.doors   if tuple(d["position"]) == pos), None)

    def _switch_at(self, pos: Pos):
        return next((s for s in self.state.switches if tuple(s["position"]) == pos), None)

    def _gate_at(self, pos: Pos):
        return next((g for g in self.state.gates   if tuple(g["position"]) == pos), None)

    def _recompute_gates_from_switches(self) -> None:
        """A gate is open if any of its linked switches is on."""
        for gate in self.state.gates:
            gid = gate.get("id")
            if not gid:
                continue
            on = any(
                bool(s.get("on")) and gid in s.get("controls", [])
                for s in self.state.switches
            )
            gate["state"] = "open" if on else "closed"

    def _toggle_switch(self, sw: Dict) -> None:
        sw["on"] = not sw.get("on", False)
        self._recompute_gates_from_switches()
