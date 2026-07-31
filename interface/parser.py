import re
from typing import List, Optional

ACTION_ORDER = (
    "TURN_LEFT",
    "TURN_RIGHT",
    "MOVE_FORWARD",
    "PICKUP",
    "DROP",
    "TOGGLE",
    "DONE",
)
VALID_ACTIONS = set(ACTION_ORDER)
ACTIONS_HINT = ", ".join(ACTION_ORDER)

_SYNONYMS = {
    "turn left": "TURN_LEFT",
    "rotate left": "TURN_LEFT",
    "turn right": "TURN_RIGHT",
    "rotate right": "TURN_RIGHT",
    "move forward": "MOVE_FORWARD",
    "go forward": "MOVE_FORWARD",
    "forward": "MOVE_FORWARD",
    "pick up": "PICKUP",
    "pickup": "PICKUP",
    "drop": "DROP",
    "drop key": "DROP",
    "put down": "DROP",
    "release": "DROP",
    "toggle": "TOGGLE",
    "done": "DONE",
    "finished": "DONE",
}

_FINAL_OUTPUT_RE = re.compile(r"(?i)^FINAL_OUTPUT\s*:\s*(.*)\s*$")


def parse_final_output(
    text: str,
    allow_regex_fallback: bool = True,
    *,
    valid_actions: set = VALID_ACTIONS,
    synonyms: dict = _SYNONYMS,
) -> Optional[List[str]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    trailing = lines[-5:] if len(lines) >= 5 else lines

    for line in reversed(trailing):
        m = _FINAL_OUTPUT_RE.match(line)
        if m:
            rest = m.group(1).strip()
            if not rest:
                return None
            out: List[str] = []
            for part in rest.split(","):
                p = part.strip()
                if not p:
                    continue
                a = normalize_action(p, valid_actions=valid_actions)
                if not a:
                    return None
                out.append(a)
            return out if out else None
        if re.match(r"(?i)^FINAL_OUTPUT\s*:", line):
            return None

    if allow_regex_fallback:
        norm = text.lower()
        matches = []
        for phrase, canonical in synonyms.items():
            pattern = re.escape(phrase).replace(r"\ ", r"\s+")
            for m in re.finditer(pattern, norm):
                matches.append((m.start(), canonical))
        if matches:
            matches.sort(key=lambda x: x[0])
            return [matches[-1][1]]

    return None


def normalize_action(raw: str, *, valid_actions: set = VALID_ACTIONS) -> str:
    verb = raw.strip().upper().replace(" ", "_")
    return verb if verb in valid_actions else ""
