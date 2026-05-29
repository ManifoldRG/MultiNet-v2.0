"""JSON and hash helpers for scorer artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from gridworld.task_spec import TaskSpecification


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=json_default)
        f.write("\n")


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=json_default)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def task_spec_from_payload(data: dict[str, Any]) -> TaskSpecification:
    if "task_spec" in data and isinstance(data["task_spec"], dict):
        return TaskSpecification.from_dict(data["task_spec"])
    if "TaskSpecification" in data and isinstance(data["TaskSpecification"], dict):
        return TaskSpecification.from_dict(data)
    required_fields = {"task_id", "maze", "goal", "max_steps"}
    if not required_fields.issubset(data):
        raise ValueError(
            "Input JSON is not a task artifact. Expected task fields or a nested task_spec."
        )
    return TaskSpecification.from_dict(data)
