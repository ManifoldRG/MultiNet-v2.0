"""Runtime signal reconstruction from recorded episode state transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gridworld.task_spec import TaskSpecification
from gridworld.task_validator import TaskValidator


_MISSING = object()
_TERMINATED_REASON_ALIASES = {
    "goal_reached": "goal_reached",
    "success": "goal_reached",
    "hazard": "hazard",
    "max_steps": "max_steps",
    "truncated": "max_steps",
    "deadlock": "deadlock",
    "exhausted": "deadlock",
    "invalid_action_excess": "invalid_action_excess",
    "parse_failed": "invalid_action_excess",
}


@dataclass(frozen=True)
class RuntimeDiagnostics:
    """Signals reconstructed from one recorded run."""

    distractor_interactions: int
    irreversible_failures: int
    mechanism_interaction_order: list[str]
    failure_point: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "distractor_interactions": self.distractor_interactions,
            "irreversible_failures": self.irreversible_failures,
            "mechanism_interaction_order": list(self.mechanism_interaction_order),
            "failure_point": dict(self.failure_point) if self.failure_point is not None else None,
        }


def normalize_terminated_reason(run: dict[str, Any], success: bool) -> str:
    """Return the canonical termination enum or reject an off-contract reason."""
    raw_reason = run.get("terminated_reason") or run.get("end_reason")
    if raw_reason is None:
        if success:
            return "goal_reached"
        if run.get("truncated"):
            return "max_steps"
        raise ValueError("Runtime scoring requires terminated_reason for failed runs")

    normalized = _TERMINATED_REASON_ALIASES.get(str(raw_reason))
    if normalized is None:
        allowed = ", ".join(sorted(set(_TERMINATED_REASON_ALIASES.values())))
        raise ValueError(
            f"Unsupported terminated_reason {raw_reason!r}; expected one of: {allowed}"
        )
    return normalized


def collect_runtime_diagnostics(run: dict[str, Any]) -> dict[str, Any]:
    """Return required Stage 4 diagnostics, computing them when necessary."""
    provided = {
        key: _provided_signal(run, key)
        for key in (
            "distractor_interactions",
            "irreversible_failures",
            "path_choice",
            "mechanism_interaction_order",
            "failure_point",
        )
    }
    needs_core_diagnostics = any(
        provided[key] is _MISSING
        for key in ("distractor_interactions", "irreversible_failures")
    )
    needs_test_three_diagnostics = _is_test_three_run(run) and any(
        provided[key] is _MISSING
        for key in ("mechanism_interaction_order", "failure_point")
    )

    reconstructed: dict[str, Any] = {}
    spec = _task_spec_from_run(run)
    if spec is not None:
        reconstructed = _reconstruct_diagnostics(run, spec).to_dict()
    elif needs_core_diagnostics or needs_test_three_diagnostics:
        raise ValueError(
            "Runtime scoring requires task_spec or precomputed "
            "distractor_interactions and irreversible_failures telemetry"
        )

    signals: dict[str, Any] = {}
    for key, value in provided.items():
        if key in reconstructed:
            signals[key] = reconstructed[key]
        elif value is not _MISSING:
            signals[key] = value

    if _is_test_two_run(run) and "path_choice" not in signals:
        raise ValueError("Test 2 runtime rows require path_choice telemetry")
    if _is_test_three_run(run):
        missing = [
            key
            for key in ("mechanism_interaction_order", "failure_point")
            if key not in signals
        ]
        if missing:
            raise ValueError(
                "Test 3 runtime rows require telemetry: " + ", ".join(missing)
            )
    return signals


def _provided_signal(run: dict[str, Any], key: str) -> Any:
    if key in run:
        return run[key]
    signals = run.get("signals")
    if isinstance(signals, dict) and key in signals:
        return signals[key]
    return _MISSING


def _task_spec_from_run(run: dict[str, Any]) -> TaskSpecification | None:
    raw_spec = run.get("task_spec")
    if not isinstance(raw_spec, dict):
        return None
    spec = TaskSpecification.from_dict(raw_spec)
    schema_valid, schema_errors = spec.validate()
    if not schema_valid:
        raise ValueError(
            f"Runtime task_spec failed schema validation: {'; '.join(schema_errors)}"
        )
    return spec


def _reconstruct_diagnostics(
    run: dict[str, Any],
    spec: TaskSpecification,
) -> RuntimeDiagnostics:
    validator = TaskValidator(spec)
    transitions = _extract_transitions(run)
    if (
        (_requires_transition_telemetry(spec) or _is_test_three_run(run))
        and _run_steps(run) > 0
        and not transitions
    ):
        raise ValueError(
            "Runtime scoring requires state_before/state_after transition telemetry "
            "for tasks with interactive or irreversible mechanisms"
        )
    if spec.mechanisms.blocks and any(
        "block_positions" not in transition[state_key]
        for transition in transitions
        for state_key in ("before", "after")
    ):
        raise ValueError(
            "Runtime scoring requires block_positions in transition telemetry "
            "for tasks with pushable blocks"
        )

    distractor_interactions = 0
    irreversible_failures = 0
    interaction_order: list[str] = []
    failure_point: dict[str, Any] | None = None
    for transition in transitions:
        labels = _transition_labels(spec, transition["before"], transition["after"])
        interaction_order.extend(labels)
        distractor_interactions += sum(
            1 for label in labels if validator.action_label_matches_distractor(label)
        )

        irreversible_labels = [
            label for label in labels if validator.is_irreversible_action_label(label)
        ]
        if (
            irreversible_labels
            and validator.is_beatable_from_snapshot(transition["before"])
            and not validator.is_beatable_from_snapshot(transition["after"])
        ):
            irreversible_failures += 1
            if failure_point is None:
                failure_point = {
                    "step": transition["step"],
                    "action": transition.get("action"),
                    "interactions": irreversible_labels,
                }

    return RuntimeDiagnostics(
        distractor_interactions=distractor_interactions,
        irreversible_failures=irreversible_failures,
        mechanism_interaction_order=interaction_order,
        failure_point=failure_point,
    )


def _extract_transitions(run: dict[str, Any]) -> list[dict[str, Any]]:
    transcript = run.get("transcript")
    if isinstance(transcript, list):
        transitions = [
            {
                "step": int(item.get("step_index", index + 1)),
                "action": item.get("action"),
                "before": item["state_before"],
                "after": item["state_after"],
            }
            for index, item in enumerate(transcript)
            if isinstance(item, dict)
            and item.get("kind") == "step"
            and isinstance(item.get("state_before"), dict)
            and isinstance(item.get("state_after"), dict)
        ]
        if transitions:
            return transitions

    trajectory = run.get("trajectory")
    if not isinstance(trajectory, list):
        return []
    state_rows = [item for item in trajectory if isinstance(item, dict)]
    transitions = []
    for index, item in enumerate(state_rows):
        before = item.get("state")
        after = (
            state_rows[index + 1].get("state")
            if index + 1 < len(state_rows)
            else run.get("final_state")
        )
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        transitions.append(
            {
                "step": int(item.get("step", index + 1)),
                "action": item.get("action_name", item.get("action")),
                "before": before,
                "after": after,
            }
        )
    return transitions


def _transition_labels(
    spec: TaskSpecification,
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[str]:
    labels: list[str] = []
    before_keys = _string_set(before, "collected_keys")
    after_keys = _string_set(after, "collected_keys")
    labels.extend(f"pickup:{key_id}" for key_id in sorted(after_keys - before_keys))

    before_doors = _string_set(before, "open_doors")
    after_doors = _string_set(after, "open_doors")
    labels.extend(f"open_door:{door_id}" for door_id in sorted(after_doors - before_doors))

    before_switches = _string_set(before, "active_switches")
    after_switches = _string_set(after, "active_switches")
    labels.extend(
        f"toggle:{switch_id}"
        for switch_id in sorted(before_switches ^ after_switches)
    )

    before_blocks = _block_positions(before)
    after_blocks = _block_positions(after)
    labels.extend(
        f"push:{block_id}:{after_blocks[block_id][0]},{after_blocks[block_id][1]}"
        for block_id in sorted(before_blocks.keys() & after_blocks.keys())
        if before_blocks[block_id] != after_blocks[block_id]
    )

    before_pos = _agent_position(before)
    after_pos = _agent_position(after)
    direction = before.get("agent_direction")
    direction_vectors = ((1, 0), (0, 1), (-1, 0), (0, -1))
    if (
        before_pos is not None
        and after_pos is not None
        and isinstance(direction, int)
        and 0 <= direction < len(direction_vectors)
    ):
        dx, dy = direction_vectors[direction]
        front = before_pos[0] + dx, before_pos[1] + dy
        for teleporter in spec.mechanisms.teleporters:
            endpoints = [
                (teleporter.position_a.to_tuple(), teleporter.position_b.to_tuple())
            ]
            if teleporter.bidirectional:
                endpoints.append(
                    (teleporter.position_b.to_tuple(), teleporter.position_a.to_tuple())
                )
            if any(
                front == origin and after_pos == destination
                for origin, destination in endpoints
            ):
                labels.append(f"teleport:{teleporter.id}")
                break
    return labels


def _string_set(state: dict[str, Any], key: str) -> set[str]:
    raw = state.get(key, [])
    return {str(value) for value in raw} if isinstance(raw, (list, set, tuple)) else set()


def _block_positions(state: dict[str, Any]) -> dict[str, tuple[int, int]]:
    raw = state.get("block_positions", {})
    if not isinstance(raw, dict):
        return {}
    return {
        str(block_id): (int(pos[0]), int(pos[1]))
        for block_id, pos in raw.items()
        if isinstance(pos, (list, tuple)) and len(pos) >= 2
    }


def _agent_position(state: dict[str, Any]) -> tuple[int, int] | None:
    raw = state.get("agent_position")
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    return int(raw[0]), int(raw[1])


def _run_steps(run: dict[str, Any]) -> int:
    for key in ("steps", "steps_taken", "steps_used"):
        if run.get(key) is not None:
            return int(run[key])
    return 0


def _requires_transition_telemetry(spec: TaskSpecification) -> bool:
    return bool(
        spec.distractors
        or spec.mechanisms.blocks
        or spec.mechanisms.teleporters
        or (spec.rules.key_consumption and spec.mechanisms.doors)
        or any(
            switch.switch_type == "one_shot"
            for switch in spec.mechanisms.switches
        )
    )


def _is_test_two_run(run: dict[str, Any]) -> bool:
    experiment = str(run.get("experiment", "")).lower().replace("-", "_")
    return "test2" in experiment or "complexity_distance" in experiment


def _is_test_three_run(run: dict[str, Any]) -> bool:
    experiment = str(run.get("experiment", "")).lower().replace("-", "_")
    return "test3" in experiment or "mechanism_order" in experiment
