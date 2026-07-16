"""Strict-lockstep batch runner over a working set of ``EpisodeStepper``s.

The batch-API models (Claude/Kimi/Moonshot) are cheapest and highest-throughput
when many prompts are submitted as ONE batch. This runner drives up to
``max_batches`` episodes in lockstep: every round it collects exactly one pending
query per live episode, submits them together via ``agent.generate_batch`` (in
stable unit order), feeds each reply back into its own stepper, then checkpoints
and loops. Terminated episodes free a slot that ``refill`` (or queued ``add``
units) tops back up, so the batch stays saturated until the pool drains.

HARD ISOLATION CONTRACT (inherited from ``EpisodeStepper``): every
``LockstepUnit`` owns its own ``ExperimentRunner``/backend/stepper. Steppers are
never shared — that is what keeps each episode's ``last_rgb``/querying/transcript
state from colliding with another's inside a single round.

Round loop (per the task brief, adapted to the real seams):

  1. Top the working set up to ``max_batches`` from queued/refilled units.
  2. For each live unit call ``next_query()`` — which *drains local actions* until
     it needs a query or the episode ends. ``None`` means finished this round:
     capture ``result()`` and delete its checkpoint. A returned message list
     means the unit participates; **this is the one legal checkpoint boundary**
     (both stepper buffers are empty right after ``next_query`` returns), so we
     checkpoint here, NOT after ``apply_reply`` (which leaves queued-but-undrained
     actions — ``save_checkpoint`` would rightly refuse).
  3. ``agent.generate_batch([...messages...])`` in stable unit order.
  4. ``apply_reply`` each reply into its unit's stepper. Batch-item stub replies
     (``stop_reason`` like ``batch_expired``) carry empty text and flow through as
     ordinary parse failures — no special-casing needed.
  5. ``on_round(active_units)`` — a heartbeat hook for the coordinator.

A stepper that raises is caught, recorded as ``{"error": str(exc)}`` for that
unit, and the batch continues; one bad episode never kills the run.
"""

from __future__ import annotations

import inspect
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

from interface.episode_checkpoint import save_checkpoint
from interface.episode_step import EpisodeStepper

logger = logging.getLogger(__name__)


@dataclass
class LockstepUnit:
    """One episode in the working set: its id, its stepper, its checkpoint file.

    ``checkpoint_path`` is optional; when set, the runner writes a query-boundary
    checkpoint each round the unit is active and deletes it when the unit ends.
    """

    unit_id: str
    stepper: EpisodeStepper
    checkpoint_path: Optional[Path] = None


def _accepts_deadline_kwarg(func: Callable) -> bool:
    """True iff ``func`` explicitly names a ``deadline_s`` parameter.

    The served agents (scripted double, qwen fan-out) take only ``batch``; the
    Claude/Kimi/Moonshot batch clients surface the round deadline through their
    *config* (``batch_deadline_s``), not a call kwarg. So for every current agent
    this returns ``False`` and we call ``generate_batch(batch)`` plainly. The shim
    exists only so a future agent that DOES accept ``deadline_s`` gets it passed.
    We deliberately do not count ``**kwargs`` — passing a kwarg an agent merely
    swallows is worse than not passing it.
    """
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return False
    param = sig.parameters.get("deadline_s")
    return param is not None and param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


class LockstepBatchRunner:
    def __init__(
        self,
        agent,
        *,
        max_batches: int,
        round_deadline_s: float = 7200.0,
        refill: Optional[Callable[[], Optional[LockstepUnit]]] = None,
        on_round: Optional[Callable[[List[LockstepUnit]], None]] = None,
    ) -> None:
        if max_batches < 1:
            raise ValueError(f"max_batches must be >= 1, got {max_batches}")
        generate_batch = getattr(agent, "generate_batch", None)
        if not callable(generate_batch):
            raise TypeError("agent must provide a callable generate_batch(batch)")
        self._agent = agent
        self._max_batches = max_batches
        self._round_deadline_s = round_deadline_s
        self._refill = refill
        self._on_round = on_round
        self._pending: Deque[LockstepUnit] = deque()
        self._results: Dict[str, dict] = {}
        self._batch_accepts_deadline = _accepts_deadline_kwarg(generate_batch)

    def add(self, unit: LockstepUnit) -> None:
        """Enqueue a unit into the pool; it enters the working set when a slot is free."""
        self._pending.append(unit)

    def run(self) -> Dict[str, dict]:
        """Drive every queued/refilled unit to completion; return unit_id -> result.

        A unit whose stepper raises anywhere in the round contributes
        ``{"error": str(exc)}`` instead of a normal result and is dropped; other
        units in the batch are unaffected.
        """
        working: List[LockstepUnit] = []
        while True:
            self._refill_working(working)
            if not working:
                break

            batch_units: List[LockstepUnit] = []
            batch_messages: List[List[dict]] = []
            still_active: List[LockstepUnit] = []
            for unit in working:
                try:
                    self._ensure_started(unit)
                    messages = unit.stepper.next_query()
                    if messages is None:
                        self._finish(unit)
                        continue
                    # Query boundary: both stepper buffers are empty right now,
                    # which is the ONLY point save_checkpoint accepts (after
                    # apply_reply the parsed actions sit undrained and it would
                    # refuse). The write stays INSIDE the guard so a failed
                    # checkpoint (OSError disk-full, etc.) errors only this unit
                    # and never aborts the in-flight batch.
                    if unit.checkpoint_path is not None:
                        save_checkpoint(unit.checkpoint_path, unit.stepper)
                except Exception as exc:  # noqa: BLE001 - isolate one bad episode
                    self._record_error(unit, exc)
                    continue
                batch_units.append(unit)
                batch_messages.append(messages)
                still_active.append(unit)

            working = still_active

            if batch_units:
                try:
                    replies = self._call_batch(batch_messages)
                except Exception as exc:  # noqa: BLE001 - a raised batch tick must not
                    # discard the whole round. A batch client can still raise (e.g. a
                    # cancel-on-ended 4xx that escaped its salvage, or a hard HTTP
                    # error): mark every unit that was in flight THIS round as an
                    # error result via the normal per-unit path, so previously
                    # completed results survive and run() returns instead of the
                    # worker crashing and dropping self._results.
                    for unit in batch_units:
                        self._record_error(unit, exc)
                    replies = None
                    working = []
                if replies is not None:
                    if len(replies) != len(batch_units):
                        raise RuntimeError(
                            "generate_batch returned "
                            f"{len(replies)} replies for {len(batch_units)} prompts "
                            "(must be input-order, one per prompt)"
                        )
                    errored_ids: set[int] = set()
                    for unit, reply in zip(batch_units, replies):
                        try:
                            unit.stepper.apply_reply(reply)
                        except Exception as exc:  # noqa: BLE001 - isolate one bad episode
                            self._record_error(unit, exc)
                            errored_ids.add(id(unit))
                    if errored_ids:
                        # Identity-based drop; value-remove on a mutable dataclass is
                        # fragile (two units could compare equal).
                        working = [u for u in working if id(u) not in errored_ids]

            # Heartbeat on EVERY round that had active units at the top (we only
            # get here past the `not working` break), including a silent round
            # where all units finished and no batch was submitted. The
            # coordinator worker keys its liveness cadence off this callback, so
            # it must fire every round or units risk being marked stale.
            if self._on_round is not None:
                self._on_round(list(working))

        return self._results

    # -- internals -----------------------------------------------------------

    def _refill_working(self, working: List[LockstepUnit]) -> None:
        """Top the working set up to ``max_batches`` from the pool/refill source."""
        while len(working) < self._max_batches:
            unit = self._next_unit()
            if unit is None:
                return
            working.append(unit)

    def _next_unit(self) -> Optional[LockstepUnit]:
        """Next unit from queued ``add`` units first, then the ``refill`` callback."""
        if self._pending:
            return self._pending.popleft()
        if self._refill is not None:
            return self._refill()
        return None

    @staticmethod
    def _ensure_started(unit: LockstepUnit) -> None:
        """Start a fresh stepper lazily; leave an already-started/resumed one alone.

        ``start()`` (and ``resume_stepper``) set ``transcript``; a stepper straight
        out of ``__init__`` has no such attribute, so its absence is the clean
        "not yet started" signal and resumed steppers are never double-started.
        """
        if not hasattr(unit.stepper, "transcript"):
            unit.stepper.start()

    def _call_batch(self, batch_messages: List[List[dict]]):
        if self._batch_accepts_deadline:
            return self._agent.generate_batch(
                batch_messages, deadline_s=self._round_deadline_s
            )
        return self._agent.generate_batch(batch_messages)

    def _finish(self, unit: LockstepUnit) -> None:
        try:
            self._results[unit.unit_id] = unit.stepper.result()
        except Exception as exc:  # noqa: BLE001 - a bad result() is still just one unit
            self._record_error(unit, exc)
            return
        self._delete_checkpoint(unit)

    def _record_error(self, unit: LockstepUnit, exc: Exception) -> None:
        logger.exception("unit %s failed: %s", unit.unit_id, exc)
        self._results[unit.unit_id] = {"error": str(exc)}
        self._delete_checkpoint(unit)

    @staticmethod
    def _delete_checkpoint(unit: LockstepUnit) -> None:
        if unit.checkpoint_path is not None:
            Path(unit.checkpoint_path).unlink(missing_ok=True)
