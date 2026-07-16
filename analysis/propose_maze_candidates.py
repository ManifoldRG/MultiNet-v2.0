#!/usr/bin/env python3
"""Propose balanced 50-maze benchmark panels from the OGBench maze corpus.

The selector uses executable BFS action length, rather than geometric distance.
Beatability, optimal action count, and object counts come from
``gridworld.task_validator.compute_difficulty`` (the repo's difficulty
authority, fed the same ``plan_bfs_path`` result used for the path-signature
analysis), so panels always agree with ``analysis.mazes.maze_features`` and the
validator on lengths. The module distinguishes three notions that are easy to
conflate:

* ``object_count``: raw key/door/switch/gate objects (the legacy analysis field),
* ``path_depth``: key-door or switch-gate *stages required by the BFS path*, and
* ``option_count``: required stages plus off-path/distractor choices.

That makes a D2 maze derived from M6 a five-option maze: K -> S -> K on the
solution path, plus a wrong key and an inactive switch.

Examples::

    python -m analysis.propose_maze_candidates --profiles all
    python -m analysis.propose_maze_candidates --profiles long_tail \
        --seed 17 --proposals 200 --keep 6

Outputs include a corpus feature table, ranked CSV/JSON lists, diagnostic PNGs,
and a Markdown summary. Re-running with another seed produces alternative lists.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import functools
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# Permit both ``python -m analysis.propose_maze_candidates`` and direct execution.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gridworld.baselines import plan_bfs_path  # noqa: E402
from gridworld.task_spec import TaskSpecification  # noqa: E402
from gridworld.task_validator import compute_difficulty  # noqa: E402


DEFAULT_MAZE_ROOT = REPO_ROOT / "ogbench/ogbench/procgen/maze_jsons"
DEFAULT_OUTPUT = REPO_ROOT / "analysis/candidate_mazes"

SIGNATURE_ORDER = ("none", "K", "S", "K->K", "K->S", "S->K", "K->S->K", "other")
SIGNATURE_COLORS = {
    "none": "#9aa0a6",
    "K": "#d95f02",
    "S": "#1b9e77",
    "K->K": "#e6ab02",
    "K->S": "#7570b3",
    "S->K": "#1f78b4",
    "K->S->K": "#e7298a",
    "other": "#222222",
}

REPORT_BANDS: tuple[tuple[str, int, int | None], ...] = (
    ("<10", 0, 9),
    ("10-15", 10, 15),
    ("16-19", 16, 19),
    ("20-25", 20, 25),
    ("26-39", 26, 39),
    ("40-59", 40, 59),
    ("60-79", 60, 79),
    ("80+", 80, None),
)

COMPARISON_BANDS: tuple[tuple[str, int, int | None], ...] = (
    ("short comparison (26-39)", 26, 39),
    ("medium comparison (40-59)", 40, 59),
    ("long comparison (60-81)", 60, 81),
    ("tail comparison (82+)", 82, None),
)


@dataclass(frozen=True)
class Profile:
    """Targets and soft constraints for one panel design."""

    name: str
    description: str
    length_segments: tuple[tuple[int, int, int], ...]
    signature_targets: tuple[tuple[str, int], ...]
    depth_targets: tuple[tuple[int, int], ...]
    distractor_target: int
    five_option_target: int
    b_target: int
    b_min: int
    matched_ks_target: int
    matched_depth2_target: int
    family_minimums: tuple[tuple[str, int], ...]
    early_le15_min: int
    early_20_25_min: int
    max_length_min: int
    comparison_band_min: int
    length_weight: float = 4.0
    signature_weight: float = 1.5
    depth_weight: float = 1.4
    family_weight: float = 1.0
    distractor_weight: float = 1.0

    # Cached: pure function of the frozen profile, but called on every one of
    # the ~proposals*iterations loss evaluations in the anneal loop. Callers
    # must not mutate the returned array.
    @functools.cache
    def targets(self) -> np.ndarray:
        pieces: list[np.ndarray] = []
        for lo, hi, count in self.length_segments:
            pieces.append(np.linspace(lo, hi, count))
        result = np.concatenate(pieces)
        if len(result) != 50:
            raise ValueError(f"profile {self.name!r} defines {len(result)} lengths, not 50")
        return np.sort(result)


PROFILES: dict[str, Profile] = {
    "balanced": Profile(
        name="balanced",
        description="Best overall compromise: two early clusters, broad mechanism coverage, and a long tail.",
        length_segments=((10, 15, 6), (20, 25, 6), (28, 39, 7), (42, 58, 9),
                         (61, 78, 9), (81, 105, 13)),
        signature_targets=(("none", 5), ("K", 6), ("S", 12), ("K->K", 5),
                           ("K->S", 6), ("S->K", 6), ("K->S->K", 10)),
        depth_targets=((0, 5), (1, 18), (2, 17), (3, 10)),
        distractor_target=18,
        five_option_target=3,
        b_target=10,
        b_min=8,
        matched_ks_target=6,
        matched_depth2_target=3,
        family_minimums=(("M1", 8), ("M2", 8), ("M3", 3), ("M4", 3),
                         ("M5", 3), ("M6", 3)),
        early_le15_min=2,
        early_20_25_min=2,
        max_length_min=106,
        comparison_band_min=2,
    ),
    "long_tail": Profile(
        name="long_tail",
        description="Maximizes 80+ action coverage and preserves the corpus maximum while retaining both early clusters.",
        length_segments=((10, 15, 4), (20, 25, 4), (28, 39, 6), (42, 58, 7),
                         (61, 79, 9), (82, 105, 20)),
        signature_targets=(("none", 5), ("K", 6), ("S", 12), ("K->K", 5),
                           ("K->S", 6), ("S->K", 6), ("K->S->K", 10)),
        depth_targets=((0, 5), (1, 18), (2, 17), (3, 10)),
        distractor_target=18,
        five_option_target=3,
        b_target=10,
        b_min=8,
        matched_ks_target=5,
        matched_depth2_target=2,
        family_minimums=(("M1", 8), ("M2", 8), ("M3", 2), ("M4", 2),
                         ("M5", 2), ("M6", 2)),
        early_le15_min=2,
        early_20_25_min=2,
        max_length_min=106,
        comparison_band_min=2,
        length_weight=5.2,
    ),
    "mechanism_rich": Profile(
        name="mechanism_rich",
        description="Favors depth-2/depth-3 chains, distractors, and five-option D2/M6 mazes.",
        length_segments=((10, 15, 5), (20, 25, 5), (28, 39, 7), (42, 58, 8),
                         (61, 79, 10), (82, 105, 15)),
        signature_targets=(("none", 3), ("K", 5), ("S", 11), ("K->K", 6),
                           ("K->S", 7), ("S->K", 7), ("K->S->K", 11)),
        depth_targets=((0, 3), (1, 16), (2, 20), (3, 11)),
        distractor_target=20,
        five_option_target=6,
        b_target=9,
        b_min=8,
        matched_ks_target=4,
        matched_depth2_target=3,
        family_minimums=(("M1", 4), ("M2", 4), ("M3", 3), ("M4", 3),
                         ("M5", 3), ("M6", 3)),
        early_le15_min=2,
        early_20_25_min=1,
        max_length_min=106,
        comparison_band_min=1,
        signature_weight=1.9,
        depth_weight=2.2,
        distractor_weight=4.0,
    ),
    "pairwise": Profile(
        name="pairwise",
        description="Prioritizes matched K-vs-S geometries and repeated depth-2 order comparisons.",
        length_segments=((10, 15, 6), (20, 25, 7), (28, 39, 8), (42, 58, 9),
                         (61, 79, 10), (82, 105, 10)),
        signature_targets=(("none", 4), ("K", 8), ("S", 16), ("K->K", 5),
                           ("K->S", 5), ("S->K", 6), ("K->S->K", 6)),
        depth_targets=((0, 4), (1, 24), (2, 16), (3, 6)),
        distractor_target=14,
        five_option_target=2,
        b_target=10,
        b_min=8,
        matched_ks_target=9,
        matched_depth2_target=4,
        family_minimums=(("M1", 8), ("M2", 8), ("M3", 4), ("M4", 4),
                         ("M5", 4), ("M6", 2)),
        early_le15_min=2,
        early_20_25_min=2,
        max_length_min=106,
        comparison_band_min=2,
        signature_weight=2.0,
        depth_weight=1.7,
    ),
}


def _position_tuple(obj) -> tuple[int, int]:
    return int(obj.position.x), int(obj.position.y)


def _corpus_fingerprint(paths: Sequence[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _path_mechanism_events(spec: TaskSpecification, labels: Sequence[str],
                           positions: Sequence[tuple[int, int]]) -> list[tuple[int, str, str]]:
    """Return ordered required barrier events as ``(action_index, K/S, id)``."""
    events: list[tuple[int, str, str]] = []
    for action_index, label in enumerate(labels):
        if label.startswith("open_door:"):
            events.append((action_index, "K", label.split(":", 1)[1]))

    # A switch action can occur far before its gate. A closed gate traversed by
    # the optimal path is the actual required stage, so locate first traversal.
    for gate in spec.mechanisms.gates:
        if gate.initial_state == "open":
            continue
        gate_pos = _position_tuple(gate)
        try:
            position_index = list(positions).index(gate_pos)
        except ValueError:
            continue
        events.append((max(0, position_index - 1), "S", gate.id))

    # A barrier should only count once even if the path revisits it.
    dedup: dict[tuple[str, str], tuple[int, str, str]] = {}
    for event in events:
        dedup.setdefault((event[1], event[2]), event)
    return sorted(dedup.values())


def _mechanism_option_counts(spec: TaskSpecification, path_depth: int) -> tuple[int, int, int, int]:
    """Return option, distractor, inactive-switch, unmatched-key counts."""
    doors_by_color = Counter(door.requires_key for door in spec.mechanisms.doors)
    keys_by_color = Counter(key.color for key in spec.mechanisms.keys)
    unmatched_keys = sum(
        max(0, count - doors_by_color.get(color, 0))
        for color, count in keys_by_color.items()
    )

    gate_ids = {gate.id for gate in spec.mechanisms.gates}
    active_switches = 0
    inactive_switches = 0
    controlled_gates: set[str] = set()
    for switch in spec.mechanisms.switches:
        valid_controls = set(switch.controls) & gate_ids
        if valid_controls:
            active_switches += 1
            controlled_gates.update(valid_controls)
        else:
            inactive_switches += 1

    unmatched_gates = sum(
        gate.initial_state != "open" and gate.id not in controlled_gates
        for gate in spec.mechanisms.gates
    )
    option_count = (
        len(spec.mechanisms.doors)
        + active_switches
        + inactive_switches
        + unmatched_keys
        + unmatched_gates
        + len(spec.mechanisms.blocks)
        + len(spec.mechanisms.teleporters)
    )
    distractor_count = max(0, option_count - path_depth)
    return option_count, distractor_count, inactive_switches, unmatched_keys


def _feature_row(path: Path, maze_root: Path) -> dict:
    payload = json.loads(path.read_text())
    spec = TaskSpecification.from_dict(payload)
    # One executable-planner run feeds both this module's path-signature
    # analysis (action labels + positions) and, via ``bfs_path=``, the repo's
    # difficulty authority. Beatability/optimal_steps/states_explored come
    # from ``compute_difficulty`` so panels can never disagree with
    # ``analysis.mazes.maze_features`` or the validator on action lengths.
    planned = plan_bfs_path(spec)
    difficulty = compute_difficulty(spec, bfs_path=planned)
    events = _path_mechanism_events(spec, planned.action_labels, planned.positions)
    path_symbols = [event[1] for event in events]
    signature = "->".join(path_symbols) if path_symbols else "none"
    if signature not in SIGNATURE_ORDER:
        signature = "other"
    option_count, distractor_count, inactive_switches, unmatched_keys = (
        _mechanism_option_counts(spec, len(events))
    )

    m = spec.mechanisms
    relative = path.relative_to(maze_root)
    family = relative.parts[0]
    width, height = spec.maze.dimensions
    metadata = payload.get("metadata") or {}
    stem_tokens = path.stem.split("_")
    topology = next((token for token in stem_tokens if token in {"corridor", "dense", "empty", "room"}),
                    metadata.get("wall_topology", "other"))
    # JSON ``seed`` is 0 in many ``*_1`` fixtures. The filename suffix is the
    # corpus' actual paired-layout variant and must define controlled matches.
    variant_token = stem_tokens[-1] if stem_tokens else ""
    variant = int(variant_token) if variant_token.isdigit() else payload.get("seed")
    comparison_cell = f"{width}x{height}:{topology}:{variant}"

    return {
        "maze_id": relative.with_suffix("").as_posix(),
        "source_path": path.relative_to(REPO_ROOT).as_posix(),
        "task_id": spec.task_id,
        "family": family,
        "family_group": family[:1],
        "is_b": family.startswith("B"),
        "width": width,
        "height": height,
        "grid_area": width * height,
        "topology": topology,
        "variant": variant,
        "comparison_cell": comparison_cell,
        "chain_pattern": metadata.get("chain_pattern", ""),
        "source_category": metadata.get("source_category", ""),
        "distractor_type": metadata.get("distractor_type", ""),
        "is_beatable": difficulty.is_beatable,
        "optimal_steps": difficulty.optimal_steps,
        "states_explored": difficulty.states_explored,
        "path_signature": signature,
        "path_depth": len(events),
        "path_key_stages": path_symbols.count("K"),
        "path_switch_stages": path_symbols.count("S"),
        "path_event_ids": ";".join(event[2] for event in events),
        "option_count": option_count,
        "distractor_count": distractor_count,
        "five_option": option_count >= 5 and len(events) >= 3 and distractor_count >= 2,
        "inactive_switches": inactive_switches,
        "unmatched_keys": unmatched_keys,
        "object_count": difficulty.mechanism_count,
        "n_keys": len(m.keys),
        "n_doors": len(m.doors),
        "n_switches": len(m.switches),
        "n_gates": len(m.gates),
        "known_length_risk": family == "D3",
    }


# Bump when _feature_row's semantics change (not just the corpus), so cached
# feature tables built under the old rules are rebuilt. v2: beatability/
# optimal_steps/states_explored/object_count sourced from compute_difficulty.
_FEATURE_VERSION = 2


def load_or_build_features(maze_root: Path, output_dir: Path, refresh: bool = False) -> pd.DataFrame:
    paths = sorted(maze_root.glob("*/*.json"))
    if not paths:
        raise FileNotFoundError(f"no maze JSONs found below {maze_root}")
    fingerprint = _corpus_fingerprint(paths, maze_root)
    cache_path = output_dir / "all_maze_features.csv"
    meta_path = output_dir / "all_maze_features.meta.json"
    if not refresh and cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (
            meta.get("fingerprint") == fingerprint
            and meta.get("feature_version") == _FEATURE_VERSION
        ):
            return pd.read_csv(cache_path, keep_default_na=False)

    rows = [_feature_row(path, maze_root) for path in paths]
    df = pd.DataFrame(rows).sort_values(["optimal_steps", "maze_id"]).reset_index(drop=True)
    duplicate_counts = df.groupby("task_id").maze_id.transform("count")
    df["duplicate_task_id"] = duplicate_counts > 1
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    meta_path.write_text(json.dumps({
        "fingerprint": fingerprint,
        "feature_version": _FEATURE_VERSION,
        "maze_root": str(maze_root),
        "maze_count": len(df),
        "duplicate_task_ids": sorted(df.loc[df.duplicate_task_id, "task_id"].unique()),
    }, indent=2) + "\n")
    return df


def _in_band(value: int, lo: int, hi: int | None) -> bool:
    return value >= lo and (hi is None or value <= hi)


def _counts(values: Iterable) -> Counter:
    return Counter(values)


def _matched_cells(selected: pd.DataFrame, families: Sequence[str]) -> int:
    by_family = {
        family: set(selected.loc[selected.family == family, "comparison_cell"])
        for family in families
    }
    if not by_family:
        return 0
    return len(set.intersection(*by_family.values()))


def _selection_loss(selected: pd.DataFrame, profile: Profile) -> float:
    """Soft objective; lower is better. Hard constraints live in the sampler."""
    steps = np.sort(selected.optimal_steps.to_numpy(dtype=float))
    targets = profile.targets()
    length_loss = float(np.mean(((steps - targets) / 5.0) ** 2))
    gaps = np.diff(steps)
    gap_loss = float(np.sum(np.maximum(0.0, gaps - 11.0) ** 2) / 25.0)

    signature_counts = _counts(selected.path_signature)
    signature_loss = sum(
        abs(signature_counts.get(signature, 0) - target)
        for signature, target in profile.signature_targets
    )
    depth_counts = _counts(int(v) for v in selected.path_depth)
    depth_loss = sum(abs(depth_counts.get(depth, 0) - target) for depth, target in profile.depth_targets)

    distractor_loss = abs(int((selected.distractor_count > 0).sum()) - profile.distractor_target)
    five_loss = max(0, profile.five_option_target - int(selected.five_option.sum())) * 3.0
    b_loss = abs(int(selected.is_b.sum()) - profile.b_target) * 4.0

    matched_ks = _matched_cells(selected, ("M1", "M2"))
    matched_depth2 = _matched_cells(selected, ("M3", "M4", "M5"))
    match_loss = max(0, profile.matched_ks_target - matched_ks) * 12.0
    match_loss += max(0, profile.matched_depth2_target - matched_depth2) * 8.0

    # Every core mechanism family should have replicates. D families are pooled
    # because each contains several source signatures.
    family_counts = _counts(selected.family)
    family_shortage = sum(max(0, 2 - family_counts.get(family, 0)) for family in (
        "M1", "M2", "M3", "M4", "M5", "M6", "D1", "D2", "D3"
    ))
    family_shortage += 5.0 * sum(
        max(0, minimum - family_counts.get(family, 0))
        for family, minimum in profile.family_minimums
    )

    # The corpus has only two <=15 anchors (9 and 11) and three 20-25 anchors.
    # Preserve them explicitly; otherwise a smoothness objective can discard the
    # very clusters the benchmark is intended to probe.
    early_shortage = 10.0 * max(0, profile.early_le15_min - int((selected.optimal_steps <= 15).sum()))
    early_shortage += 10.0 * max(
        0,
        profile.early_20_25_min - int(selected.optimal_steps.between(20, 25).sum()),
    )
    tail_max_shortage = 2.5 * max(0, profile.max_length_min - int(selected.optimal_steps.max()))

    # Reward repeated K/S observations in broad length bands. B1 stays visible
    # but does not substitute for the controlled M1-vs-M2 comparison.
    band_shortage = 0
    for _, lo, hi in COMPARISON_BANDS:
        band = selected[selected.optimal_steps.map(lambda x: _in_band(int(x), lo, hi))]
        band_shortage += 9.0 * max(
            0, profile.comparison_band_min - int((band.family == "M1").sum())
        )
        band_shortage += 9.0 * max(
            0, profile.comparison_band_min - int((band.family == "M2").sum())
        )

    duplicate_id_loss = int(selected.task_id.duplicated().sum()) * 1000.0
    return (
        profile.length_weight * length_loss
        + 0.9 * gap_loss
        + profile.signature_weight * signature_loss
        + profile.depth_weight * depth_loss
        + profile.distractor_weight * distractor_loss
        + five_loss
        + b_loss
        + match_loss
        + profile.family_weight * family_shortage
        + early_shortage
        + tail_max_shortage
        + 1.1 * band_shortage
        + duplicate_id_loss
    )


def _initial_selection(df: pd.DataFrame, profile: Profile, rng: random.Random) -> set[int]:
    """Length-aware randomized initialization with the required B1 floor."""
    eligible = df.index[df.is_beatable].tolist()
    b_indices = df.index[df.is_beatable & df.is_b].tolist()
    selected = set(rng.sample(b_indices, min(profile.b_min, len(b_indices))))
    selected_task_ids = set(df.loc[list(selected), "task_id"])

    def add_index(index: int) -> bool:
        task_id = df.at[index, "task_id"]
        if index in selected or task_id in selected_task_ids:
            return False
        selected.add(index)
        selected_task_ids.add(task_id)
        return True

    def add_matched_bundles(families: Sequence[str], count: int) -> None:
        cells_by_family = {
            family: set(df.loc[df.is_beatable & (df.family == family), "comparison_cell"])
            for family in families
        }
        common_cells = list(set.intersection(*cells_by_family.values())) if cells_by_family else []
        rng.shuffle(common_cells)
        bundles_added = 0
        for cell in common_cells:
            indices: list[int] = []
            for family in families:
                choices = df.index[
                    df.is_beatable & (df.family == family) & (df.comparison_cell == cell)
                ].tolist()
                rng.shuffle(choices)
                choice = next((idx for idx in choices if df.at[idx, "task_id"] not in selected_task_ids), None)
                if choice is None:
                    indices = []
                    break
                indices.append(choice)
            if not indices:
                continue
            for index in indices:
                add_index(index)
            bundles_added += 1
            if bundles_added >= count:
                break

    # Start inside the important interaction basins. A one-maze swap optimizer
    # otherwise has difficulty discovering a three-family matched triplet.
    add_matched_bundles(("M1", "M2"), profile.matched_ks_target)
    add_matched_bundles(("M3", "M4", "M5"), profile.matched_depth2_target)

    early_groups = (
        (df.index[df.is_beatable & (df.optimal_steps <= 15)].tolist(), profile.early_le15_min),
        (df.index[df.is_beatable & df.optimal_steps.between(20, 25)].tolist(), profile.early_20_25_min),
    )
    for choices, minimum in early_groups:
        rng.shuffle(choices)
        added = 0
        for index in choices:
            if add_index(index):
                added += 1
            if added >= minimum:
                break

    targets = list(profile.targets())
    rng.shuffle(targets)
    while len(selected) < 50:
        target = targets[len(selected) % len(targets)]
        choices = [
            idx for idx in eligible
            if idx not in selected and df.at[idx, "task_id"] not in selected_task_ids
        ]
        if not choices:
            raise RuntimeError("not enough unique, beatable task IDs for a 50-maze selection")
        sample = rng.sample(choices, min(18, len(choices)))
        idx = min(sample, key=lambda i: abs(float(df.at[i, "optimal_steps"]) - target)
                  + rng.random() * 3.0)
        add_index(idx)
    return selected


def _anneal(df: pd.DataFrame, profile: Profile, rng: random.Random,
            iterations: int) -> tuple[float, tuple[int, ...]]:
    selected = _initial_selection(df, profile, rng)
    eligible = set(df.index[df.is_beatable])
    current_loss = _selection_loss(df.loc[list(selected)], profile)
    best_loss, best = current_loss, tuple(sorted(selected))

    for iteration in range(iterations):
        out_idx = rng.choice(tuple(selected))
        if bool(df.at[out_idx, "is_b"]) and int(df.loc[list(selected), "is_b"].sum()) <= profile.b_min:
            continue
        candidates = tuple(eligible - selected)
        if not candidates:
            # Exactly 50 beatable mazes: the initial selection is the only
            # possible panel and there is nothing to swap in.
            break
        in_idx = rng.choice(candidates)
        in_task_id = df.at[in_idx, "task_id"]
        if any(df.at[idx, "task_id"] == in_task_id for idx in selected if idx != out_idx):
            continue

        proposal = set(selected)
        proposal.remove(out_idx)
        proposal.add(in_idx)
        proposal_loss = _selection_loss(df.loc[list(proposal)], profile)
        progress = iteration / max(1, iterations - 1)
        temperature = 2.5 * (0.04 / 2.5) ** progress
        if proposal_loss < current_loss or rng.random() < math.exp((current_loss - proposal_loss) / temperature):
            selected = proposal
            current_loss = proposal_loss
            if current_loss < best_loss:
                best_loss, best = current_loss, tuple(sorted(selected))
    return best_loss, best


def generate_proposals(df: pd.DataFrame, profile: Profile, seed: int,
                       proposals: int, iterations: int, keep: int) -> list[tuple[float, tuple[int, ...]]]:
    """Generate many restarts, then retain strong but non-identical panels."""
    raw: list[tuple[float, tuple[int, ...]]] = []
    for restart in range(proposals):
        rng = random.Random(seed + restart * 104729)
        raw.append(_anneal(df, profile, rng, iterations))
    raw.sort(key=lambda item: item[0])

    retained: list[tuple[float, tuple[int, ...]]] = []
    seen: set[tuple[int, ...]] = set()
    for item in raw:
        indices = item[1]
        if indices in seen:
            continue
        candidate_set = set(indices)
        # Retain alternatives that differ by at least four mazes.
        if any(len(candidate_set.symmetric_difference(set(other[1]))) < 8 for other in retained):
            continue
        seen.add(indices)
        retained.append(item)
        if len(retained) >= keep:
            break
    if not retained:
        raise RuntimeError(f"no proposal retained for profile {profile.name}")
    return retained


def _cluster_label(steps: int) -> str:
    for name, lo, hi in REPORT_BANDS:
        if _in_band(steps, lo, hi):
            return name
    return "<10"


def candidate_table(df: pd.DataFrame, indices: Sequence[int]) -> pd.DataFrame:
    selected = df.loc[list(indices)].sort_values(["optimal_steps", "path_depth", "maze_id"]).copy()
    selected.insert(0, "rank", range(1, len(selected) + 1))
    selected.insert(5, "length_cluster", selected.optimal_steps.map(_cluster_label))
    return selected


def _summary_dict(table: pd.DataFrame, profile: Profile, loss: float) -> dict:
    steps = table.optimal_steps.astype(int)
    gaps = steps.sort_values().diff().dropna()
    family_counts = _counts(table.family)
    signature_counts = _counts(table.path_signature)
    depth_counts = _counts(int(v) for v in table.path_depth)
    clusters = {
        name: int(steps.map(lambda value: _in_band(value, lo, hi)).sum())
        for name, lo, hi in REPORT_BANDS
    }
    comparisons = {}
    for name, lo, hi in COMPARISON_BANDS:
        band = table[table.optimal_steps.map(lambda value: _in_band(int(value), lo, hi))]
        comparisons[name] = {
            "M1_key": int((band.family == "M1").sum()),
            "M2_switch": int((band.family == "M2").sum()),
            "B1_switch": int((band.family == "B1").sum()),
            "depth2": int((band.path_depth == 2).sum()),
            "depth3": int((band.path_depth == 3).sum()),
        }
    return {
        "profile": profile.name,
        "description": profile.description,
        "loss": round(loss, 3),
        "maze_count": len(table),
        "length_min": int(steps.min()),
        "length_median": float(steps.median()),
        "length_max": int(steps.max()),
        "largest_adjacent_gap": int(gaps.max()) if len(gaps) else 0,
        "length_clusters": clusters,
        "families": dict(sorted(family_counts.items())),
        "path_signatures": {key: signature_counts.get(key, 0) for key in SIGNATURE_ORDER
                            if signature_counts.get(key, 0)},
        "path_depths": dict(sorted(depth_counts.items())),
        "distractor_mazes": int((table.distractor_count > 0).sum()),
        "five_option_mazes": int(table.five_option.sum()),
        "b_mazes": int(table.is_b.sum()),
        "matched_M1_M2_cells": _matched_cells(table, ("M1", "M2")),
        "matched_M3_M4_M5_cells": _matched_cells(table, ("M3", "M4", "M5")),
        "comparison_bands": comparisons,
        "duplicate_task_ids": sorted(table.loc[table.task_id.duplicated(False), "task_id"].unique()),
        "known_D3_length_risk_count": int(table.known_length_risk.sum()),
    }


def _plot_candidate(table: pd.DataFrame, summary: dict, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_curve, ax_complexity) = plt.subplots(
        2, 1, figsize=(13.5, 9.5), gridspec_kw={"height_ratios": (2.15, 1)}, constrained_layout=True
    )
    ax_curve.plot(table["rank"], table.optimal_steps, color="#b8b8b8", linewidth=1.2, zorder=1)
    for signature in SIGNATURE_ORDER:
        subset = table[table.path_signature == signature]
        if subset.empty:
            continue
        no_distractor = subset[subset.distractor_count == 0]
        distracted = subset[subset.distractor_count > 0]
        if not no_distractor.empty:
            ax_curve.scatter(no_distractor["rank"], no_distractor.optimal_steps,
                             s=45 + 28 * no_distractor.path_depth, color=SIGNATURE_COLORS[signature],
                             marker="o", edgecolor="white", linewidth=0.7, zorder=3)
        if not distracted.empty:
            ax_curve.scatter(distracted["rank"], distracted.optimal_steps,
                             s=55 + 28 * distracted.path_depth, color=SIGNATURE_COLORS[signature],
                             marker="X", edgecolor="#222222", linewidth=0.55, zorder=4)

    for _, row in table[table.is_b].iterrows():
        ax_curve.annotate("B", (row["rank"], row.optimal_steps), xytext=(0, 7),
                          textcoords="offset points", ha="center", fontsize=7, color="#333333")
    ax_curve.axhspan(10, 15, color="#76c7c0", alpha=0.08)
    ax_curve.axhspan(20, 25, color="#f2c14e", alpha=0.09)
    ax_curve.axhspan(80, max(108, int(table.optimal_steps.max()) + 3), color="#e76f51", alpha=0.06)
    ax_curve.set_xlim(0, 51)
    ax_curve.set_ylim(0, max(110, int(table.optimal_steps.max()) + 5))
    ax_curve.set_xlabel("candidate rank (sorted by BFS-estimated optimal actions)")
    ax_curve.set_ylabel("optimal action count")
    ax_curve.set_title(
        f"{summary['profile'].replace('_', ' ').title()} candidate: length curve and mechanisms\n"
        f"50 mazes | {summary['b_mazes']} B1 | {summary['five_option_mazes']} five-option | "
        f"max={summary['length_max']} | largest gap={summary['largest_adjacent_gap']}",
        fontsize=15, fontweight="bold",
    )

    # A deterministic vertical offset exposes overlapping points without changing values materially.
    offsets = np.array([((rank * 37) % 11 - 5) * 0.035 for rank in table["rank"]])
    for signature in SIGNATURE_ORDER:
        mask = table.path_signature == signature
        if not mask.any():
            continue
        markers = np.where(table.loc[mask, "distractor_count"].to_numpy() > 0, "X", "o")
        for marker in ("o", "X"):
            marker_mask = markers == marker
            sub = table.loc[mask].iloc[np.flatnonzero(marker_mask)]
            sub_offsets = offsets[np.flatnonzero(mask)][marker_mask]
            ax_complexity.scatter(sub.optimal_steps, sub.option_count + sub_offsets,
                                  s=48 + 22 * sub.path_depth, color=SIGNATURE_COLORS[signature],
                                  marker=marker, edgecolor="#222222" if marker == "X" else "white",
                                  linewidth=0.5, alpha=0.9)
    five_option_rows = table[table.five_option].sort_values("optimal_steps")
    if not five_option_rows.empty:
        # Label only the endpoints; full IDs live in the companion CSV/JSON and
        # labeling every point makes the dense 58/61 cluster unreadable.
        endpoint_rows = five_option_rows.iloc[[0, -1]].drop_duplicates("maze_id")
        for endpoint_number, (_, row) in enumerate(endpoint_rows.iterrows()):
            y_offset = 6 if endpoint_number == 0 else -13
            ax_complexity.annotate(f"five-option ({int(row.optimal_steps)} actions)",
                                   (row.optimal_steps, row.option_count),
                                   xytext=(5, y_offset), textcoords="offset points", fontsize=7)
    ax_complexity.set_xlabel("BFS-estimated optimal actions")
    ax_complexity.set_ylabel("mechanism options\n(path + distractors)")
    ax_complexity.set_ylim(-0.5, max(5.7, float(table.option_count.max()) + 0.7))

    signature_handles = [
        Line2D([0], [0], marker="o", linestyle="", label=signature.replace("->", "→"),
               markerfacecolor=SIGNATURE_COLORS[signature], markeredgecolor="white", markersize=8)
        for signature in SIGNATURE_ORDER if (table.path_signature == signature).any()
    ]
    shape_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="#555555", label="required path only",
               markerfacecolor="#bbbbbb", markersize=8),
        Line2D([0], [0], marker="X", linestyle="", color="#555555", label="has distractor option",
               markerfacecolor="#bbbbbb", markersize=8),
    ]
    ax_curve.legend(handles=signature_handles + shape_handles, loc="upper left", ncol=3,
                    fontsize=8, frameon=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_corpus(df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    beatable = df[df.is_beatable].sort_values(["optimal_steps", "maze_id"]).reset_index(drop=True)
    beatable["rank"] = np.arange(1, len(beatable) + 1)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax, hist) = plt.subplots(2, 1, figsize=(13.5, 8.5),
                                  gridspec_kw={"height_ratios": (2.2, 1)}, constrained_layout=True)
    ax.plot(beatable["rank"], beatable.optimal_steps, color="#c7c7c7", linewidth=1)
    for signature in SIGNATURE_ORDER:
        subset = beatable[beatable.path_signature == signature]
        if subset.empty:
            continue
        ax.scatter(subset["rank"], subset.optimal_steps,
                   s=20 + 10 * subset.option_count, color=SIGNATURE_COLORS[signature],
                   marker="o", alpha=0.8, edgecolor="none", label=signature.replace("->", "→"))
    ax.set_title(f"OGBench maze corpus: {len(beatable)} beatable JSONs", fontsize=15, fontweight="bold")
    ax.set_xlabel("corpus rank (sorted by BFS-estimated optimal actions)")
    ax.set_ylabel("optimal action count")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    bins = np.arange(0, max(111, int(beatable.optimal_steps.max()) + 6), 5)
    for signature in SIGNATURE_ORDER:
        values = beatable.loc[beatable.path_signature == signature, "optimal_steps"]
        if len(values):
            hist.hist(values, bins=bins, histtype="step", linewidth=2,
                      color=SIGNATURE_COLORS[signature], label=signature.replace("->", "→"))
    hist.axvspan(10, 15, color="#76c7c0", alpha=0.1)
    hist.axvspan(20, 25, color="#f2c14e", alpha=0.12)
    hist.axvspan(80, bins[-1], color="#e76f51", alpha=0.08)
    hist.set_xlabel("BFS-estimated optimal actions")
    hist.set_ylabel("maze count / 5-action bin")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _markdown_summary(corpus: pd.DataFrame, result_rows: list[dict]) -> str:
    beatable = corpus[corpus.is_beatable]
    longest = beatable.sort_values(["optimal_steps", "option_count"], ascending=False).head(15)
    five = beatable[beatable.five_option].sort_values(["optimal_steps", "maze_id"])
    duplicates = beatable[beatable.duplicate_task_id].sort_values(["task_id", "maze_id"])
    lines = [
        "# OGBench 50-maze candidate panels",
        "",
        f"Corpus: {len(corpus)} JSON files, {len(beatable)} BFS-beatable. Length means executable ",
        "turn/move/pickup/toggle actions from the current BFS planner.",
        "",
        "`path_depth` counts required key-door/switch-gate stages; `option_count` adds distractors. ",
        "D3 lengths are flagged because `analysis/.cache/phase1_notes.md` documents a known ",
        "BFS over-count on a dense dead-end decoy-key maze (31 reported versus a 23-step legal solve).",
        "",
        "## Candidate overview",
        "",
        "| profile | min | median | max | largest gap | 10-15 | 20-25 | 80+ | depth 2 | depth 3 | distractor | five-option | B1 | M1/M2 matched |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result_rows:
        depths = row["path_depths"]
        clusters = row["length_clusters"]
        lines.append(
            f"| {row['profile']} | {row['length_min']} | {row['length_median']:.1f} | "
            f"{row['length_max']} | {row['largest_adjacent_gap']} | {clusters.get('10-15', 0)} | "
            f"{clusters.get('20-25', 0)} | {clusters.get('80+', 0)} | "
            f"{depths.get(2, depths.get('2', 0))} | {depths.get(3, depths.get('3', 0))} | "
            f"{row['distractor_mazes']} | {row['five_option_mazes']} | {row['b_mazes']} | "
            f"{row['matched_M1_M2_cells']} |"
        )

    for row in result_rows:
        lines.extend(["", f"## {row['profile'].replace('_', ' ').title()}", "", row["description"], ""])
        cluster_text = ", ".join(f"{name}: {count}" for name, count in row["length_clusters"].items())
        signature_text = ", ".join(f"{name.replace('->', '→')}: {count}"
                                   for name, count in row["path_signatures"].items())
        family_text = ", ".join(f"{name}: {count}" for name, count in row["families"].items())
        lines.extend([
            f"- Length clusters — {cluster_text}.",
            f"- Path signatures — {signature_text}.",
            f"- Families — {family_text}.",
            f"- Matched geometry cells: M1-vs-M2 = {row['matched_M1_M2_cells']}; "
            f"M3-vs-M4-vs-M5 = {row['matched_M3_M4_M5_cells']}.",
            f"- Known-risk D3 lengths in panel: {row['known_D3_length_risk_count']}.",
        ])
        risk_lines = []
        if row["largest_adjacent_gap"] > 11:
            risk_lines.append(f"length curve has a {row['largest_adjacent_gap']}-action adjacent gap")
        if row["five_option_mazes"] < 2:
            risk_lines.append("fewer than two five-option mazes")
        weak_bands = [name for name, counts in row["comparison_bands"].items()
                      if counts["M1_key"] < 2 or counts["M2_switch"] < 2]
        if weak_bands:
            risk_lines.append("fewer than two controlled M1/M2 observations in " + ", ".join(weak_bands))
        if risk_lines:
            lines.append("- Risks — " + "; ".join(risk_lines) + ".")
        else:
            lines.append("- Risks — no automated coverage warning; D3 optimality caveat still applies.")

    lines.extend(["", "## Corpus longest tail", "",
                  "| steps | depth | options | distractors | signature | maze |", "|---:|---:|---:|---:|---|---|"])
    for _, row in longest.iterrows():
        lines.append(f"| {int(row.optimal_steps)} | {int(row.path_depth)} | {int(row.option_count)} | "
                     f"{int(row.distractor_count)} | {row.path_signature.replace('->', '→')} | `{row.maze_id}` |")

    lines.extend(["", "## Five-option corpus", "",
                  "These are the 3-on-path + 2-distractor cases.", "",
                  "| steps | maze |", "|---:|---|"])
    for _, row in five.iterrows():
        lines.append(f"| {int(row.optimal_steps)} | `{row.maze_id}` |")

    if not duplicates.empty:
        lines.extend(["", "## Data-integrity warning", "",
                      "The following distinct files share a `task_id`; no generated panel includes both:", ""])
        for task_id, group in duplicates.groupby("task_id"):
            lines.append(f"- `{task_id}`: " + ", ".join(f"`{maze_id}`" for maze_id in group.maze_id))
    return "\n".join(lines) + "\n"


def _parse_profiles(value: str) -> list[str]:
    if value == "all":
        return list(PROFILES)
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in PROFILES]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown profiles: {', '.join(unknown)}")
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maze-root", type=Path, default=DEFAULT_MAZE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profiles", default="all",
                        help="all or comma-separated: " + ", ".join(PROFILES))
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--proposals", type=int, default=30,
                        help="independent randomized restarts per profile")
    parser.add_argument("--iterations", type=int, default=400,
                        help="swap iterations per randomized restart")
    parser.add_argument("--keep", type=int, default=3,
                        help="alternative lists retained per profile")
    parser.add_argument("--refresh-features", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile_names = _parse_profiles(args.profiles)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = load_or_build_features(args.maze_root.resolve(), args.output_dir.resolve(),
                                      refresh=args.refresh_features)
    if not args.no_plots:
        _plot_corpus(features, args.output_dir / "corpus_overview.png")

    best_summaries: list[dict] = []
    for profile_offset, profile_name in enumerate(profile_names):
        profile = PROFILES[profile_name]
        proposals = generate_proposals(
            features,
            profile,
            seed=args.seed + profile_offset * 1_000_003,
            proposals=args.proposals,
            iterations=args.iterations,
            keep=args.keep,
        )
        for candidate_number, (loss, indices) in enumerate(proposals, start=1):
            table = candidate_table(features, indices)
            summary = _summary_dict(table, profile, loss)
            summary["candidate_number"] = candidate_number
            stem = f"{profile_name}_{candidate_number:02d}"
            table.to_csv(args.output_dir / f"{stem}.csv", index=False)
            (args.output_dir / f"{stem}.json").write_text(json.dumps({
                "summary": summary,
                "mazes": table.to_dict(orient="records"),
            }, indent=2) + "\n")
            if not args.no_plots:
                _plot_candidate(table, summary, args.output_dir / f"{stem}.png")
            if candidate_number == 1:
                best_summaries.append(summary)

    (args.output_dir / "candidate_summary.json").write_text(
        json.dumps(best_summaries, indent=2) + "\n"
    )
    (args.output_dir / "README.md").write_text(_markdown_summary(features, best_summaries))
    print(f"Wrote {len(profile_names) * args.keep} candidate lists to {args.output_dir}")
    for summary in best_summaries:
        print(
            f"{summary['profile']:14s} max={summary['length_max']:3d} "
            f"80+={summary['length_clusters']['80+']:2d} depth3={summary['path_depths'].get(3, 0):2d} "
            f"five={summary['five_option_mazes']:2d} B1={summary['b_mazes']:2d} "
            f"gap={summary['largest_adjacent_gap']:2d} loss={summary['loss']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
