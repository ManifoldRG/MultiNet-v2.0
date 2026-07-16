#!/usr/bin/env python3
"""Package generated maze candidate lists as illustrated, self-contained folders.

For every ``*_NN.json`` candidate emitted by
``analysis.propose_maze_candidates``, this script creates::

    analysis/candidate_mazes/sets/<candidate>/
      SUMMARY.md
      candidate.csv
      candidate.json
      candidate_curve.png
      images/001_<family>__<maze>.png ... images/050_...

The per-maze PNGs are copied from the canonical OGBench render corpus. The
summary explains the experimental intent, length/depth/mechanism distribution,
matched comparisons, distractor and long-tail choices, risks, and every maze's
role in the panel.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Iterable, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CANDIDATE_DIR = REPO_ROOT / "analysis/candidate_mazes"
DEFAULT_OUTPUT_DIR = DEFAULT_CANDIDATE_DIR / "sets"
DEFAULT_RENDER_DIR = REPO_ROOT / "ogbench/ogbench/procgen/maze_images"

CANDIDATE_PATTERN = re.compile(
    r"^(balanced|long_tail|mechanism_rich|pairwise)_(\d{2})$"
)

CLUSTER_ORDER = ("<10", "10-15", "16-19", "20-25", "26-39", "40-59", "60-79", "80+")

PROFILE_INTRO = {
    "balanced": (
        "The balanced candidate is designed as the general-purpose benchmark panel. "
        "It repeats controlled single-stage "
        "key-door and switch-gate comparisons across several spatial scales, includes "
        "ordered depth-2 and depth-3 chains, and still allocates substantial mass to the "
        "80+ action tail. It is the closest thing to a single panel that can support "
        "difficulty-curve, mechanism, distractor, and long-horizon analyses together."
    ),
    "long_tail": (
        "The long-tail candidate treats path length as the primary stress variable. It "
        "retains the controlled key-versus-switch comparisons of the balanced design, but "
        "moves many discretionary slots into 14x14 corridor and "
        "dense layouts above 80 actions. This makes it well suited to measuring planning "
        "degradation, context or memory pressure, compounding navigation errors, and the "
        "interaction between mechanisms and very long execution horizons."
    ),
    "mechanism_rich": (
        "The mechanism-rich candidate focuses on complexity that cannot be explained by "
        "length alone. It deliberately increases distractor-bearing D1/D2/D3 fixtures, "
        "depth-2 ordering problems, depth-3 K->S->K chains, and the five-option cases that "
        "combine three required stages with a wrong key and an inactive switch. It keeps "
        "one pure K/S control pair in each feasible length region, but uses the released "
        "slots for mechanism and distractor stress rather than maximum replication."
    ),
    "pairwise": (
        "The pairwise candidate is optimized for controlled causal comparisons. It "
        "selects repeated M1 key-door and M2 switch-gate mazes on identical geometry and "
        "variant cells, together with matched M3/M4/M5 triplets that compare K->S, S->K, "
        "and K->K ordering. It preserves B1 structural variants, some "
        "distractors, depth-3 fixtures, and a long tail, but prioritizes clean within-layout "
        "contrasts over maximal distractor prevalence."
    ),
}

SIGNATURE_DESCRIPTION = {
    "none": "navigation only; no mechanism stage is required on the optimal path",
    "K": "one required key-door stage",
    "S": "one required switch-gate stage",
    "K->K": "two sequential key-door stages",
    "K->S": "a key-door stage followed by a switch-gate stage",
    "S->K": "a switch-gate stage followed by a key-door stage",
    "K->S->K": "the deepest available chain: key-door, switch-gate, then key-door",
    "other": "a mechanism sequence outside the core benchmark signatures",
}

FAMILY_DESCRIPTION = {
    "S1": "minimal empty-room spatial baseline",
    "S2": "short winding-corridor navigation baseline",
    "S3": "medium corridor navigation baseline",
    "S4": "medium dense-layout navigation baseline",
    "S5": "long 14x14 corridor navigation baseline",
    "S6": "long 14x14 dense navigation baseline",
    "B1": "white-switch B1 structural variant retained as a distinct switch condition",
    "M1": "controlled single key-door condition",
    "M2": "controlled single switch-gate condition",
    "M3": "controlled K->S depth-2 condition",
    "M4": "controlled S->K depth-2 condition",
    "M5": "controlled K->K depth-2 condition",
    "M6": "controlled K->S->K depth-3 condition",
    "D1": "wrong-key distractor condition",
    "D2": "wrong-key plus inactive-switch distractor condition",
    "D3": "dead-end yellow key-door distractor condition",
}

CLUSTER_DESCRIPTION = {
    "<10": "the absolute shortest navigation anchor",
    "10-15": "the requested early navigation anchor",
    "16-19": "the currently empty bridge region in the corpus",
    "20-25": "the second early cluster, still before mechanism comparisons begin",
    "26-39": "short mechanism-capable mazes",
    "40-59": "medium-length mazes where ordering and distractor effects become separable",
    "60-79": "long mazes bridging the medium regime to the extreme tail",
    "80+": "the extreme long-horizon tail dominated by 14x14 structures",
}


def candidate_paths(candidate_dir: Path) -> list[Path]:
    """Return candidate JSONs, excluding aggregate summary files."""
    return sorted(
        path
        for path in candidate_dir.glob("*_[0-9][0-9].json")
        if CANDIDATE_PATTERN.match(path.stem)
    )


def _image_name(row: pd.Series) -> str:
    safe_id = str(row.maze_id).replace("/", "__")
    return f"{int(row['rank']):03d}_{safe_id}.png"


def _render_source(row: pd.Series, render_dir: Path) -> Path:
    return render_dir / f"{row.maze_id}.png"


def _fmt_signature(value: str) -> str:
    return str(value).replace("->", "→")


def _fmt_counts(values: Iterable[str], order: Sequence[str] | None = None) -> str:
    counts = Counter(values)
    keys = list(order) if order else sorted(counts)
    return ", ".join(f"{key}: {counts[key]}" for key in keys if counts.get(key, 0))


def _cluster_count_text(table: pd.DataFrame) -> str:
    counts = Counter(table.length_cluster)
    return ", ".join(f"{cluster}: {counts.get(cluster, 0)}" for cluster in CLUSTER_ORDER)


def _matched_groups(table: pd.DataFrame, families: Sequence[str]) -> list[tuple[str, dict[str, pd.Series]]]:
    groups: list[tuple[str, dict[str, pd.Series]]] = []
    for cell, cell_rows in table.groupby("comparison_cell"):
        family_rows: dict[str, pd.Series] = {}
        for family in families:
            matches = cell_rows[cell_rows.family == family]
            if matches.empty:
                break
            family_rows[family] = matches.iloc[0]
        if len(family_rows) == len(families):
            groups.append((str(cell), family_rows))
    return sorted(groups, key=lambda item: min(int(row.optimal_steps) for row in item[1].values()))


def _profile_variant_comment(profile: str, candidate_number: int, table: pd.DataFrame) -> str:
    tail = int((table.optimal_steps >= 80).sum())
    distractors = int((table.distractor_count > 0).sum())
    depth_three = int((table.path_depth == 3).sum())
    if candidate_number == 1:
        prefix = "This is the lowest-loss retained variant for the profile."
    elif candidate_number == 2:
        prefix = "This is the second retained alternative, chosen to differ by at least four mazes from variant 01."
    else:
        prefix = "This is the third retained alternative, providing another materially different panel under the same design goals."
    return (
        f"{prefix} Its realized allocation contains {tail} mazes at 80+ actions, "
        f"{distractors} mazes with at least one distractor option, and {depth_three} "
        "depth-3 mazes. The exact choices below should therefore be treated as a designed "
        "experimental panel, not merely a random sample from the corpus."
    )


def _headline_table(table: pd.DataFrame) -> list[str]:
    steps = table.optimal_steps.astype(int)
    return [
        "| measure | value |",
        "|---|---:|",
        f"| mazes | {len(table)} |",
        f"| optimal-action range | {steps.min()}–{steps.max()} |",
        f"| median optimal actions | {steps.median():.1f} |",
        f"| largest adjacent length gap | {int(steps.sort_values().diff().max())} |",
        f"| B1 mazes | {int(table.is_b.sum())} |",
        f"| depth-2 mazes | {int((table.path_depth == 2).sum())} |",
        f"| depth-3 mazes | {int((table.path_depth == 3).sum())} |",
        f"| distractor mazes | {int((table.distractor_count > 0).sum())} |",
        f"| five-option mazes | {int(table.five_option.sum())} |",
        f"| 80+ action mazes | {int((steps >= 80).sum())} |",
        f"| D3 length-risk mazes | {int(table.known_length_risk.sum())} |",
    ]


def _comparison_section(table: pd.DataFrame) -> list[str]:
    lines = [
        "## Controlled comparisons",
        "",
        "### Single-stage key versus switch",
        "",
        "M1/M2 pairs below share size, topology, and filename variant. Their one-action "
        "length differences largely reflect the executable pickup/toggle semantics rather "
        "than different spatial geometry.",
        "",
    ]
    pairs = _matched_groups(table, ("M1", "M2"))
    if pairs:
        lines.extend(["| matched cell | M1 key maze | steps | M2 switch maze | steps |", "|---|---|---:|---|---:|"])
        for cell, rows in pairs:
            lines.append(
                f"| `{cell}` | `{rows['M1'].maze_id}` | {int(rows['M1'].optimal_steps)} | "
                f"`{rows['M2'].maze_id}` | {int(rows['M2'].optimal_steps)} |"
            )
    else:
        lines.append("No exact M1/M2 geometry pair is present in this variant.")

    lines.extend([
        "",
        "### Depth-2 ordering triplets",
        "",
        "M3/M4/M5 triplets hold geometry fixed while changing the required ordering among "
        "K→S, S→K, and K→K. These are the cleanest fixtures for separating mechanism type "
        "from dependency depth.",
        "",
    ])
    triplets = _matched_groups(table, ("M3", "M4", "M5"))
    if triplets:
        lines.extend([
            "| matched cell | M3 K→S | steps | M4 S→K | steps | M5 K→K | steps |",
            "|---|---|---:|---|---:|---|---:|",
        ])
        for cell, rows in triplets:
            lines.append(
                f"| `{cell}` | `{rows['M3'].maze_id}` | {int(rows['M3'].optimal_steps)} | "
                f"`{rows['M4'].maze_id}` | {int(rows['M4'].optimal_steps)} | "
                f"`{rows['M5'].maze_id}` | {int(rows['M5'].optimal_steps)} |"
            )
    else:
        lines.append("No complete M3/M4/M5 geometry triplet is present in this variant.")
    return lines


def _special_cases_section(table: pd.DataFrame) -> list[str]:
    lines = [
        "## Distractors, five-option cases, and B1 structure",
        "",
        f"This panel contains {int((table.distractor_count > 0).sum())} distractor mazes. "
        "D1 adds a wrong key, D2 adds both a wrong key and an inactive switch, and D3 "
        "places a yellow key-door option in a dead-end branch. Distractor counts are kept "
        "separate from path depth: a D2/M6 maze is depth 3 but has five total options.",
        "",
    ]
    five = table[table.five_option].sort_values("optimal_steps")
    if not five.empty:
        lines.extend([
            "### Five-option mazes",
            "",
            "| rank | steps | maze | composition |",
            "|---:|---:|---|---|",
        ])
        for _, row in five.iterrows():
            lines.append(
                f"| {int(row['rank'])} | {int(row.optimal_steps)} | `{row.maze_id}` | "
                "K→S→K required + wrong key + inactive switch |"
            )
        lines.append("")

    b_rows = table[table.is_b].sort_values("optimal_steps")
    lines.extend([
        "### B1 selection",
        "",
        "B1 is treated as a protected structural stratum rather than interchangeable with "
        "M2. The chosen B1 mazes are:",
        "",
        "| rank | steps | size/topology | maze |",
        "|---:|---:|---|---|",
    ])
    for _, row in b_rows.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {int(row.optimal_steps)} | "
            f"{int(row.width)}x{int(row.height)} {row.topology} | `{row.maze_id}` |"
        )
    return lines


def _long_tail_section(table: pd.DataFrame) -> list[str]:
    tail = table[table.optimal_steps >= 80].sort_values("optimal_steps")
    lines = [
        "## Long-tail allocation",
        "",
        f"The panel contains {len(tail)} mazes at 80+ estimated optimal actions. These "
        "fixtures test whether errors grow with execution horizon, whether mechanism costs "
        "compound with navigation length, and whether corridor versus dense topology changes "
        "the model's ability to preserve a plan.",
        "",
        "| rank | steps | depth | options | signature | topology | maze |",
        "|---:|---:|---:|---:|---|---|---|",
    ]
    for _, row in tail.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {int(row.optimal_steps)} | {int(row.path_depth)} | "
            f"{int(row.option_count)} | {_fmt_signature(row.path_signature)} | {row.topology} | "
            f"`{row.maze_id}` |"
        )
    return lines


def _risks_section(table: pd.DataFrame) -> list[str]:
    d3_count = int(table.known_length_risk.sum())
    lines = [
        "## Interpretation risks and gaps",
        "",
        "1. The corpus does not contain replicated mechanism mazes at the requested earliest "
        "lengths. Only the 9/11-action S1 anchors occupy the first cluster, there are no "
        "16–19-action mazes, and the 20–25-action fixtures require no on-path mechanism. "
        "Controlled mechanism comparisons begin at 26 actions.",
        f"2. This panel contains {d3_count} D3 maze(s). Prior analysis documents at least "
        "one D3 case where the BFS estimate is 31 actions but a legal 23-step model solve "
        "exists. D3 values should be treated as estimated lengths pending planner/runtime "
        "reconciliation.",
        "3. Length, topology, and mechanism density are not fully factorial in the source "
        "corpus. The matched-pair and matched-triplet analyses should be preferred when "
        "making causal mechanism claims.",
        "4. Two source D2 files share one task ID. The candidate generator enforces unique "
        "task IDs, so this panel does not include both, but downstream manifests should still "
        "use the packaged source path as the audit key.",
    ]
    return lines


def _maze_role(row: pd.Series) -> str:
    family = str(row.family)
    signature = str(row.path_signature)
    family_text = FAMILY_DESCRIPTION.get(family, f"{family} family fixture")
    signature_text = SIGNATURE_DESCRIPTION.get(signature, SIGNATURE_DESCRIPTION["other"])
    return f"This is a {family_text}. Its optimal path is {signature_text}."


def _why_selected(row: pd.Series, profile: str) -> str:
    cluster = str(row.length_cluster)
    base = f"It represents {CLUSTER_DESCRIPTION.get(cluster, cluster)}"
    if profile == "long_tail" and int(row.optimal_steps) >= 80:
        return base + " and directly contributes to the panel's high-horizon emphasis."
    if profile == "mechanism_rich" and int(row.distractor_count) > 0:
        return base + " while increasing the panel's off-path choice and distractor load."
    if profile == "pairwise" and str(row.family) in {"M1", "M2", "M3", "M4", "M5"}:
        return base + " and participates in the controlled within-layout comparison design."
    if bool(row.is_b):
        return base + " while preserving the distinct B1 switch structure."
    return base + " and helps maintain the intended smooth difficulty curve."


def _maze_catalog(table: pd.DataFrame, profile: str) -> list[str]:
    lines = [
        "## Illustrated maze-by-maze walkthrough",
        "",
        "The catalog is ordered by BFS-estimated optimal action count. Marker terminology "
        "matches the candidate-curve plot: path depth counts required stages, while options "
        "include off-path distractors.",
        "",
    ]
    for cluster in CLUSTER_ORDER:
        cluster_rows = table[table.length_cluster == cluster].sort_values("rank")
        if cluster_rows.empty:
            continue
        lines.extend([f"### {cluster} actions", "", CLUSTER_DESCRIPTION[cluster].capitalize() + ".", ""])
        for _, row in cluster_rows.iterrows():
            image_name = _image_name(row)
            source_rel = Path("../../../..") / str(row.source_path)
            lines.extend([
                f"#### {int(row['rank']):02d}. `{row.maze_id}` — {int(row.optimal_steps)} actions",
                "",
                f"![{row.maze_id} maze render](images/{image_name})",
                "",
                f"- **Structure:** {int(row.width)}x{int(row.height)} {row.topology}, "
                f"family {row.family}, variant {row.variant}.",
                f"- **Mechanism path:** {_fmt_signature(row.path_signature)}; depth "
                f"{int(row.path_depth)}, {int(row.option_count)} total option(s), "
                f"{int(row.distractor_count)} distractor option(s).",
                f"- **Experimental role:** {_maze_role(row)}",
                f"- **Why selected:** {_why_selected(row, profile)}",
            ])
            if bool(row.five_option):
                lines.append(
                    "- **Five-option stress case:** three required stages plus a wrong key "
                    "and an inactive switch."
                )
            if bool(row.known_length_risk):
                lines.append(
                    "- **Length caveat:** D3 dead-end fixture; the BFS estimate may be inflated "
                    "relative to legal runtime execution."
                )
            lines.extend([f"- **Source:** [JSON]({source_rel.as_posix()})", ""])
    return lines


def build_summary(stem: str, table: pd.DataFrame) -> str:
    match = CANDIDATE_PATTERN.match(stem)
    if match is None:
        raise ValueError(f"invalid candidate stem: {stem}")
    profile, number_text = match.groups()
    candidate_number = int(number_text)
    title = stem.replace("_", " ").title()
    signature_counts = _fmt_counts(
        (_fmt_signature(value) for value in table.path_signature),
        order=("none", "K", "S", "K→K", "K→S", "S→K", "K→S→K", "other"),
    )
    family_counts = _fmt_counts(table.family)
    has_s1_anchors = bool((table.family == "S1").any())
    if has_s1_anchors:
        anchor_text = (
            "The two earliest clusters are intentionally navigation controls. Mechanism "
            "comparisons start in the 26–39 band and are carried through medium, long, "
            "and tail regimes."
        )
    else:
        anchor_text = (
            "This exploratory variant deliberately omits the 9/11-action S1 empty rooms. "
            "Their depth-0 control role is replaced by the 77/89-action S5 corridor pair, "
            "shifting two observations into long-horizon navigation without introducing a "
            "mechanism confound. Mechanism comparisons still begin in the 26–39 band."
        )

    lines = [
        f"# {title}: illustrated 50-maze candidate set",
        "",
        PROFILE_INTRO[profile],
        "",
        _profile_variant_comment(profile, candidate_number, table),
        "",
        "![Candidate optimal-length curve](candidate_curve.png)",
        "",
        "## Headline composition",
        "",
        *_headline_table(table),
        "",
        f"**Length clusters:** {_cluster_count_text(table)}.",
        "",
        f"**Path signatures:** {signature_counts}.",
        "",
        f"**Families:** {family_counts}.",
        "",
        anchor_text + " "
        "The candidate curve above shows mechanism signature by color, distractor-bearing "
        "mazes with X markers, and B1 fixtures with a `B` label.",
        "",
        *_comparison_section(table),
        "",
        *_special_cases_section(table),
        "",
        *_long_tail_section(table),
        "",
        *_risks_section(table),
        "",
        *_maze_catalog(table, profile),
    ]
    return "\n".join(lines).rstrip() + "\n"


def package_candidate(candidate_json: Path, candidate_dir: Path, output_dir: Path,
                      render_dir: Path) -> dict:
    payload = json.loads(candidate_json.read_text())
    table = pd.DataFrame(payload["mazes"]).sort_values("rank").reset_index(drop=True)
    stem = candidate_json.stem
    set_dir = output_dir / stem
    images_dir = set_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    missing: list[str] = []
    for _, row in table.iterrows():
        source = _render_source(row, render_dir)
        destination = images_dir / _image_name(row)
        if not source.exists():
            missing.append(str(source))
            continue
        shutil.copy2(source, destination)
        copied.append(destination.name)
    if missing:
        sample = "\n".join(f"  - {path}" for path in missing[:10])
        raise FileNotFoundError(f"missing {len(missing)} canonical render(s):\n{sample}")

    # Candidate revisions can change rank-prefixed filenames. Remove only stale
    # generated PNGs in this package, leaving all other artifacts untouched.
    expected_images = set(copied)
    for old_image in images_dir.glob("*.png"):
        if old_image.name not in expected_images:
            old_image.unlink()

    source_csv = candidate_dir / f"{stem}.csv"
    source_plot = candidate_dir / f"{stem}.png"
    shutil.copy2(candidate_json, set_dir / "candidate.json")
    shutil.copy2(source_csv, set_dir / "candidate.csv")
    shutil.copy2(source_plot, set_dir / "candidate_curve.png")
    (set_dir / "SUMMARY.md").write_text(build_summary(stem, table))
    (set_dir / "image_manifest.json").write_text(json.dumps({
        "candidate": stem,
        "image_count": len(copied),
        "images": copied,
    }, indent=2) + "\n")
    return {
        "candidate": stem,
        "profile": CANDIDATE_PATTERN.match(stem).group(1),
        "image_count": len(copied),
        "set_dir": set_dir,
        "summary": payload["summary"],
    }


def build_index(packages: Sequence[dict], output_dir: Path) -> None:
    lines = [
        "# Illustrated candidate-set packages",
        "",
        "Each folder contains 50 canonical OGBench maze renders, a detailed illustrated "
        "walkthrough, the candidate CSV/JSON, the optimal-length curve, and an image manifest.",
        "",
        "| candidate | images | median | 80+ | depth 2 | depth 3 | distractors | five-option | B1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for package in packages:
        summary = package["summary"]
        depths = summary["path_depths"]
        lines.append(
            f"| [{package['candidate']}]({package['candidate']}/SUMMARY.md) | "
            f"{package['image_count']} | {summary['length_median']:.1f} | "
            f"{summary['length_clusters']['80+']} | {depths.get('2', depths.get(2, 0))} | "
            f"{depths.get('3', depths.get(3, 0))} | {summary['distractor_mazes']} | "
            f"{summary['five_option_mazes']} | {summary['b_mazes']} |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--render-dir", type=Path, default=DEFAULT_RENDER_DIR)
    parser.add_argument(
        "--sets",
        default="all",
        help="all or comma-separated candidate stems, e.g. balanced_01,long_tail_02",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = candidate_paths(args.candidate_dir)
    if args.sets != "all":
        wanted = {item.strip() for item in args.sets.split(",") if item.strip()}
        paths = [path for path in paths if path.stem in wanted]
        missing_names = wanted - {path.stem for path in paths}
        if missing_names:
            raise SystemExit("candidate set(s) not found: " + ", ".join(sorted(missing_names)))
    if not paths:
        raise SystemExit(f"no candidate JSONs found in {args.candidate_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    packages = []
    for path in paths:
        package = package_candidate(path, args.candidate_dir, args.output_dir, args.render_dir)
        packages.append(package)
        try:
            display_dir = package["set_dir"].relative_to(REPO_ROOT)
        except ValueError:
            display_dir = package["set_dir"]
        print(
            f"{package['candidate']}: {package['image_count']} images -> "
            f"{display_dir}"
        )
    build_index(packages, args.output_dir)
    print(f"Packaged {len(packages)} candidate sets in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
