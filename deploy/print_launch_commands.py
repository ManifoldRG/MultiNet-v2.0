"""Print the exact distributed-role command to run on each node (no SSH)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

from deploy.cluster import Cluster, load_cluster

PIPELINE = "multinet-run-pipeline"


def build_commands(
    cluster: Cluster,
    run_config: str,
    manifest: str,
    conditions: Optional[str] = None,
    seeds: str = "0",
    prompt_variant: Optional[str] = None,
) -> Dict[str, str]:
    c = cluster.coordinator
    url = cluster.coordinator_url
    storage = f" --storage-config {cluster.storage_config}" if cluster.storage_config else ""
    cond = f' --conditions "{conditions}"' if conditions else ""
    pv = f" --prompt-variant {prompt_variant}" if prompt_variant else ""

    cmds: Dict[str, str] = {}
    cmds["coordinator-prepare"] = (
        f"{PIPELINE} --distributed-role coordinator-prepare "
        f"--run-config {run_config} --manifest {manifest}{cond}{pv} "
        f"--seeds {seeds} --artifacts-root {c.artifacts_root} --run-set-id {c.run_set_id}"
    )
    cmds["coordinator-serve"] = (
        f"{PIPELINE} --distributed-role coordinator-serve "
        f"--artifacts-root {c.artifacts_root} --host {c.host} --port {c.port}{storage}"
    )
    for w in cluster.workers:
        cmds[f"worker:{w.name}"] = (
            f"{PIPELINE} --distributed-role worker --coordinator-url {url} "
            f"--artifacts-root {c.artifacts_root} --model-group {w.model_group} "
            f"--hardware-profile {w.hardware_profile}"
        )
    for ac in cluster.api_clients:
        # --max-units 0 drains every pending unit for the group; without it the
        # role defaults to a single unit and the API run is left incomplete.
        cmds[f"api-client:{ac.name}"] = (
            f"{PIPELINE} --distributed-role coordinator-run-api-client "
            f"--artifacts-root {c.artifacts_root} --model-group {ac.model_group} --max-units 0"
        )
    cmds["coordinator-finalize"] = (
        f"{PIPELINE} --distributed-role coordinator-finalize "
        f"--artifacts-root {c.artifacts_root} --run-set-id {c.run_set_id}{storage}"
    )
    return cmds


def plan_model_groups(cluster: Cluster) -> Optional[Set[str]]:
    plan_path = Path(cluster.coordinator.artifacts_root) / "distributed" / "job_plan.json"
    if not plan_path.is_file():
        return None
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    return {str(m.get("model_group")) for m in (data.get("models") or {}).values()}


def group_mismatches(cluster: Cluster) -> List[str]:
    plan_groups = plan_model_groups(cluster)
    if plan_groups is None:
        return []
    return [
        f"model_group {g!r} is in the cluster inventory but not in the prepared plan {sorted(plan_groups)}"
        for g in sorted(cluster.model_groups())
        if g not in plan_groups
    ]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Print per-node launch commands for a distributed run.")
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--conditions")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--prompt-variant")
    args = parser.parse_args(argv)

    cluster = load_cluster(args.cluster)
    cmds = build_commands(
        cluster, args.run_config, args.manifest, args.conditions, args.seeds, args.prompt_variant
    )
    print("# Coordinator node:")
    print("  " + cmds["coordinator-prepare"])
    print("  " + cmds["coordinator-serve"])
    for ac in cluster.api_clients:
        print(f"  # api-client {ac.name} (runs on coordinator):")
        print("  " + cmds[f"api-client:{ac.name}"])
    print("  # after workers finish:")
    print("  " + cmds["coordinator-finalize"])
    for w in cluster.workers:
        print(f"\n# Worker node {w.name} ({w.host}):")
        print("  " + cmds[f"worker:{w.name}"])

    for warning in group_mismatches(cluster):
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
