"""Derive a distributed-run VM topology from a run_config's ``models`` dict.

Pure logic (no cloud calls), unit-tested. A CLI wrapper (added in Task 4) lets
launch_distributed.sh consume the same function.
"""
from __future__ import annotations

import re
from typing import Any

# provider -> required credential env var (only API providers need one).
_CREDENTIAL_BY_PROVIDER = {"kimi": "MOONSHOT_API_KEY", "claude": "ANTHROPIC_API_KEY"}


def sanitize_vm_name(raw: str) -> str:
    """Coerce ``raw`` to an RFC1035 GCE instance name: lowercase, only [a-z0-9-],
    no leading/trailing '-', <=63 chars."""
    s = re.sub(r"[^a-z0-9-]+", "-", raw.lower())
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:63].rstrip("-")


def _classify(model: dict[str, Any]) -> str:
    if model.get("hardware_profile") == "local-gpu":
        return "gpu"
    if str(model.get("provider")) in {"kimi", "claude"}:
        return "api"
    # Fall through: any other hardware_profile is treated as an API worker.
    return "api"


def derive_topology(run_config: dict[str, Any], run_id: str,
                    gpu_worker_count: int | None = None) -> dict[str, Any]:
    models = run_config.get("models", {}) or {}
    workers: list[dict[str, Any]] = []
    creds: set[str] = set()
    has_gpu = False
    for key, model in models.items():
        if not model.get("group"):
            raise ValueError(f"model {key!r} is missing required 'group' (needed for VM names)")
        kind = _classify(model)
        group = str(model.get("group"))
        provider = str(model.get("provider"))
        count = int(model.get("worker_count", 1))
        if kind == "gpu" and gpu_worker_count is not None:
            count = int(gpu_worker_count)
        if kind == "gpu":
            has_gpu = True
        cred = _CREDENTIAL_BY_PROVIDER.get(provider)
        if cred:
            creds.add(cred)
        for i in range(count):
            workers.append({
                "name": sanitize_vm_name(f"{run_id}-{group}-{i}"),
                "kind": kind,
                "model_group": group,
                "provider": provider,
                "model": str(model.get("model", "")),
            })
    return {
        "coordinator": {"name": sanitize_vm_name(f"{run_id}-coord")},
        "workers": workers,
        "has_gpu": has_gpu,
        "required_credentials": sorted(creds),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import os
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Derive distributed-run VM topology.")
    parser.add_argument("run_config", help="Path to a run_config JSON.")
    parser.add_argument("run_id", help="Run id (used for VM names).")
    parser.add_argument("--check-credentials", action="store_true",
                        help="Exit 3 if a required API credential env var is unset.")
    args = parser.parse_args(argv)

    cfg = json.loads(Path(args.run_config).read_text())
    gpu_wc_env = os.environ.get("QWEN_WORKER_COUNT")
    gpu_wc = int(gpu_wc_env) if gpu_wc_env else None
    topology = derive_topology(cfg, args.run_id, gpu_worker_count=gpu_wc)

    if args.check_credentials:
        missing = [c for c in topology["required_credentials"] if not os.environ.get(c)]
        if missing:
            print(f"Missing required credential env var(s): {', '.join(missing)}", file=sys.stderr)
            return 3

    print(json.dumps(topology, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
