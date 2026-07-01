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
    # Default unknown profiles to gpu only if they declared local-gpu; otherwise api.
    return "api"


def derive_topology(run_config: dict[str, Any], run_id: str) -> dict[str, Any]:
    models = run_config.get("models", {}) or {}
    workers: list[dict[str, Any]] = []
    creds: set[str] = set()
    has_gpu = False
    for model in models.values():
        kind = _classify(model)
        group = str(model.get("group"))
        provider = str(model.get("provider"))
        count = int(model.get("worker_count", 1))
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
            })
    return {
        "coordinator": {"name": sanitize_vm_name(f"{run_id}-coord")},
        "workers": workers,
        "has_gpu": has_gpu,
        "required_credentials": sorted(creds),
    }
