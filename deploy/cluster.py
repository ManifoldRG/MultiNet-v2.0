"""Load and validate the single cluster inventory file (deploy/cluster.*.json)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set


class ClusterConfigError(ValueError):
    """Raised when a cluster inventory file is missing or malformed."""


@dataclass
class Coordinator:
    host: str
    port: int
    artifacts_root: str
    run_set_id: str


@dataclass
class Worker:
    name: str
    host: str
    model_group: str
    hardware_profile: str
    venv: Optional[str] = None


@dataclass
class ApiClient:
    name: str
    model_group: str
    runs_on: str = "coordinator"


@dataclass
class Cluster:
    coordinator: Coordinator
    workers: List[Worker] = field(default_factory=list)
    api_clients: List[ApiClient] = field(default_factory=list)
    storage_config: Optional[str] = None

    @property
    def coordinator_url(self) -> str:
        return f"http://{self.coordinator.host}:{self.coordinator.port}"

    def model_groups(self) -> Set[str]:
        groups = {w.model_group for w in self.workers}
        groups.update(c.model_group for c in self.api_clients)
        return groups

    def worker(self, name: str) -> Worker:
        for w in self.workers:
            if w.name == name:
                return w
        raise ClusterConfigError(f"No worker named {name!r} in cluster.")


def _require(d: dict, key: str, ctx: str):
    if not isinstance(d, dict) or key not in d:
        raise ClusterConfigError(f"{ctx} is missing required field {key!r}.")
    return d[key]


def load_cluster(path: str | Path) -> Cluster:
    path = Path(path)
    if not path.is_file():
        raise ClusterConfigError(f"Cluster file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ClusterConfigError(f"Cluster file {path} is not valid JSON: {exc}") from exc

    cr = _require(raw, "coordinator", "cluster")
    coordinator = Coordinator(
        host=str(_require(cr, "host", "coordinator")),
        port=int(_require(cr, "port", "coordinator")),
        artifacts_root=str(_require(cr, "artifacts_root", "coordinator")),
        run_set_id=str(_require(cr, "run_set_id", "coordinator")),
    )
    workers = [
        Worker(
            name=str(_require(w, "name", "worker")),
            host=str(_require(w, "host", "worker")),
            model_group=str(_require(w, "model_group", "worker")),
            hardware_profile=str(_require(w, "hardware_profile", "worker")),
            venv=(str(w["venv"]) if w.get("venv") else None),
        )
        for w in raw.get("workers", [])
    ]
    api_clients = [
        ApiClient(
            name=str(_require(c, "name", "api_client")),
            model_group=str(_require(c, "model_group", "api_client")),
            runs_on=str(c.get("runs_on", "coordinator")),
        )
        for c in raw.get("api_clients", [])
    ]
    names = [w.name for w in workers] + [c.name for c in api_clients]
    if len(names) != len(set(names)):
        raise ClusterConfigError(f"Duplicate node names in cluster: {names}")
    return Cluster(
        coordinator=coordinator,
        workers=workers,
        api_clients=api_clients,
        storage_config=(str(raw["storage_config"]) if raw.get("storage_config") else None),
    )
