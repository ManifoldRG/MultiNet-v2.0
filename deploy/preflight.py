# deploy/preflight.py
"""Per-node readiness checks for the distributed run. Run on each node before launch."""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from deploy.cluster import Cluster, load_cluster
from deploy.envkeys import resolve_keys

ROLES = ("coordinator", "worker", "api-client")


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    fatal: bool = True


@dataclass
class Probes:
    gpu: Callable[[], tuple[bool, str]]
    vram_free_gb: Callable[[], float]
    weights_present: Callable[[str], bool]
    kernels: Callable[[], Dict[str, bool]]
    http_ok: Callable[[str], bool]
    port_free: Callable[[str, int], bool]
    gsutil_present: Callable[[], bool]
    keys: Callable[[], Dict[str, Optional[str]]]


def provider_for_group(model_group: str) -> str:
    return "kimi" if "kimi" in model_group.lower() else "anthropic"


# --- default probe implementations (used by main; tests inject fakes) ---

def _run(cmd: List[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return p.returncode, (p.stdout + p.stderr)
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)


def _probe_gpu() -> tuple[bool, str]:
    code, out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    lines = [ln for ln in out.strip().splitlines() if ln.strip()]
    return (code == 0 and bool(lines), lines[0] if lines else out.strip())


def _probe_vram_free_gb() -> float:
    code, out = _run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
    vals = [float(x) for x in out.strip().splitlines() if x.strip().replace(".", "", 1).isdigit()]
    return (max(vals) / 1024.0) if (code == 0 and vals) else 0.0


def _probe_weights_present(model_group: str) -> bool:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    if not hub.is_dir():
        return False
    if "qwen" not in model_group.lower():
        return True  # API groups have no local weights
    return any("qwen" in p.name.lower() for p in hub.glob("models--*"))


def _probe_kernels() -> Dict[str, bool]:
    status: Dict[str, bool] = {}
    for mod in ("flash_attn", "fla", "causal_conv1d"):
        try:
            __import__(mod)
            status[mod] = True
        except Exception:  # noqa: BLE001 - any import error => kernel unavailable
            status[mod] = False
    return status


def _probe_http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return 200 <= resp.status < 500
    except urllib.error.HTTPError as exc:
        return exc.code < 500
    except Exception:  # noqa: BLE001
        return False


def _probe_port_free(host: str, port: int) -> bool:
    bind_host = "0.0.0.0" if host not in ("127.0.0.1", "localhost") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, port))
            return True
        except OSError:
            return False


def default_probes() -> Probes:
    return Probes(
        gpu=_probe_gpu,
        vram_free_gb=_probe_vram_free_gb,
        weights_present=_probe_weights_present,
        kernels=_probe_kernels,
        http_ok=_probe_http_ok,
        port_free=_probe_port_free,
        gsutil_present=lambda: shutil.which("gsutil") is not None,
        keys=resolve_keys,
    )


# --- role checks ---

def _check_worker(cluster: Cluster, node: str, probes: Probes, min_vram_gb: float) -> List[CheckResult]:
    worker = cluster.worker(node)
    out: List[CheckResult] = []

    gpu_ok, gpu_detail = probes.gpu()
    out.append(CheckResult("gpu", gpu_ok, gpu_detail))

    free = probes.vram_free_gb()
    out.append(CheckResult("vram_free", free >= min_vram_gb, f"{free:.1f} GB free (need >= {min_vram_gb:.0f})"))

    out.append(
        CheckResult("weights", probes.weights_present(worker.model_group), f"cache for {worker.model_group}")
    )

    kernels = probes.kernels()
    missing = [k for k, ok in kernels.items() if not ok]
    out.append(
        CheckResult("fast_kernels", not missing, ("all active" if not missing else f"missing: {missing}"), fatal=False)
    )

    url = f"{cluster.coordinator_url}/status"
    out.append(CheckResult("coordinator_reachable", probes.http_ok(url), url))
    return out


def _check_coordinator(cluster: Cluster, probes: Probes) -> List[CheckResult]:
    out: List[CheckResult] = []
    free = probes.port_free(cluster.coordinator.host, cluster.coordinator.port)
    out.append(
        CheckResult("port", free, f"{cluster.coordinator.port} " + ("free" if free else "busy (already serving?)"), fatal=False)
    )
    plan = Path(cluster.coordinator.artifacts_root) / "distributed" / "job_plan.json"
    out.append(CheckResult("plan_prepared", plan.is_file(), str(plan), fatal=False))
    if cluster.storage_config:
        out.append(CheckResult("gsutil", probes.gsutil_present(), "needed for storage mirroring", fatal=False))
    return out


def _check_api_client(cluster: Cluster, probes: Probes) -> List[CheckResult]:
    keys = probes.keys()
    out: List[CheckResult] = []
    for client in cluster.api_clients:
        provider = provider_for_group(client.model_group)
        present = bool(keys.get(provider))
        out.append(CheckResult(f"key:{client.name}", present, f"{provider} key {'present' if present else 'MISSING'}"))
    if not cluster.api_clients:
        out.append(CheckResult("api_clients", True, "none declared", fatal=False))
    return out


def run_preflight(
    cluster: Cluster,
    node: Optional[str],
    role: str,
    probes: Probes,
    *,
    min_vram_gb: float = 55.0,
) -> tuple[List[CheckResult], int]:
    if role == "worker":
        if not node:
            raise ValueError("worker preflight requires --node")
        results = _check_worker(cluster, node, probes, min_vram_gb)
    elif role == "coordinator":
        results = _check_coordinator(cluster, probes)
    elif role == "api-client":
        results = _check_api_client(cluster, probes)
    else:
        raise ValueError(f"Unknown role {role!r}; expected one of {ROLES}")
    exit_code = 1 if any(r.fatal and not r.ok for r in results) else 0
    return results, exit_code


def _print(role: str, node: Optional[str], results: List[CheckResult]) -> None:
    label = f"{role}" + (f" [{node}]" if node else "")
    print(f"Preflight: {label}")
    for r in results:
        mark = "PASS" if r.ok else ("WARN" if not r.fatal else "FAIL")
        print(f"  [{mark}] {r.name}: {r.detail}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Per-node preflight checks for a distributed run.")
    parser.add_argument("--cluster", required=True, help="Path to cluster inventory JSON.")
    parser.add_argument("--role", required=True, choices=ROLES)
    parser.add_argument("--node", help="Worker name (required for --role worker).")
    parser.add_argument("--min-vram-gb", type=float, default=55.0)
    args = parser.parse_args(argv)

    cluster = load_cluster(args.cluster)
    results, code = run_preflight(cluster, args.node, args.role, default_probes(), min_vram_gb=args.min_vram_gb)
    _print(args.role, args.node, results)
    print("RESULT:", "PASS" if code == 0 else "FAIL")
    return code


if __name__ == "__main__":
    sys.exit(main())
