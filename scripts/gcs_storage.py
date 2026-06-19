"""gsutil-backed mirroring of coordinator artifacts to a Cloud Storage bucket.

The distributed coordinator stores artifacts on its own (possibly ephemeral)
local disk. To survive a coordinator loss, verified per-unit runs and the final
aggregate are mirrored to a shared GCS bucket via ``gsutil``. Configuration comes
from a JSON file (``--storage-config``); when no bucket is set, mirroring is a
no-op so local/dev runs are unaffected.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class StorageConfig:
    bucket: Optional[str] = None  # destination prefix, e.g. gs://my-bucket/runs
    enabled: bool = True
    gsutil: str = "gsutil"

    @property
    def active(self) -> bool:
        return bool(self.enabled and self.bucket)

    @classmethod
    def from_file(cls, path: str | Path) -> "StorageConfig":
        import json

        data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            bucket=data.get("bucket") or data.get("gcs_bucket"),
            enabled=bool(data.get("enabled", True)),
            gsutil=str(data.get("gsutil", "gsutil")),
        )


def gcs_destination(bucket: str, *parts: str) -> str:
    """Join a gs:// prefix with path parts, trimming stray slashes."""
    base = bucket.rstrip("/")
    suffix = "/".join(str(p).strip("/") for p in parts if str(p).strip("/"))
    return f"{base}/{suffix}" if suffix else base


def mirror_to_bucket(
    local_path: str | Path,
    dest_uri: str,
    *,
    gsutil: str = "gsutil",
    max_attempts: int = 3,
    timeout: Optional[float] = 600.0,
    run: Callable[..., Any] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Copy a file or directory to ``dest_uri`` via gsutil, with bounded retries.

    Directories are mirrored with ``rsync -r`` (idempotent, so re-running at
    finalize is safe); single files use ``cp``. Raises RuntimeError if every
    attempt fails.
    """
    local_path = Path(local_path)
    if local_path.is_dir():
        cmd = [gsutil, "-m", "rsync", "-r", str(local_path), dest_uri]
    else:
        cmd = [gsutil, "cp", str(local_path), dest_uri]

    last_stderr = ""
    for attempt in range(1, max_attempts + 1):
        result = run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return
        last_stderr = getattr(result, "stderr", "") or ""
        if attempt < max_attempts:
            sleep(2 ** (attempt - 1))
    raise RuntimeError(
        f"gsutil mirror to {dest_uri} failed after {max_attempts} attempts: {last_stderr.strip()}"
    )
