"""Resume an infra-killed episode from its archive.

One gate for every worker type (serial, lockstep, local run_pipeline): the
killed run's recorded actions are replayed against a fresh backend and checked
step by step (``resume_stepper``), and only then does the agent continue. A
divergence, another code SHA, or a chat mode whose state is not checkpointed
refuses the resume (``ResumeRefused``) instead of continuing on a guess.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_NAME = "resume_checkpoint.json"


class ResumeRefused(RuntimeError):
    """The killed run cannot be resumed automatically; the unit fails terminally."""


@functools.lru_cache(maxsize=1)
def current_code_sha() -> str:
    """The code this process runs: the verified-sync sentinel on fleet VMs, else
    git HEAD (``-dirty`` with uncommitted tracked changes), else ``unknown``."""
    deployed = REPO_ROOT / ".deployed_sha"
    if deployed.is_file():
        return deployed.read_text(encoding="utf-8").strip() or "unknown"
    try:
        sha = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        diff = subprocess.run(["git", "-C", str(REPO_ROOT), "diff", "HEAD"],
                              capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return _sha_with_dirt(sha, diff)


def _sha_with_dirt(sha: str, diff: str) -> str:
    """HEAD, plus a digest of uncommitted tracked changes so two different dirty
    trees never compare equal."""
    if not diff.strip():
        return sha
    return f"{sha}-dirty-{hashlib.sha256(diff.encode()).hexdigest()[:12]}"


def unpack_archive(archive_b64: str, dest: Path) -> Path:
    """Extract a coordinator-sent run archive into a fresh ``dest``."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(archive_b64)), mode="r:gz") as tar:
        for member in tar.getmembers():
            path = Path(member.name)
            if member.issym() or member.islnk() or path.is_absolute() or ".." in path.parts:
                raise ResumeRefused(f"unsafe member in the resume archive: {member.name}")
        tar.extractall(dest)
    return dest


def resumed_stepper(
    runner,
    archive_dir: Path,
    *,
    expected_inputs_hash: Optional[str],
    resume_count: int,
    label: str,
) -> tuple[Any, dict[str, Any]]:
    """Replay the killed run in ``archive_dir`` on ``runner`` (fresh, not reset)
    and return the stepper at the killed query boundary plus the provenance to
    stamp on the finished episode."""
    from interface.episode_checkpoint import resume_stepper
    from scripts.resume_from_archive import build_checkpoint_payload, rehydrate_agent_messages

    episode = json.loads((archive_dir / "episode.json").read_text(encoding="utf-8"))
    run_inputs = json.loads((archive_dir / "run_inputs.json").read_text(encoding="utf-8"))
    if expected_inputs_hash is not None and run_inputs.get("inputs_hash") != expected_inputs_hash:
        raise ResumeRefused(
            f"refusing to resume {label}: the archive's inputs_hash {run_inputs.get('inputs_hash')!r} "
            f"is not this unit's {expected_inputs_hash!r}"
        )
    as_run = run_inputs.get("code_sha") or "unrecorded"
    here = current_code_sha()
    if as_run in ("unrecorded", "unknown") or as_run != here:
        raise ResumeRefused(
            f"refusing to resume {label} automatically: it ran on code {as_run}, this worker runs {here}. "
            "Replay-verify checks the environment, not the prompts, so a cross-sha resume is a manual job: "
            f"run scripts/resume_from_archive.py with a worktree of {as_run}."
        )
    if runner.config.chat_history != "stateless":
        raise ResumeRefused(
            f"refusing to resume {label}: only chat_history='stateless' resumes automatically for now "
            f"(got {runner.config.chat_history!r}; its chat state is not checkpointed)"
        )
    try:
        payload = build_checkpoint_payload(episode)
    except SystemExit as exc:  # not an infra kill: nothing to resume
        raise ResumeRefused(f"refusing to resume {label}: {exc}") from None
    rehydrate_agent_messages(payload["transcript"], archive_dir)
    checkpoint = archive_dir / CHECKPOINT_NAME
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    try:
        stepper = resume_stepper(checkpoint, runner=runner)
    except ValueError as exc:
        raise ResumeRefused(f"refusing to resume {label}: replay diverged from the archived run: {exc}") from None

    killed_at = payload["query_count"]
    provenance = {
        "resumed_from": {"unit": label, "end_reason": episode.get("end_reason"), "killed_at_query": killed_at},
        "resume": {
            "as_run_sha": as_run,
            "resume_sha": here,
            "replay_verified": True,
            "resume_query_index": killed_at,
            "resume_count": int(resume_count),
        },
    }
    return stepper, provenance
