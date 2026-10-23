from __future__ import annotations

from pathlib import Path

import deploy


def test_deploy_exposes_repo_root():
    assert (deploy.REPO_ROOT / "pyproject.toml").is_file()
