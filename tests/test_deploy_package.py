from __future__ import annotations

from pathlib import Path

import deploy


def test_deploy_exposes_repo_root():
    assert (deploy.REPO_ROOT / "pyproject.toml").is_file()


def test_future_directions_doc_flags_vllm():
    doc = deploy.REPO_ROOT / "docs" / "future_directions.md"
    assert doc.is_file()
    text = doc.read_text(encoding="utf-8").lower()
    assert "vllm" in text
    assert "sglang" in text
