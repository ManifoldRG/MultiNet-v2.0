"""Load local API keys from gitignored ``api_key.txt`` at repo root."""

from __future__ import annotations

import os
from pathlib import Path

_API_KEY_FILE = Path(__file__).resolve().parents[2] / "api_key.txt"


def ensure_api_keys_from_file() -> None:
    """If ``api_key.txt`` exists: line 1 → Anthropic, line 2 → Moonshot."""
    if not _API_KEY_FILE.is_file():
        return
    lines = [line.strip() for line in _API_KEY_FILE.read_text().splitlines() if line.strip()]
    if lines and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = lines[0]
    if len(lines) > 1 and not os.environ.get("MOONSHOT_API_KEY"):
        os.environ["MOONSHOT_API_KEY"] = lines[1]
