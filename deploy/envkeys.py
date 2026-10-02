"""Resolve provider API keys from env / .env / api_key.txt (in that precedence)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping, Optional

from deploy import REPO_ROOT

PROVIDER_ENV: Dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "kimi": "MOONSHOT_API_KEY",
}
# api_key.txt line order (matches interface/agents/api_keys.py): line 1 -> anthropic, line 2 -> kimi.
_API_KEY_FILE_ORDER = ("anthropic", "kimi")


def parse_dotenv(path: Path) -> Dict[str, str]:
    """Parse simple ``KEY=VALUE`` lines. Ignores blanks/comments; strips an
    optional ``export`` prefix and surrounding quotes/whitespace."""
    values: Dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        val = val.strip().strip('"').strip("'").strip()
        if key:
            values[key] = val
    return values


def _api_key_file_keys(repo_root: Path) -> Dict[str, str]:
    path = repo_root / "api_key.txt"
    if not path.is_file():
        return {}
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return {provider: value for provider, value in zip(_API_KEY_FILE_ORDER, lines)}


def resolve_keys(
    repo_root: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Optional[str]]:
    """Return ``{"anthropic": key|None, "kimi": key|None}`` with precedence
    env > .env > api_key.txt."""
    root = repo_root or REPO_ROOT
    environ = os.environ if env is None else env
    dotenv = parse_dotenv(root / ".env")
    file_keys = _api_key_file_keys(root)
    resolved: Dict[str, Optional[str]] = {}
    for provider, env_name in PROVIDER_ENV.items():
        value = (environ.get(env_name) or "").strip()
        if not value:
            value = (dotenv.get(env_name) or "").strip()
        if not value:
            value = (file_keys.get(provider) or "").strip()
        resolved[provider] = value or None
    return resolved
