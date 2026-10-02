"""Confirm API keys with a tiny 'Hello, how are you?' round-trip per provider."""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Dict, List, Optional

from deploy.envkeys import PROVIDER_ENV, resolve_keys

HELLO = "Hello, how are you?"


def hello_roundtrip(provider: str, key: str) -> Dict[str, object]:
    """Send one short turn through the real agent for ``provider``.

    Raises on transport/auth failure; the caller turns that into a 'failed'
    result. Imports are local so a missing optional dep for one provider does
    not break the other."""
    messages = [{"role": "user", "content": HELLO}]
    if provider == "anthropic":
        from interface.agents.claude import ClaudeAnthropicAgent, ClaudeAnthropicConfig

        agent = ClaudeAnthropicAgent(config=ClaudeAnthropicConfig(max_tokens=32), api_key=key)
    elif provider == "kimi":
        from interface.agents.kimi_k26 import KimiK26Agent, KimiK26Config

        agent = KimiK26Agent(config=KimiK26Config(max_tokens=32), api_key=key)
    else:
        raise ValueError(f"Unknown provider: {provider!r}")
    reply = agent(messages)
    return {"provider": provider, "ok": True, "reply": reply, "usage": agent.last_usage}


def run_checks(
    keys: Dict[str, Optional[str]],
    roundtrip: Callable[[str, str], Dict[str, object]] = hello_roundtrip,
) -> tuple[List[Dict[str, object]], int]:
    """Round-trip each present key; skip absent ones. Exit code 1 iff a present
    key failed (an absent key is not an error)."""
    results: List[Dict[str, object]] = []
    exit_code = 0
    for provider in PROVIDER_ENV:
        key = keys.get(provider)
        if not key:
            results.append({"provider": provider, "status": "skipped", "detail": "no key found"})
            continue
        try:
            res = roundtrip(provider, key)
            results.append(
                {
                    "provider": provider,
                    "status": "ok",
                    "detail": str(res.get("reply", ""))[:80],
                    "usage": res.get("usage"),
                }
            )
        except Exception as exc:  # noqa: BLE001 - report any provider failure, keep going
            results.append({"provider": provider, "status": "failed", "detail": str(exc)})
            exit_code = 1
    return results, exit_code


def _print(results: List[Dict[str, object]]) -> None:
    for r in results:
        print(f"[{str(r['status']).upper():7}] {r['provider']:9} {r.get('detail', '')}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Round-trip API keys with a 'Hello, how are you?' turn."
    )
    parser.add_argument("--provider", choices=sorted(PROVIDER_ENV), help="Check only this provider.")
    args = parser.parse_args(argv)

    keys = resolve_keys()
    if args.provider:
        keys = {args.provider: keys.get(args.provider)}
    results, code = run_checks(keys)
    _print(results)
    if all(r["status"] == "skipped" for r in results):
        print("No API keys found (env / .env / api_key.txt). Nothing to check.")
    return code


if __name__ == "__main__":
    sys.exit(main())
