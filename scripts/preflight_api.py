"""Launch preflight: every API key and account a run-config uses must work
before any VM starts (next-run-fixes item 2).

Each API model gets one tiny live request through its real agent, with its own
model id, so an invalidated key (GPT-6 Astra's 401 on 2026-10-13), a model the
key cannot use and an exhausted account all fail here. Moonshot also exposes a
balance API: it must cover the model's spend cap (or a $1 floor). Anthropic and
OpenAI expose no balance to a standard API key (credit and cost reports need an
admin key), so for them the probe is the check.

    python -m scripts.preflight_api --run-config RC [--run-config RC ...]

Exits 1 if any model fails. SKIP_API_PREFLIGHT=1 skips it (dev only, loudly).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from interface.agents.billing import BillingExhausted

API_PROVIDERS = ("claude", "kimi", "openai")
MIN_BALANCE_USD = 1.0
MOONSHOT_BALANCE_URL = "https://api.moonshot.ai/v1/users/me/balance"
_PROMPT = [{"role": "user", "content": "Reply with the word OK."}]


def _probe(provider: str, cfg: dict[str, Any]) -> None:
    """One tiny request through the run's own agent and model (cents at most)."""
    from scripts.run_pipeline import _build_agent_from_spec

    small = {k: v for k, v in cfg.items() if k not in ("spend_cap_usd", "service_tier", "effort")}
    small.update({"timeout": 120, "max_attempts": 2})
    if provider == "openai":
        # max_completion_tokens covers reasoning too: leave room for a visible reply.
        small.update({"max_tokens": 256, "reasoning_effort": "low"})
    else:
        small.update({"max_tokens": 16, "enable_thinking": False})
    agent, _ = _build_agent_from_spec(provider, small)
    agent(_PROMPT)


def _moonshot_balance() -> float:
    req = urllib.request.Request(
        MOONSHOT_BALANCE_URL, headers={"Authorization": f"Bearer {os.environ.get('MOONSHOT_API_KEY', '')}"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode())
    return float(payload["data"]["available_balance"])


def preflight(
    run_config: dict[str, Any],
    *,
    probe: Callable[[str, dict[str, Any]], None] = _probe,
    balance: Callable[[], float] = _moonshot_balance,
) -> list[dict[str, str]]:
    rows = []
    for name, cfg in run_config.get("models", {}).items():
        provider = str(cfg.get("provider") or "").lower()
        row = {"model": name, "provider": provider, "status": "ok", "detail": ""}
        rows.append(row)
        if provider not in API_PROVIDERS:
            row.update(status="skipped", detail="no API account (local or served model)")
            continue
        try:
            probe(provider, cfg)
        except BillingExhausted as exc:
            row.update(status="failed", detail=f"billing: {exc}"[:300])
            continue
        except Exception as exc:  # noqa: BLE001 - any failure here blocks the launch
            row.update(status="failed", detail=f"{type(exc).__name__}: {exc}"[:300])
            continue
        if provider != "kimi":
            row["detail"] = "key and model answered (no balance API for this key type)"
            continue
        need = max(float(cfg.get("spend_cap_usd") or 0.0), MIN_BALANCE_USD)
        try:
            available = balance()
        except Exception as exc:  # noqa: BLE001
            row.update(status="failed", detail=f"Moonshot balance check failed: {exc}"[:300])
            continue
        if available < need:
            row.update(status="failed", detail=(
                f"Moonshot balance ${available:.2f} is below ${need:.2f} "
                f"(the model's spend cap, or the ${MIN_BALANCE_USD:.2f} floor)"))
        else:
            row["detail"] = f"key answered; Moonshot balance ${available:.2f} covers ${need:.2f}"
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-config", action="append", required=True)
    args = parser.parse_args(argv)
    if os.environ.get("SKIP_API_PREFLIGHT") == "1":
        print("!!! API PREFLIGHT SKIPPED (SKIP_API_PREFLIGHT=1): dev only. An invalid key or an empty "
              "account will not be caught before VMs start. !!!", file=sys.stderr)
        return 0
    failed = False
    for path in args.run_config:
        rows = preflight(json.loads(Path(path).read_text(encoding="utf-8")))
        print(f"API preflight: {path}")
        for row in rows:
            detail = " ".join(row["detail"].split())  # API error bodies are multi-line JSON
            print(f"  {row['status']:8s} {row['model']:24s} {row['provider']:14s} {detail}")
            failed = failed or row["status"] == "failed"
    if failed:
        print("API preflight FAILED: fix the key or top up the account before launching.", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
