# VM Setup & Orchestration Tooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the tooling that takes a bare A100 VM to a verified Qwen3.6 runner, plus an API-key round-trip smoke and a cluster inventory/preflight/launch-command helper, so the 4-node smoke (2 Qwen workers + 1 Kimi API-client + coordinator) is push-button.

**Architecture:** A new top-level importable package `deploy/` holds the Python tooling (key resolution, key smoke, cluster model, preflight, launch-command printer) plus two operator-run artifacts (`setup_qwen_vm.sh`, `smoke_qwen.py`). Inference stays on the existing HF-`generate()` agent (`interface/agents/qwen35_vl.py`); the only product code touch is making that agent prefer a Qwen3.6 model class. CI-testable Python is built test-first; the shell installer and GPU smoke are verified via `--dry-run`/`shellcheck` and documented as operator steps.

**Tech Stack:** Python ≥3.10 (stdlib only for the new modules — `argparse`, `json`, `urllib`, `subprocess`, `socket`, `shutil`, `dataclasses`), pytest, bash, HuggingFace Transformers + Torch (cu128) for the operator-run pieces.

## Global Constraints

- **Branch discipline:** work only on `Distributed-run-pipeline`; never commit/push to `main`.
- **Commit trailer:** every commit ends with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` (use a second `-m` flag).
- **Importability:** new modules live in `deploy/` (a package); tests run via `python -m pytest` from repo root. `deploy*` must be added to `[tool.setuptools.packages.find].include` in `pyproject.toml`.
- **New modules are stdlib-only** — no new pip dependencies for `deploy/`.
- **Key resolution precedence:** `env > .env > api_key.txt`. Providers: `anthropic` → `ANTHROPIC_API_KEY`, `kimi` → `MOONSHOT_API_KEY`. `api_key.txt` order: line 1 = anthropic, line 2 = kimi.
- **No quantization for Qwen3.6:** `load_in_4bit=false`, bf16, `attn_implementation="flash_attention_2"`.
- **Model ids:** default `Qwen/Qwen3.6-27B`; MoE `Qwen/Qwen3.6-35B-A3B`.
- **Torch pin (proven stack):** `torch==2.9.1+cu128 torchvision torchaudio` from `https://download.pytorch.org/whl/cu128`; `transformers` latest (pin after the smoke confirms Qwen3.6 loads).
- **Throughput:** 100 tok/s is **measured and reported**, never asserted; the smoke labels it "single-stream HF generate".
- **Kernel build failures are warn-only** (`flash-attn`, `flash-linear-attention`, `causal-conv1d`).

---

### Task 1: `deploy/` package scaffold + packaging wire-up + `docs/future_directions.md`

**Files:**
- Create: `deploy/__init__.py`
- Create: `docs/future_directions.md`
- Modify: `pyproject.toml` (add `"deploy*"` to `[tool.setuptools.packages.find].include`)
- Test: `tests/test_deploy_package.py`

**Interfaces:**
- Consumes: nothing.
- Produces: importable package `deploy`; `deploy.REPO_ROOT: Path` (repo root, = `Path(__file__).resolve().parents[1]`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_deploy_package.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_deploy_package.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'deploy'`.

- [ ] **Step 3: Create the package**

```python
# deploy/__init__.py
"""Operator/ops tooling for VM setup and distributed-run orchestration."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
```

- [ ] **Step 4: Wire packaging**

In `pyproject.toml`, under `[tool.setuptools.packages.find]`, add `"deploy*",` to the `include` list (alongside `"scripts*"`).

- [ ] **Step 5: Write `docs/future_directions.md`**

```markdown
# Future Directions

Deferred work intentionally kept out of the current scope. Each item records the
"why now" trigger so we can pick it up without re-deriving context.

## Inference engine: vLLM / SGLang serving (the real path to 100 tok/s)

The Qwen runner currently uses in-process HuggingFace `model.generate()`
(`interface/agents/qwen35_vl.py`). Single-stream HF generate on a 27B dense
hybrid (Gated DeltaNet + gated attention) will likely land **below** the
~100 tok/s A100 target even with `flash-attn` + `flash-linear-attention` +
`causal-conv1d` and bf16. `deploy/smoke_qwen.py` measures and reports the actual
rate so this is visible rather than assumed.

The realistic route to the target without quantization is a serving engine:

- **vLLM ≥ 0.19** or **SGLang ≥ 0.5.10**, both of which support Qwen3.6 and expose
  an **OpenAI-compatible** HTTP API. A local server gives continuous batching,
  prefix caching, paged KV, and (SGLang) multi-token prediction (~3–5× decode).
- Integration would add a `qwen_vllm` agent that POSTs to `localhost` — nearly
  identical in shape to the existing `interface/agents/kimi_k26.py` (also an
  OpenAI-compatible client). No change to the coordinator/worker lifecycle:
  the worker VM runs its own server, the agent just points at it.

This is deferred because we chose not to rebuild the backend now.

## Other deferred items

- **SSH auto-launcher** for the cluster (read `deploy/cluster.example.json` and
  start each node's role over SSH). Today we print the commands instead.
- **FP8 / quantized serving** as a throughput/VRAM lever (explicitly avoided now).
- **Multi-GPU tensor parallelism** for larger checkpoints.
- **systemd units** generated from the cluster inventory for unattended runners.
- **Pre-release cleanup action** to keep `docs/superpowers/**` and generated
  artifacts out of the release repo on merge (tracked separately).
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_deploy_package.py -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add deploy/__init__.py docs/future_directions.md pyproject.toml tests/test_deploy_package.py
git commit -m "deploy: package scaffold + future_directions doc" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `.env.example` + `.gitignore` negation + `deploy/envkeys.py`

**Files:**
- Create: `.env.example`
- Modify: `.gitignore` (add `!.env.example` after the `.env.*` line)
- Create: `deploy/envkeys.py`
- Test: `tests/test_envkeys.py`

**Interfaces:**
- Consumes: `deploy.REPO_ROOT`.
- Produces:
  - `PROVIDER_ENV: dict[str, str]` = `{"anthropic": "ANTHROPIC_API_KEY", "kimi": "MOONSHOT_API_KEY"}`
  - `parse_dotenv(path: Path) -> dict[str, str]`
  - `resolve_keys(repo_root: Path | None = None, env: Mapping[str, str] | None = None) -> dict[str, str | None]` (keys `"anthropic"`, `"kimi"`; value is the key string or `None`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_envkeys.py
from __future__ import annotations

from deploy.envkeys import PROVIDER_ENV, parse_dotenv, resolve_keys


def test_parse_dotenv_ignores_comments_and_strips(tmp_path):
    p = tmp_path / ".env"
    p.write_text(
        "# comment\n"
        "\n"
        'ANTHROPIC_API_KEY="sk-ant-xyz"\n'
        "export MOONSHOT_API_KEY = 'ms-123' \n",
        encoding="utf-8",
    )
    parsed = parse_dotenv(p)
    assert parsed["ANTHROPIC_API_KEY"] == "sk-ant-xyz"
    assert parsed["MOONSHOT_API_KEY"] == "ms-123"


def test_resolve_keys_precedence_env_over_dotenv_over_file(tmp_path):
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=from-dotenv\n", encoding="utf-8")
    (tmp_path / "api_key.txt").write_text("from-file-anthropic\nfrom-file-kimi\n", encoding="utf-8")
    # env wins for anthropic; .env is absent for kimi so api_key.txt line 2 wins
    keys = resolve_keys(repo_root=tmp_path, env={"ANTHROPIC_API_KEY": "from-env"})
    assert keys["anthropic"] == "from-env"
    assert keys["kimi"] == "from-file-kimi"


def test_resolve_keys_absent_is_none(tmp_path):
    keys = resolve_keys(repo_root=tmp_path, env={})
    assert keys == {"anthropic": None, "kimi": None}
    assert set(PROVIDER_ENV) == {"anthropic", "kimi"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_envkeys.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'deploy.envkeys'`.

- [ ] **Step 3: Implement `deploy/envkeys.py`**

```python
# deploy/envkeys.py
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
```

- [ ] **Step 4: Create `.env.example` and the `.gitignore` negation**

`.env.example`:

```bash
# Copy to .env and fill in. Loaded by deploy/check_api_keys.py and the pipeline
# (env vars). For shell use:  set -a && source .env && set +a
# Alternative (already supported): a gitignored api_key.txt with line 1 = Anthropic,
# line 2 = Moonshot.
ANTHROPIC_API_KEY=
MOONSHOT_API_KEY=
```

In `.gitignore`, immediately after the `.env.*` line (in the "Secrets / local config" block), add:

```
!.env.example
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_envkeys.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Verify `.env.example` is no longer ignored**

Run: `git check-ignore .env.example; echo "exit=$?"`
Expected: prints `exit=1` (not ignored).

- [ ] **Step 7: Commit**

```bash
git add deploy/envkeys.py .env.example .gitignore tests/test_envkeys.py
git commit -m "deploy: .env.example + key resolution (env/.env/api_key.txt)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `deploy/check_api_keys.py` — "Hello, how are you?" round-trip

**Files:**
- Create: `deploy/check_api_keys.py`
- Test: `tests/test_check_api_keys.py`

**Interfaces:**
- Consumes: `deploy.envkeys.PROVIDER_ENV`, `deploy.envkeys.resolve_keys`; agents `interface.agents.claude.ClaudeAnthropicAgent`/`ClaudeAnthropicConfig`, `interface.agents.kimi_k26.KimiK26Agent`/`KimiK26Config`.
- Produces:
  - `HELLO: str` = `"Hello, how are you?"`
  - `hello_roundtrip(provider: str, key: str) -> dict` (raises on failure; returns `{"provider","ok","reply","usage"}`)
  - `run_checks(keys: dict[str, str | None], roundtrip=hello_roundtrip) -> tuple[list[dict], int]` (each result has `provider`, `status` ∈ {`ok`,`skipped`,`failed`}, `detail`; exit code 1 iff a *present* key failed)
  - `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_check_api_keys.py
from __future__ import annotations

import deploy.check_api_keys as cak


def test_run_checks_present_ok_absent_skipped_failed_nonzero():
    calls = []

    def fake_roundtrip(provider, key):
        calls.append((provider, key))
        if provider == "kimi":
            raise RuntimeError("Moonshot API HTTP 401: bad key")
        return {"provider": provider, "ok": True, "reply": "I am well!", "usage": {"output_tokens": 5}}

    keys = {"anthropic": "good-key", "kimi": "bad-key"}
    results, code = cak.run_checks(keys, roundtrip=fake_roundtrip)

    by_provider = {r["provider"]: r for r in results}
    assert by_provider["anthropic"]["status"] == "ok"
    assert by_provider["kimi"]["status"] == "failed"
    assert code == 1
    assert ("anthropic", "good-key") in calls and ("kimi", "bad-key") in calls


def test_run_checks_absent_key_is_skipped_and_not_called():
    calls = []

    def fake_roundtrip(provider, key):
        calls.append(provider)
        return {"provider": provider, "ok": True, "reply": "hi", "usage": None}

    results, code = cak.run_checks({"anthropic": None, "kimi": "k"}, roundtrip=fake_roundtrip)
    by_provider = {r["provider"]: r for r in results}
    assert by_provider["anthropic"]["status"] == "skipped"
    assert calls == ["kimi"]
    assert code == 0


def test_hello_roundtrip_routes_to_kimi_agent(monkeypatch):
    seen = {}

    class FakeKimi:
        def __init__(self, config=None, api_key=None):
            seen["config"] = config
            seen["api_key"] = api_key
            self.last_usage = {"output_tokens": 3}

        def __call__(self, messages):
            seen["messages"] = messages
            return "Doing great!"

    import interface.agents.kimi_k26 as kimi_mod
    monkeypatch.setattr(kimi_mod, "KimiK26Agent", FakeKimi)

    out = cak.hello_roundtrip("kimi", "ms-key")
    assert out["ok"] is True and out["reply"] == "Doing great!"
    assert seen["api_key"] == "ms-key"
    assert seen["messages"] == [{"role": "user", "content": cak.HELLO}]
    assert seen["config"].max_tokens == 32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_check_api_keys.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'deploy.check_api_keys'`.

- [ ] **Step 3: Implement `deploy/check_api_keys.py`**

```python
# deploy/check_api_keys.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_check_api_keys.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add deploy/check_api_keys.py tests/test_check_api_keys.py
git commit -m "deploy: API-key Hello round-trip smoke (Claude + Kimi)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `deploy/cluster.py` + `deploy/cluster.example.json`

**Files:**
- Create: `deploy/cluster.py`
- Create: `deploy/cluster.example.json`
- Test: `tests/test_cluster.py`

**Interfaces:**
- Consumes: nothing (stdlib).
- Produces:
  - dataclasses `Coordinator(host, port, artifacts_root, run_set_id)`, `Worker(name, host, model_group, hardware_profile, venv=None)`, `ApiClient(name, model_group, runs_on="coordinator")`, `Cluster(coordinator, workers, api_clients, storage_config=None)`
  - `Cluster.coordinator_url -> str`, `Cluster.model_groups() -> set[str]`, `Cluster.worker(name) -> Worker`
  - `load_cluster(path) -> Cluster`
  - `ClusterConfigError(ValueError)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cluster.py
from __future__ import annotations

import json

import pytest

from deploy import REPO_ROOT
from deploy.cluster import ClusterConfigError, load_cluster


def test_loads_example_inventory():
    cluster = load_cluster(REPO_ROOT / "deploy" / "cluster.example.json")
    assert cluster.coordinator_url.startswith("http://")
    assert cluster.coordinator.port > 0
    assert len(cluster.workers) >= 1
    assert "kimi-api" in cluster.model_groups()
    assert cluster.worker(cluster.workers[0].name).model_group


def test_missing_coordinator_raises(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"workers": []}), encoding="utf-8")
    with pytest.raises(ClusterConfigError):
        load_cluster(p)


def test_missing_worker_field_raises(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(
        json.dumps(
            {
                "coordinator": {"host": "h", "port": 8765, "artifacts_root": "a", "run_set_id": "r"},
                "workers": [{"name": "w1", "host": "h2", "hardware_profile": "local-gpu"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ClusterConfigError):
        load_cluster(p)


def test_bad_json_raises(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ClusterConfigError):
        load_cluster(p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cluster.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'deploy.cluster'`.

- [ ] **Step 3: Implement `deploy/cluster.py`**

```python
# deploy/cluster.py
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
```

- [ ] **Step 4: Create `deploy/cluster.example.json`**

```json
{
  "coordinator": {
    "host": "10.0.0.1",
    "port": 8765,
    "artifacts_root": "artifacts/cond/prompt",
    "run_set_id": "cond_prompt"
  },
  "storage_config": "gridworld/fixtures/storage_config.example.json",
  "workers": [
    {"name": "qwen-1", "host": "10.0.0.2", "model_group": "qwen36-27b", "hardware_profile": "local-gpu", "venv": "/opt/qwen/.venv-qwen"},
    {"name": "qwen-2", "host": "10.0.0.3", "model_group": "qwen36-27b", "hardware_profile": "local-gpu"}
  ],
  "api_clients": [
    {"name": "kimi", "model_group": "kimi-api", "runs_on": "coordinator"}
  ]
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_cluster.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add deploy/cluster.py deploy/cluster.example.json tests/test_cluster.py
git commit -m "deploy: cluster inventory model + example" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: `deploy/preflight.py` — per-node readiness checker

**Files:**
- Create: `deploy/preflight.py`
- Test: `tests/test_preflight.py`

**Interfaces:**
- Consumes: `deploy.cluster.Cluster`/`load_cluster`, `deploy.envkeys.resolve_keys`.
- Produces:
  - `CheckResult(name: str, ok: bool, detail: str, fatal: bool = True)`
  - `Probes` dataclass of injectable callables: `gpu() -> tuple[bool,str]`, `vram_free_gb() -> float`, `weights_present(model_group: str) -> bool`, `kernels() -> dict[str,bool]`, `http_ok(url: str) -> bool`, `port_free(host: str, port: int) -> bool`, `gsutil_present() -> bool`, `keys() -> dict[str,str|None]`
  - `default_probes() -> Probes`
  - `provider_for_group(model_group: str) -> str` (`"kimi"` if `"kimi"` in group else `"anthropic"`)
  - `run_preflight(cluster, node, role, probes, *, min_vram_gb: float = 55.0) -> tuple[list[CheckResult], int]` (exit code 1 iff any fatal check failed)
  - `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preflight.py
from __future__ import annotations

from deploy.cluster import ApiClient, Cluster, Coordinator, Worker
from deploy.preflight import Probes, run_preflight


def _cluster():
    return Cluster(
        coordinator=Coordinator(host="10.0.0.1", port=8765, artifacts_root="/tmp/art", run_set_id="rs"),
        workers=[Worker(name="qwen-1", host="10.0.0.2", model_group="qwen36-27b", hardware_profile="local-gpu")],
        api_clients=[ApiClient(name="kimi", model_group="kimi-api")],
        storage_config=None,
    )


def _probes(**overrides):
    base = dict(
        gpu=lambda: (True, "NVIDIA A100-SXM4-80GB"),
        vram_free_gb=lambda: 79.0,
        weights_present=lambda g: True,
        kernels=lambda: {"flash_attn": True, "fla": True, "causal_conv1d": True},
        http_ok=lambda url: True,
        port_free=lambda h, p: True,
        gsutil_present=lambda: True,
        keys=lambda: {"anthropic": "a", "kimi": "k"},
    )
    base.update(overrides)
    return Probes(**base)


def test_worker_all_pass():
    results, code = run_preflight(_cluster(), "qwen-1", "worker", _probes())
    assert code == 0
    assert all(r.ok for r in results if r.fatal)


def test_worker_no_gpu_is_fatal():
    results, code = run_preflight(_cluster(), "qwen-1", "worker", _probes(gpu=lambda: (False, "no nvidia-smi")))
    assert code == 1


def test_worker_low_vram_is_fatal():
    results, code = run_preflight(_cluster(), "qwen-1", "worker", _probes(vram_free_gb=lambda: 20.0))
    assert code == 1


def test_worker_missing_weights_is_fatal():
    results, code = run_preflight(_cluster(), "qwen-1", "worker", _probes(weights_present=lambda g: False))
    assert code == 1


def test_worker_missing_kernel_is_warn_only():
    probes = _probes(kernels=lambda: {"flash_attn": False, "fla": True, "causal_conv1d": True})
    results, code = run_preflight(_cluster(), "qwen-1", "worker", probes)
    assert code == 0
    assert any((not r.ok) and (not r.fatal) for r in results)


def test_worker_coordinator_unreachable_is_fatal():
    results, code = run_preflight(_cluster(), "qwen-1", "worker", _probes(http_ok=lambda url: False))
    assert code == 1


def test_coordinator_port_busy_is_warn_only():
    results, code = run_preflight(_cluster(), None, "coordinator", _probes(port_free=lambda h, p: False))
    assert code == 0


def test_api_client_missing_key_is_fatal():
    probes = _probes(keys=lambda: {"anthropic": "a", "kimi": None})
    results, code = run_preflight(_cluster(), None, "api-client", probes)
    assert code == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_preflight.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'deploy.preflight'`.

- [ ] **Step 3: Implement `deploy/preflight.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_preflight.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add deploy/preflight.py tests/test_preflight.py
git commit -m "deploy: per-node preflight checker (worker/coordinator/api-client)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: `deploy/print_launch_commands.py` — per-node command printer

**Files:**
- Create: `deploy/print_launch_commands.py`
- Test: `tests/test_print_launch_commands.py`

**Interfaces:**
- Consumes: `deploy.cluster.Cluster`/`load_cluster`.
- Produces:
  - `PIPELINE: str` = `"multinet-run-pipeline"`
  - `build_commands(cluster, run_config, manifest, conditions=None, seeds="0", prompt_variant=None) -> dict[str, str]` (keys: `coordinator-prepare`, `coordinator-serve`, `worker:<name>`, `api-client:<name>`, `coordinator-finalize`)
  - `plan_model_groups(cluster) -> set[str] | None` (None when no `job_plan.json`)
  - `group_mismatches(cluster) -> list[str]`
  - `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_print_launch_commands.py
from __future__ import annotations

import json
from pathlib import Path

from deploy.cluster import ApiClient, Cluster, Coordinator, Worker
from deploy.print_launch_commands import build_commands, group_mismatches


def _cluster(artifacts_root: str, storage=None):
    return Cluster(
        coordinator=Coordinator(host="10.0.0.1", port=8765, artifacts_root=artifacts_root, run_set_id="rs"),
        workers=[Worker(name="qwen-1", host="10.0.0.2", model_group="qwen36-27b", hardware_profile="local-gpu")],
        api_clients=[ApiClient(name="kimi", model_group="kimi-api")],
        storage_config=storage,
    )


def test_build_commands_wires_coordinator_url_and_flags():
    cmds = build_commands(
        _cluster("artifacts/x", storage="gridworld/fixtures/storage_config.example.json"),
        run_config="rc.json",
        manifest="m.json",
        conditions="Prompt",
        prompt_variant="minimal",
    )
    assert "--distributed-role coordinator-prepare" in cmds["coordinator-prepare"]
    assert '--conditions "Prompt"' in cmds["coordinator-prepare"]
    assert "--prompt-variant minimal" in cmds["coordinator-prepare"]
    assert "--coordinator-url http://10.0.0.1:8765" in cmds["worker:qwen-1"]
    assert "--model-group qwen36-27b" in cmds["worker:qwen-1"]
    assert "coordinator-run-api-client" in cmds["api-client:kimi"]
    assert "--storage-config gridworld/fixtures/storage_config.example.json" in cmds["coordinator-serve"]


def test_group_mismatch_detected_against_plan(tmp_path):
    art = tmp_path / "art"
    (art / "distributed").mkdir(parents=True)
    plan = {"models": {"qwen": {"model_group": "qwen36-27b"}}}  # note: no kimi-api group
    (art / "distributed" / "job_plan.json").write_text(json.dumps(plan), encoding="utf-8")

    warnings = group_mismatches(_cluster(str(art)))
    assert any("kimi-api" in w for w in warnings)


def test_group_mismatch_empty_when_no_plan(tmp_path):
    assert group_mismatches(_cluster(str(tmp_path / "missing"))) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_print_launch_commands.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'deploy.print_launch_commands'`.

- [ ] **Step 3: Implement `deploy/print_launch_commands.py`**

```python
# deploy/print_launch_commands.py
"""Print the exact distributed-role command to run on each node (no SSH)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

from deploy.cluster import Cluster, load_cluster

PIPELINE = "multinet-run-pipeline"


def build_commands(
    cluster: Cluster,
    run_config: str,
    manifest: str,
    conditions: Optional[str] = None,
    seeds: str = "0",
    prompt_variant: Optional[str] = None,
) -> Dict[str, str]:
    c = cluster.coordinator
    url = cluster.coordinator_url
    storage = f" --storage-config {cluster.storage_config}" if cluster.storage_config else ""
    cond = f' --conditions "{conditions}"' if conditions else ""
    pv = f" --prompt-variant {prompt_variant}" if prompt_variant else ""

    cmds: Dict[str, str] = {}
    cmds["coordinator-prepare"] = (
        f"{PIPELINE} --distributed-role coordinator-prepare "
        f"--run-config {run_config} --manifest {manifest}{cond}{pv} "
        f"--seeds {seeds} --artifacts-root {c.artifacts_root} --run-set-id {c.run_set_id}"
    )
    cmds["coordinator-serve"] = (
        f"{PIPELINE} --distributed-role coordinator-serve "
        f"--artifacts-root {c.artifacts_root} --host {c.host} --port {c.port}{storage}"
    )
    for w in cluster.workers:
        cmds[f"worker:{w.name}"] = (
            f"{PIPELINE} --distributed-role worker --coordinator-url {url} "
            f"--artifacts-root {c.artifacts_root} --model-group {w.model_group} "
            f"--hardware-profile {w.hardware_profile}"
        )
    for ac in cluster.api_clients:
        cmds[f"api-client:{ac.name}"] = (
            f"{PIPELINE} --distributed-role coordinator-run-api-client "
            f"--artifacts-root {c.artifacts_root} --model-group {ac.model_group}"
        )
    cmds["coordinator-finalize"] = (
        f"{PIPELINE} --distributed-role coordinator-finalize "
        f"--artifacts-root {c.artifacts_root} --run-set-id {c.run_set_id}{storage}"
    )
    return cmds


def plan_model_groups(cluster: Cluster) -> Optional[Set[str]]:
    plan_path = Path(cluster.coordinator.artifacts_root) / "distributed" / "job_plan.json"
    if not plan_path.is_file():
        return None
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    return {str(m.get("model_group")) for m in (data.get("models") or {}).values()}


def group_mismatches(cluster: Cluster) -> List[str]:
    plan_groups = plan_model_groups(cluster)
    if plan_groups is None:
        return []
    return [
        f"model_group {g!r} is in the cluster inventory but not in the prepared plan {sorted(plan_groups)}"
        for g in sorted(cluster.model_groups())
        if g not in plan_groups
    ]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Print per-node launch commands for a distributed run.")
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--conditions")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--prompt-variant")
    args = parser.parse_args(argv)

    cluster = load_cluster(args.cluster)
    cmds = build_commands(
        cluster, args.run_config, args.manifest, args.conditions, args.seeds, args.prompt_variant
    )
    print("# Coordinator node:")
    print("  " + cmds["coordinator-prepare"])
    print("  " + cmds["coordinator-serve"])
    for ac in cluster.api_clients:
        print(f"  # api-client {ac.name} (runs on coordinator):")
        print("  " + cmds[f"api-client:{ac.name}"])
    print("  # after workers finish:")
    print("  " + cmds["coordinator-finalize"])
    for w in cluster.workers:
        print(f"\n# Worker node {w.name} ({w.host}):")
        print("  " + cmds[f"worker:{w.name}"])

    for warning in group_mismatches(cluster):
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_print_launch_commands.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add deploy/print_launch_commands.py tests/test_print_launch_commands.py
git commit -m "deploy: per-node launch-command printer + plan group cross-check" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Qwen3.6 product touches — model-class preference + smoke run-config

**Files:**
- Modify: `interface/agents/qwen35_vl.py` (extract the class-name tuple to a module constant; add the Qwen3.6 entry; iterate the constant)
- Create: `gridworld/fixtures/run_config.smoke_qwen36_kimi.json`
- Test: `tests/test_qwen36_setup.py`

**Interfaces:**
- Consumes: existing `Qwen35VLAgent` / `scripts.run_pipeline.load_run_config`.
- Produces: `interface.agents.qwen35_vl.QWEN_MODEL_CLASS_NAMES: tuple[str, ...]` (Qwen3.6 class first, then Qwen3.5, then Auto* fallbacks).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_qwen36_setup.py
from __future__ import annotations

from interface.agents.qwen35_vl import QWEN_MODEL_CLASS_NAMES
from scripts.run_pipeline import load_run_config
from deploy import REPO_ROOT


def test_qwen36_class_is_preferred_over_qwen35():
    names = QWEN_MODEL_CLASS_NAMES
    assert "Qwen3_6ForConditionalGeneration" in names
    assert "Qwen3_5ForConditionalGeneration" in names
    assert names.index("Qwen3_6ForConditionalGeneration") < names.index("Qwen3_5ForConditionalGeneration")
    # Auto* fallbacks remain, after the explicit classes.
    assert names.index("Qwen3_5ForConditionalGeneration") < names.index("AutoModelForCausalLM")


def test_smoke_qwen36_run_config_is_no_quant_and_correct_model():
    cfg = load_run_config(REPO_ROOT / "gridworld" / "fixtures" / "run_config.smoke_qwen36_kimi.json")
    qwen = cfg["models"]["qwen36_27b_hf"]
    assert qwen["model"] == "Qwen/Qwen3.6-27B"
    assert qwen["group"] == "qwen36-27b"
    assert qwen["load_in_4bit"] is False
    assert cfg["models"]["kimi_k26"]["group"] == "kimi-api"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_qwen36_setup.py -v`
Expected: FAIL — `ImportError: cannot import name 'QWEN_MODEL_CLASS_NAMES'`.

- [ ] **Step 3: Edit `interface/agents/qwen35_vl.py`**

Add the module-level constant near the top (after `DEFAULT_QWEN35_VL_MODEL`):

```python
# Model classes to try, in preference order. Qwen3.6 first, then Qwen3.5, then
# the generic Auto* fallbacks. Kept as a constant so the ordering is testable
# without importing transformers / loading weights.
QWEN_MODEL_CLASS_NAMES = (
    "Qwen3_6ForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
    "AutoModelForImageTextToText",
    "AutoModelForVision2Seq",
    "AutoModelForCausalLM",
)
```

Then replace the body of `_model_class` so it iterates the constant:

```python
    def _model_class(self):
        import transformers

        for name in QWEN_MODEL_CLASS_NAMES:
            model_cls = getattr(transformers, name, None)
            if model_cls is not None:
                return model_cls
        raise ImportError("Transformers does not provide a usable Qwen 3.5/3.6 model class.")
```

- [ ] **Step 4: Create `gridworld/fixtures/run_config.smoke_qwen36_kimi.json`**

```json
{
  "description": "Qwen3.6 VM smoke (no quantization): 2 Qwen3.6-27B runners + 1 Kimi runner over the 3-maze smoke manifest, for coordinator work-stealing. Not for measurement.",
  "manifest": "gridworld/fixtures/manifest.smoke_eval.json",
  "conditions": null,
  "models": {
    "qwen36_27b_hf": {
      "provider": "qwen",
      "model": "Qwen/Qwen3.6-27B",
      "temperature": 0.0,
      "max_tokens": 4096,
      "enable_thinking": false,
      "device_map": {
        "": 0
      },
      "local_files_only": true,
      "torch_dtype": "bfloat16",
      "attn_implementation": "flash_attention_2",
      "load_in_4bit": false,
      "group": "qwen36-27b",
      "hardware_profile": "local-gpu",
      "worker_count": 2,
      "max_in_flight": 2,
      "tasks": [
        "all"
      ]
    },
    "kimi_k26": {
      "provider": "kimi",
      "model": "kimi-k2.6",
      "temperature": 0.0,
      "max_tokens": 4096,
      "timeout": 180,
      "group": "kimi-api",
      "worker_count": 1,
      "max_in_flight": 1,
      "tasks": [
        "all"
      ]
    }
  }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_qwen36_setup.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add interface/agents/qwen35_vl.py gridworld/fixtures/run_config.smoke_qwen36_kimi.json tests/test_qwen36_setup.py
git commit -m "qwen: prefer Qwen3.6 model class + no-quant smoke run-config" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: `deploy/smoke_qwen.py` — load + generate, report tok/s

**Files:**
- Create: `deploy/smoke_qwen.py`
- Test: `tests/test_smoke_qwen.py`

**Interfaces:**
- Consumes: `interface.agents.qwen35_vl.Qwen35VLAgent`/`Qwen35VLConfig` (operator-run path).
- Produces:
  - `TARGET_TOK_S: float = 100.0`
  - `kernels_active() -> dict[str, bool]` (keys `flash_attn`, `fla`, `causal_conv1d`)
  - `format_verdict(tok_per_s: float, target: float = TARGET_TOK_S) -> str`
  - `main(argv: list[str] | None = None) -> int` (`--model`, `--quick`, `--max-new-tokens`, `--self-test`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_smoke_qwen.py
from __future__ import annotations

from deploy.smoke_qwen import TARGET_TOK_S, format_verdict, kernels_active, main


def test_format_verdict_below_target():
    msg = format_verdict(42.0)
    assert "BELOW" in msg
    assert "42.0" in msg
    assert "future_directions" in msg


def test_format_verdict_meets_target():
    assert "MEETS" in format_verdict(TARGET_TOK_S + 1.0)


def test_kernels_active_reports_bools_for_expected_keys():
    status = kernels_active()
    assert set(status) == {"flash_attn", "fla", "causal_conv1d"}
    assert all(isinstance(v, bool) for v in status.values())


def test_self_test_mode_runs_without_gpu(capsys):
    code = main(["--self-test"])
    assert code == 0
    out = capsys.readouterr().out
    assert "kernels" in out.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_smoke_qwen.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'deploy.smoke_qwen'`.

- [ ] **Step 3: Implement `deploy/smoke_qwen.py`**

```python
# deploy/smoke_qwen.py
"""Operator smoke: load Qwen3.6 via HF generate, do a tiny turn, report tok/s.

GPU + weights required for a real run. `--self-test` runs no model and only
exercises the kernel probe + verdict formatting (used by CI and the installer's
dry-run)."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Optional

DEFAULT_MODEL = "Qwen/Qwen3.6-27B"
TARGET_TOK_S = 100.0


def kernels_active() -> Dict[str, bool]:
    status: Dict[str, bool] = {}
    for mod in ("flash_attn", "fla", "causal_conv1d"):
        try:
            __import__(mod)
            status[mod] = True
        except Exception:  # noqa: BLE001 - any import error => unavailable
            status[mod] = False
    return status


def format_verdict(tok_per_s: float, target: float = TARGET_TOK_S) -> str:
    status = "MEETS" if tok_per_s >= target else "BELOW"
    return (
        f"[{status} target] {tok_per_s:.1f} tok/s vs {target:.0f} "
        f"(single-stream HF generate; see docs/future_directions.md for the vLLM path)"
    )


def _run_model(model: str, max_new_tokens: int) -> float:
    """Load the agent (no quantization) and time a single decode. Returns tok/s."""
    from interface.agents.qwen35_vl import Qwen35VLAgent, Qwen35VLConfig

    agent = Qwen35VLAgent(
        config=Qwen35VLConfig(
            model=model,
            load_in_4bit=False,
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_2",
            max_new_tokens=max_new_tokens,
            local_files_only=True,
        )
    )
    messages = [{"role": "user", "content": "Hello, how are you? Reply in one short sentence."}]
    t0 = time.perf_counter()
    agent(messages)
    elapsed = time.perf_counter() - t0
    out_tokens = int((agent.last_usage or {}).get("output_tokens", 0))
    return (out_tokens / elapsed) if elapsed > 0 else 0.0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen3.6 load + throughput smoke.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--quick", action="store_true", help="Text-only (no image turn).")
    parser.add_argument("--self-test", action="store_true", help="No GPU: print kernels + a sample verdict.")
    args = parser.parse_args(argv)

    print(f"kernels: {kernels_active()}")
    if args.self_test:
        print(format_verdict(0.0))
        return 0

    tok_s = _run_model(args.model, args.max_new_tokens)
    print(format_verdict(tok_s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_smoke_qwen.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add deploy/smoke_qwen.py tests/test_smoke_qwen.py
git commit -m "deploy: Qwen3.6 load/throughput smoke (reports tok/s vs target)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: `deploy/setup_qwen_vm.sh` — single invasive installer

**Files:**
- Create: `deploy/setup_qwen_vm.sh` (executable)
- Test: manual — `shellcheck` + `--dry-run` (no unit test; GPU/root required for a real run)

**Interfaces:**
- Consumes: `deploy/smoke_qwen.py` (final verify step).
- Produces: a provisioned venv + Qwen3.6 weights; operator-run only.

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Single invasive setup for a Qwen3.6 A100 VM (Ubuntu/apt):
#   driver/CUDA -> venv -> torch(cu128) + transformers + fast kernels -> weights -> verify.
# Idempotent: re-running skips work already done. Use --dry-run to preview.
set -euo pipefail

MODEL="27b"            # 27b | moe | both
VENV="./.venv-qwen"
CUDA_VERSION="12.8"
SKIP_DRIVER=0
DRY_RUN=0
LOG="deploy/setup_qwen_vm.log"

QWEN_27B="Qwen/Qwen3.6-27B"
QWEN_MOE="Qwen/Qwen3.6-35B-A3B"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
TORCH_PKGS=("torch==2.9.1+cu128" "torchvision" "torchaudio")

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --venv) VENV="$2"; shift 2;;
    --cuda) CUDA_VERSION="$2"; shift 2;;
    --skip-driver) SKIP_DRIVER=1; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

run() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then "$@"; fi
}

log() { echo "[setup_qwen_vm] $*"; }

ensure_driver() {
  if [[ "$SKIP_DRIVER" -eq 1 ]]; then log "skip-driver set; not touching driver/CUDA"; return; fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi healthy; skipping driver install"
  else
    log "installing NVIDIA driver via apt (ubuntu-drivers autoinstall)"
    run sudo apt-get update
    run sudo apt-get install -y ubuntu-drivers-common
    run sudo ubuntu-drivers autoinstall
    log "driver installed; a reboot may be required before nvidia-smi works"
  fi
  if command -v nvcc >/dev/null 2>&1; then
    log "nvcc present: $(nvcc --version 2>/dev/null | tail -1 || true)"
  else
    log "installing CUDA toolkit ${CUDA_VERSION} (for building flash-attn / causal-conv1d)"
    local pkg="cuda-toolkit-${CUDA_VERSION/./-}"
    run sudo apt-get install -y "${pkg}" || log "WARN: ${pkg} not available via apt; install the toolkit manually if kernel builds fail"
  fi
}

ensure_build_deps() {
  run sudo apt-get install -y build-essential ninja-build git python3-venv python3-dev
}

ensure_venv() {
  if [[ ! -d "$VENV" ]]; then run python3 -m venv "$VENV"; fi
  # shellcheck disable=SC1091
  if [[ "$DRY_RUN" -eq 0 ]]; then source "$VENV/bin/activate"; fi
  run python -m pip install -U pip wheel setuptools
}

install_torch() {
  run python -m pip install -U --index-url "$TORCH_INDEX" "${TORCH_PKGS[@]}"
}

install_runtime() {
  run python -m pip install -U transformers accelerate bitsandbytes pillow einops "huggingface_hub[cli]"
}

install_kernels() {
  # Warn-only: the model still runs (slower) without these.
  run python -m pip install -U flash-linear-attention causal-conv1d || log "WARN: fla/causal-conv1d build failed (continuing)"
  run python -m pip install -U flash-attn --no-build-isolation || log "WARN: flash-attn build failed (continuing)"
}

download_weights() {
  local repos=()
  case "$MODEL" in
    27b) repos=("$QWEN_27B");;
    moe) repos=("$QWEN_MOE");;
    both) repos=("$QWEN_27B" "$QWEN_MOE");;
    *) echo "Unknown --model: $MODEL" >&2; exit 2;;
  esac
  export HF_HUB_ENABLE_HF_TRANSFER=0
  for repo in "${repos[@]}"; do
    log "downloading $repo"
    run hf download "$repo" --include "*.safetensors" --include "*.index.json" \
      --include "*.json" --include "*.jinja" --include "*.txt" --max-workers 16
  done
}

verify() {
  run python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    run python deploy/smoke_qwen.py --self-test
  else
    log "dry-run: would run 'python deploy/smoke_qwen.py --self-test'"
  fi
}

main() {
  mkdir -p "$(dirname "$LOG")"
  log "model=$MODEL venv=$VENV cuda=$CUDA_VERSION dry_run=$DRY_RUN"
  ensure_driver
  ensure_build_deps
  ensure_venv
  install_torch
  install_runtime
  install_kernels
  download_weights
  verify
  log "DONE. If nvidia-smi/CUDA are healthy and the smoke prints tok/s, snapshot this image for the other Qwen runners."
}

main 2>&1 | tee -a "$LOG"
```

- [ ] **Step 2: Make it executable and gitignore its log**

Run: `chmod +x deploy/setup_qwen_vm.sh`

Append to `.gitignore` (the script tees output to `deploy/setup_qwen_vm.log`, even in `--dry-run`):

```
deploy/*.log
```

- [ ] **Step 3: Lint (if shellcheck available)**

Run: `command -v shellcheck >/dev/null && shellcheck deploy/setup_qwen_vm.sh || echo "shellcheck not installed; skipping"`
Expected: no errors (warnings acceptable); or the skip message.

- [ ] **Step 4: Dry-run smoke (no changes made)**

Run: `bash deploy/setup_qwen_vm.sh --dry-run --skip-driver`
Expected: prints `+ ...` planned commands (pip/hf/python) and a final `DONE.` line; exit 0; makes no installs.

- [ ] **Step 5: Commit**

```bash
git add deploy/setup_qwen_vm.sh .gitignore
git commit -m "deploy: single invasive Qwen3.6 VM setup script" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Operator runbook (post-implementation, requires GPU/keys/cluster)

These are **operator steps**, not CI — documented here so they aren't forgotten:

1. **Provision a Qwen VM:** `bash deploy/setup_qwen_vm.sh --model 27b` (add `--model both` to also pull the MoE for the A/B probe). Then snapshot the image.
2. **Throughput/functional check:** `source .venv-qwen/bin/activate && python deploy/smoke_qwen.py` (real tok/s); compare 27B vs MoE on the validation_10 maze 3.5 failed.
3. **Keys:** copy `.env.example` → `.env`, fill in, then `python deploy/check_api_keys.py`.
4. **Cluster:** copy `deploy/cluster.example.json` → `deploy/cluster.json`, set real IPs.
5. **Preflight each node:** e.g. on a worker `python deploy/preflight.py --cluster deploy/cluster.json --role worker --node qwen-1`; on coordinator `--role coordinator`; for the API client `--role api-client`.
6. **Launch:** `python deploy/print_launch_commands.py --cluster deploy/cluster.json --run-config gridworld/fixtures/run_config.smoke_qwen36_kimi.json --manifest gridworld/fixtures/manifest.smoke_eval.json` and run each printed command on its node.

## Final full-suite check

- [ ] Run the whole new suite: `python -m pytest tests/test_deploy_package.py tests/test_envkeys.py tests/test_check_api_keys.py tests/test_cluster.py tests/test_preflight.py tests/test_print_launch_commands.py tests/test_qwen36_setup.py tests/test_smoke_qwen.py -v`
  Expected: all PASS.
- [ ] Confirm no regression in the existing agent test: `python -m pytest tests/test_kimi_k26_agent.py -q`
  Expected: PASS.
