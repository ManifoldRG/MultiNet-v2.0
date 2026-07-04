"""Deterministic per-(batch, model) summary from a batch's episode_runs.jsonl.
The summary subagent calls this so the numbers are reproducible, not free-form."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize(rows: list[dict], model_substr: str, batch_name: str) -> dict[str, Any]:
    # Case-insensitive so --model Kimi matches agent_or_model "kimi-k2.6" (etc.);
    # a case-sensitive miss silently summarizes 0 episodes.
    _needle = model_substr.lower()
    sel = [r for r in rows if _needle in str(r.get("agent_or_model", "")).lower()]
    n = len(sel)
    succ = [r for r in sel if r.get("success")]
    ors = [float(r.get("optimality_ratio") or 0.0) for r in sel]
    return {
        "model": model_substr, "batch": batch_name, "n_episodes": n,
        "n_success": len(succ), "success_rate": (len(succ) / n) if n else 0.0,
        "n_truncated": sum(1 for r in sel if r.get("truncated")),
        "mean_optimality_ratio": (sum(ors) / n) if n else 0.0,
        "mean_steps": (sum(int(r.get("steps") or 0) for r in sel) / n) if n else 0.0,
        "total_tokens": sum(int(r.get("tokens") or 0) for r in sel),
        "per_task": {str(r.get("task_id")): bool(r.get("success")) for r in sel},
    }


def to_markdown(s: dict) -> str:
    lines = [
        f"# {s['batch']} — {s['model']}",
        "",
        f"- episodes: **{s['n_episodes']}**, success: **{s['n_success']}** "
        f"({s['success_rate'] * 100:.0f}%)",
        f"- truncated (loops/step-cap): **{s['n_truncated']}**",
        f"- mean optimality ratio: {s['mean_optimality_ratio']:.3f}",
        f"- mean steps: {s['mean_steps']:.1f}",
        f"- total tokens: {s['total_tokens']:,}",
        "",
        "| task | pass |", "|---|---|",
    ]
    lines += [f"| {t} | {'✅' if ok else '❌'} |" for t, ok in sorted(s["per_task"].items())]
    return "\n".join(lines) + "\n"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Summarize one model's slice of a batch.")
    ap.add_argument("jsonl", help="Path to the batch's episode_runs.jsonl")
    ap.add_argument("--model", required=True, help="Substring of agent_or_model to select")
    ap.add_argument("--batch", required=True, help="Batch name (for the heading)")
    ap.add_argument("--out", required=True, help="Output .md path")
    a = ap.parse_args(argv)
    s = summarize(_read_jsonl(Path(a.jsonl)), a.model, a.batch)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_markdown(s))
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
