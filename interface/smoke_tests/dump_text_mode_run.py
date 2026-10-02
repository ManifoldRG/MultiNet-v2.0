"""Dump text-only model inputs (coords / ascii / json) for a scripted key-switch run.

    python -m interface.smoke_tests.dump_text_mode_run

Writes ``dump_text_mode_run.txt`` next to this file (or ``--output``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interface.agents.reply import Reply
from interface.config import ExperimentConfig
from interface.coords import agent_facing, agent_row_col
from interface.episode_step import EpisodeStepper
from interface.loader import load_task
from interface.runner import build_runner
from interface.smoke_tests.plans import key_switch_001_dump_trajectory

MAZE = ROOT / "gridworld" / "tasks" / "tier3" / "key_switch_001.json"
FORMATS = ("coords", "ascii", "json")
_HERE = Path(__file__).resolve().parent


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts = []
    n_img = 0
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") == "image_url":
            n_img += 1
            parts.append(f"[image block {n_img}]")
    return "\n".join(p for p in parts if p)


def _run_format(fmt: str, actions: list[str]) -> tuple[list[dict], dict]:
    backend, spec = load_task(MAZE)
    config = ExperimentConfig(
        prompting="standard",
        observation="text_only",
        observation_text_format=fmt,
        include_current_observation_description=True,
        observation_text_includes_facing=True,
        context_window="last_n",
        context_n=3,
        querying="step_by_step",
        chat_history="stateless",
        in_context_learning="zero_shot",
        action_space="egocentric",
        feedback="minimal",
        progress_stall_k=None,
    )
    runner = build_runner(config, backend, spec)
    stepper = EpisodeStepper(runner, verbose=False, maze_path=MAZE)
    stepper.start()
    dumps = []
    for i, action in enumerate(actions):
        messages = stepper.next_query()
        if messages is None:
            break
        system = next((m for m in messages if m.get("role") == "system"), None)
        user = next((m for m in messages if m.get("role") == "user"), None)
        dumps.append(
            {
                "query": i + 1,
                "planned_action": action,
                "pos": agent_row_col(stepper.state),
                "facing": agent_facing(stepper.state),
                "carrying": stepper.state.agent_carrying or "-",
                "system": _content_to_text(system["content"]) if system else "",
                "user": _content_to_text(user["content"]) if user else "",
            }
        )
        stepper.apply_reply(Reply(text=f"FINAL_OUTPUT: {action}"))
    stepper.next_query()
    return dumps, stepper.result()


def _compact_log(dumps: list[dict], result: dict) -> str:
    steps = [r for r in result["transcript"] if r.get("kind") == "step"]
    lines = [
        f"maze: {MAZE.name}  task_id={result['task_spec']['task_id']}",
        f"stop: scripted ({len(steps)} env steps / maze cap {result['task_spec']['max_steps']}; "
        f"stall off; stepper.end_reason={result['end_reason']!r})",
        "",
        f"{'q':>3}  {'before':<16}  {'action':<14}  {'event':<10}  after                   carry_after",
    ]
    for d, rec in zip(dumps, steps):
        before = f"{d['pos']} {d['facing']}"
        after_pos = tuple(rec["position_after_row_col"])
        after_fac = rec["facing_after"]
        after_carry = rec["state_after"].get("agent_carrying") or "-"
        lines.append(
            f"{d['query']:3d}  {before:<16}  {rec['action']:<14}  {rec['event_type']:<10}  "
            f"{after_pos} {after_fac:<5}  {after_carry}  | {rec['prompt_feedback']}"
        )
    if len(steps) != len(dumps):
        lines.append(f"(query dumps={len(dumps)} env steps={len(steps)})")
    return "\n".join(lines)


def _format_dump(fmt: str, dumps: list[dict]) -> str:
    chunks = [
        "=" * 88,
        f"FORMAT: {fmt}",
        "=" * 88,
        "",
        "[system prompt — identical every turn in stateless text_only]",
        dumps[0]["system"] if dumps else "",
        "",
    ]
    for d in dumps:
        chunks.extend(
            [
                "-" * 88,
                f"QUERY {d['query']}/{len(dumps)}  before={d['pos']} facing={d['facing']}  "
                f"carrying={d['carrying']!r}  model will emit FINAL_OUTPUT: {d['planned_action']}",
                "-" * 88,
                "[user]",
                d["user"],
                "",
            ]
        )
    return "\n".join(chunks)


def build_dump() -> str:
    actions = key_switch_001_dump_trajectory()
    parts = [
        "Sample text-mode model inputs (post-fix dump for review)",
        "Maze: gridworld/tasks/tier3/key_switch_001.json",
        "Config: observation=text_only, prompting=standard, feedback=minimal,",
        "        context_window=last_n, context_n=3, querying=step_by_step,",
        "        chat_history=stateless, in_context_learning=zero_shot,",
        "        action_space=egocentric, facing included, progress_stall_k=None.",
        "Note: production default is one_shot; that would prepend the 14x14 example",
        "maze + solution on every user turn. Omitted here so this maze is readable.",
        "Carry in the script log is from state_after (after the env step).",
        "",
    ]
    compact = None
    for fmt in FORMATS:
        dumps, result = _run_format(fmt, actions)
        if compact is None:
            compact = _compact_log(dumps, result)
            parts.extend(["SCRIPT LOG (shared across formats)", compact, ""])
        parts.append(_format_dump(fmt, dumps))
    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write text-mode coords/ascii/json prompt dump.")
    parser.add_argument(
        "--output",
        type=Path,
        default=_HERE / "dump_text_mode_run.txt",
    )
    args = parser.parse_args()
    text = build_dump()
    args.output.write_text(text, encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
