"""Smoke test prompting / observation / querying matrix with a stub agent."""

from __future__ import annotations

import argparse
import base64
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interface.config import ExperimentConfig
from interface.loader import default_maze_path, load_task
from interface.runner import build_runner
import interface.renderer as renderer_module
from interface.smoke_tests.plans import plan_to_goal_from_prompt, v01_empty_room_trajectory

_POS_RE = re.compile(r"Position:\s*\((\d+),\s*(\d+)\)")
_FACING_RE = re.compile(r"Facing:\s*([A-Z]+)")
_GOAL_RE = re.compile(r"Goal:\s*\((\d+),\s*(\d+)\)")

_ONE_BY_ONE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5nVxUAAAAASUVORK5CYII="
)


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            blk.get("text", "")
            for blk in content
            if isinstance(blk, dict) and blk.get("type") == "text"
        )
    return ""


def _parse_prompt_state(user_text: str):
    pm = _POS_RE.search(user_text)
    fm = _FACING_RE.search(user_text)
    gm = _GOAL_RE.search(user_text)
    if not (pm and fm and gm):
        return None
    return (int(pm.group(1)), int(pm.group(2))), fm.group(1), (int(gm.group(1)), int(gm.group(2)))


def _plan_to_goal_from_prompt(user_text: str, budget: int = 6) -> list[str]:
    parsed = _parse_prompt_state(user_text)
    if parsed is None:
        return ["TURN_RIGHT"]
    (row, col), facing, (grow, gcol) = parsed
    return plan_to_goal_from_prompt(row, col, facing, grow, gcol, budget)


class ProbeAgent:
    def __init__(self, full_trajectory_actions: list[str]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._full_trajectory_actions = full_trajectory_actions

    def __call__(self, messages: list[dict]) -> str:
        system_text = _extract_text(messages[0].get("content"))
        user_msg = messages[-1]
        user_content = user_msg.get("content")
        user_text = _extract_text(user_content)
        has_image = isinstance(user_content, list) and any(
            isinstance(blk, dict) and blk.get("type") == "image_url" for blk in user_content
        )

        full_mode = "You will not be queried again" in user_text
        subgoal_mode = "SUB_GOAL:" in user_text and "ACTIONS:" in user_text

        if full_mode:
            reply = (
                "SUB_GOAL: Execute maze-aware end-to-end plan.\n"
                f"ACTIONS: {', '.join(self._full_trajectory_actions)}"
            )
        elif subgoal_mode:
            chunk = _plan_to_goal_from_prompt(user_text, budget=4)
            reply = f"SUB_GOAL: Advance toward goal.\nACTIONS: {', '.join(chunk)}"
        else:
            step = _plan_to_goal_from_prompt(user_text, budget=1)[0]
            reply = f"FINAL_OUTPUT: {step}"

        self.calls.append(
            {
                "system": system_text,
                "user_content_type": type(user_content).__name__,
                "has_image": has_image,
                "user_text": user_text,
                "assistant_reply": reply,
                "agent_message_count": len(messages),
            }
        )
        return reply


def _has_step_history(user_text: str, observation: str) -> bool:
    if observation == "text_only":
        return "Recent history" in user_text
    if observation == "image_text":
        return "Recent history" in user_text or "Recent step views" in user_text
    if observation == "image_only":
        return "Recent steps" in user_text or "Action:" in user_text
    return False


def _check(name: str, passed: bool, *, detail: str = "") -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail}


def _collect_checks(cfg: ExperimentConfig, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first = calls[0]
    last = calls[-1]
    query_count = len(calls)
    system = first["system"]
    checks: list[dict[str, Any]] = []

    checks.append(_check("system includes mechanism list", "The environment may contain:" in system))
    checks.append(_check("system includes rules block", "RULES (domain logic):" in system))

    if cfg.observation == "text_only":
        checks.append(_check("text_only user content is plain string", first["user_content_type"] == "str"))
        checks.append(_check("text_only omits image blocks", not first["has_image"]))
    elif cfg.observation in ("image_text", "image_only"):
        checks.append(_check(f"{cfg.observation} user content is block list", first["user_content_type"] == "list"))
        checks.append(_check(f"{cfg.observation} includes image block", first["has_image"]))

    if cfg.observation == "image_only":
        checks.append(
            _check("image_only omits initial NL map in user", "Initial maze (fixed for this episode):" not in first["user_text"])
        )
    elif cfg.observation in ("text_only", "image_text"):
        checks.append(
            _check(f"{cfg.observation} includes initial NL map in user", "Initial maze (fixed for this episode):" in first["user_text"])
        )

    if cfg.querying == "full_trajectory":
        checks.append(
            _check("full_trajectory queries once", query_count == 1, detail=f"got {query_count}")
        )
    if cfg.querying == "step_by_step":
        checks.append(
            _check("step_by_step queries each step", query_count >= 3, detail=f"got {query_count}")
        )
    if cfg.querying == "subgoal":
        checks.append(
            _check("subgoal re-queries across chunks", query_count >= 2, detail=f"got {query_count}")
        )

    if query_count >= 2:
        if cfg.context_window == "last3":
            checks.append(
                _check(
                    "last3 includes prior-step history in later queries",
                    _has_step_history(last["user_text"], cfg.observation),
                )
            )
        if cfg.context_window == "current":
            checks.append(
                _check(
                    "current omits prior-step history",
                    not _has_step_history(last["user_text"], cfg.observation),
                )
            )

    if cfg.chat_history == "stateless":
        counts = [c["agent_message_count"] for c in calls]
        checks.append(
            _check(
                "stateless sends system+user only each query",
                all(n == 2 for n in counts),
                detail=f"counts={counts}",
            )
        )
    elif cfg.chat_history == "rolling":
        counts = [c["agent_message_count"] for c in calls]
        cap = 2 + 2 * cfg.chat_turns_max
        grew = counts[-1] > 2 if query_count >= 2 else counts[-1] == 2
        checks.append(
            _check(
                "rolling accumulates chat then trims tail",
                grew and all(n <= cap for n in counts),
                detail=f"counts={counts} cap={cap}",
            )
        )
    elif cfg.chat_history == "full":
        counts = [c["agent_message_count"] for c in calls]
        expected = 1 + 2 * (query_count - 1) + 1
        checks.append(
            _check(
                "full retains all prior chat turns",
                counts[-1] == expected,
                detail=f"counts={counts} expected_last={expected}",
            )
        )

    return checks


def _config_line(cfg: ExperimentConfig) -> str:
    d = cfg.to_dict()
    return (
        f"prompting={d['prompting']} observation={d['observation']} "
        f"context_window={d['context_window']} querying={d['querying']} "
        f"chat_history={d['chat_history']}"
    )


def _run_case(
    cfg: ExperimentConfig, maze_path: Path, label: str, full_actions: list[str], max_steps: int
):
    backend, spec = load_task(maze_path)
    spec.max_steps = max_steps
    runner = build_runner(cfg, backend, spec)
    agent = ProbeAgent(full_actions)
    result = runner.run(agent, verbose=False)
    return label, cfg, result, agent


def _suite_cases(base: ExperimentConfig, suite: str):
    all_cases = [
        (replace(base, prompting="minimal"), "prompting=minimal"),
        (replace(base, prompting="standard"), "prompting=standard"),
        (replace(base, prompting="verbose"), "prompting=verbose"),
        (replace(base, context_window="current"), "context=current"),
        (replace(base, context_window="last3"), "context=last3"),
        (replace(base, observation="text_only", context_window="last3"), "obs=text_only"),
        (replace(base, observation="image_text", context_window="last3"), "obs=image_text"),
        (replace(base, observation="image_only", context_window="last3"), "obs=image_only"),
        (replace(base, querying="step_by_step"), "query=step_by_step"),
        (replace(base, querying="subgoal"), "query=subgoal"),
        (replace(base, querying="full_trajectory"), "query=full_trajectory"),
        (replace(base, chat_history="stateless"), "chat=stateless"),
        (replace(base, chat_history="rolling"), "chat=rolling"),
        (replace(base, chat_history="full"), "chat=full"),
    ]
    if suite == "all":
        return all_cases
    if suite == "prompting":
        return [c for c in all_cases if c[1].startswith("prompting=")]
    if suite == "observation":
        return [c for c in all_cases if c[1].startswith("obs=") or c[1].startswith("context=")]
    if suite == "querying":
        return [c for c in all_cases if c[1].startswith("query=")]
    if suite == "chat":
        return [c for c in all_cases if c[1].startswith("chat=")]
    raise ValueError(f"Unknown suite: {suite}")


def _format_run_report(label: str, cfg: ExperimentConfig, checks: list[dict[str, Any]], query_count: int) -> list[str]:
    passed = sum(1 for c in checks if c["passed"])
    overall = passed == len(checks)
    lines = [
        f"--- {label} ---",
        f"config: {_config_line(cfg)}",
        f"queries_in_run: {query_count}",
    ]
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        detail = f" ({check['detail']})" if check.get("detail") else ""
        lines.append(f"  [{status}] {check['name']}{detail}")
    lines.append(f"overall: {'PASS' if overall else 'FAIL'} ({passed}/{len(checks)} checks)")
    lines.append("")
    return lines


def run_smoke_suite(
    maze_path: Path, tag: str, max_steps: int, suite: str = "all"
) -> Path:
    maze_stem = maze_path.stem
    suffix = f"_{tag}" if tag else ""
    full_trajectory_actions = v01_empty_room_trajectory()
    renderer_module.rgb_to_png_bytes = lambda _rgb: _ONE_BY_ONE_PNG
    base = ExperimentConfig(
        prompting="standard",
        observation="image_only",
        context_window="last3",
        querying="step_by_step",
    )
    selected = _suite_cases(base, suite)
    outputs = [_run_case(cfg, maze_path, label, full_trajectory_actions, max_steps) for cfg, label in selected]

    body = [
        "Interface prompt / observation / querying smoke report",
        f"maze: {maze_path.name}  max_steps: {max_steps}  runs: {len(outputs)}",
        "",
    ]
    failures: list[str] = []
    passed_runs = 0
    total_runs = len(outputs)

    for label, cfg, _result, agent in outputs:
        query_count = len(agent.calls)
        checks = _collect_checks(cfg, agent.calls)
        if all(c["passed"] for c in checks):
            passed_runs += 1
        body.extend(_format_run_report(label, cfg, checks, query_count))
        for check in checks:
            if not check["passed"]:
                detail = f" ({check['detail']})" if check.get("detail") else ""
                failures.append(f"{label}: {check['name']}{detail}")

    body.append(f"summary: {passed_runs}/{total_runs} runs passed")
    if failures:
        body.append("")
        body.append("FAILURES:")
        body.extend(f"- {f}" for f in failures)
    else:
        body.append("All runs passed.")

    out_dir = Path(__file__).resolve().parent / "results" / f"smoke_{suite}_{maze_stem}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "report.txt"
    report.write_text("\n".join(body), encoding="utf-8")

    print("\n".join(body))
    if failures:
        raise SystemExit(1)
    print(f"report={report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze", default="V01_empty_room.json")
    parser.add_argument("--tag", default="")
    parser.add_argument("--max-steps", type=int, default=5, help="Cap env steps per run (shape checks need few).")
    parser.add_argument("--suite", choices=["all", "prompting", "observation", "querying", "chat"], default="all")
    args = parser.parse_args()
    maze_path = default_maze_path(args.maze)
    run_smoke_suite(maze_path, args.tag, args.max_steps, args.suite)


if __name__ == "__main__":
    main()
