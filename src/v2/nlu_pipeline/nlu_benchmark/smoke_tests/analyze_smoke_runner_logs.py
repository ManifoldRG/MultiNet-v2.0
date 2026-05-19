from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze smoke workflow logs.")
    parser.add_argument("--maze", default="V01_empty_room.json", help="Maze JSON filename used by smoke run.")
    parser.add_argument("--tag", default="", help="Optional output tag suffix used at smoke run time.")
    args = parser.parse_args()

    maze_stem = Path(args.maze).stem
    suffix = f"_{args.tag}" if args.tag else ""
    p = Path(__file__).resolve().parent / "results" / f"smoke_runner_matrix_{maze_stem}{suffix}" / "detailed_logs.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    runs = d["runs"]
    print("runs", len(runs))

    issues: list[tuple] = []
    for r in runs:
        label = r["label"]
        cfg = r["config"]
        queries = r["queries"]
        transcript = r["transcript"]
        system_prompt = r["system_prompt"]

        if r["summary"]["query_count"] != len(queries):
            issues.append((label, "query_count_mismatch", r["summary"]["query_count"], len(queries)))

        if cfg["observation"] == "text_only":
            if any(q["has_image"] for q in queries):
                issues.append((label, "text_only_has_image"))
            if any(q["user_content_type"] != "str" for q in queries):
                issues.append((label, "text_only_content_type"))
        else:
            if any(not q["has_image"] for q in queries):
                issues.append((label, "image_mode_missing_image"))
            if any(q["user_content_type"] != "list" for q in queries):
                issues.append((label, "image_mode_not_list"))

        has_initial = "Initial maze (fixed for this episode):" in system_prompt
        if cfg["observation"] == "screenshot_only" and has_initial:
            issues.append((label, "screenshot_has_initial_maze"))
        if cfg["observation"] != "screenshot_only" and not has_initial:
            issues.append((label, "non_screenshot_missing_initial_maze"))

        has_mechanism_list = "The environment may contain:" in system_prompt
        has_rules = "RULES (domain logic):" in system_prompt
        if cfg["prompting"] == "minimal" and has_mechanism_list:
            issues.append((label, "minimal_has_mech_list"))
        if cfg["prompting"] == "standard" and (not has_mechanism_list or has_rules):
            issues.append((label, "standard_prompt_wrong"))
        if cfg["prompting"] == "verbose" and not has_rules:
            issues.append((label, "verbose_missing_rules"))

        if cfg["querying"] == "full_trajectory" and len(queries) != 1:
            issues.append((label, "full_trajectory_query_count", len(queries)))
        if cfg["querying"] == "step_by_step" and len(queries) < 2:
            issues.append((label, "step_by_step_too_few_queries", len(queries)))
        if cfg["querying"] == "subgoal":
            if len(queries) < 2:
                issues.append((label, "subgoal_too_few_queries", len(queries)))
            if not any("subgoal" in t for t in transcript):
                issues.append((label, "subgoal_metadata_missing"))

        if len(queries) >= 2:
            second_text = queries[1]["user_text"]
            has_recent = "Recent history (last 3 steps, oldest first):" in second_text
            has_action_only = "Recent steps (oldest first, action only):" in second_text
            if cfg["context_window"] == "current" and (has_recent or has_action_only):
                issues.append((label, "current_has_history"))
            if cfg["context_window"] == "last3":
                if cfg["observation"] == "screenshot_only" and not has_action_only:
                    issues.append((label, "last3_screenshot_missing_action_history"))
                if cfg["observation"] != "screenshot_only" and not has_recent:
                    issues.append((label, "last3_missing_history"))

        steps = [t["step"] for t in transcript]
        if steps != sorted(steps):
            issues.append((label, "transcript_steps_unsorted"))

    print("issues", len(issues))
    for issue in issues:
        print("ISSUE", issue)

    for r in runs:
        label = r["label"]
        cfg = r["config"]
        print(
            f"{label:24} q={r['summary']['query_count']:2} "
            f"steps={r['summary']['steps_used']:2} success={r['summary']['success']} "
            f"obs={cfg['observation']} ctx={cfg['context_window']} qry={cfg['querying']}"
        )


if __name__ == "__main__":
    main()
