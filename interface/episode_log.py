"""Episode log serialization — flush runner result to exhaustive on-disk artifacts."""

from __future__ import annotations

import base64
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from gridworld.backends.base import GridState

from interface.coords import agent_facing, agent_row_col, inventory_list
from interface.renderer import rgb_to_png_bytes


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "record"


def state_snapshot(state: GridState) -> dict[str, Any]:
    snap = state.to_dict()
    snap["facing"] = agent_facing(state)
    snap["position_row_col"] = list(agent_row_col(state))
    snap["inventory"] = inventory_list(state)
    return snap


def _write_rgb(path: Path, rgb: np.ndarray | None) -> str | None:
    if rgb is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rgb_to_png_bytes(rgb))
    return path.name


def serialize_message_content(content: Any, dest_dir: Path, file_prefix: str) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content

    blocks: list[dict[str, Any]] = []
    img_idx = 0
    for blk in content:
        if blk.get("type") == "text":
            blocks.append({"type": "text", "text": blk["text"]})
        elif blk.get("type") == "image_url":
            url = blk["image_url"]["url"]
            _, _, b64 = url.partition(";base64,")
            img_name = f"{file_prefix}_img{img_idx:02d}.png"
            img_idx += 1
            (dest_dir / img_name).write_bytes(base64.b64decode(b64))
            blocks.append({"type": "image", "file": img_name})
    return blocks


def serialize_messages(messages: list[dict], dest_dir: Path, prefix: str) -> list[dict]:
    return [
        {
            "role": msg["role"],
            "content": serialize_message_content(msg["content"], dest_dir, f"{prefix}_{msg['role']}_{i:02d}"),
        }
        for i, msg in enumerate(messages)
    ]


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return None
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def flush_episode_log(result: dict[str, Any], out_dir: Path) -> Path:
    """Write PNG frames and JSON-safe ``episode.json`` from a ``runner.run()`` result."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    frames_dir = out_dir / "frames"
    queries_dir = out_dir / "queries"

    transcript_out: list[dict[str, Any]] = []
    for rec in result["transcript"]:
        copy = dict(rec)
        kind = copy["kind"]

        if kind == "reset":
            if _write_rgb(frames_dir / "reset.png", copy.pop("_reset_frame_rgb", None)):
                copy["reset_frame"] = "frames/reset.png"
        elif kind == "query":
            qidx = copy["query_index"]
            qdir = queries_dir / f"query_{qidx:03d}"
            qdir.mkdir(parents=True, exist_ok=True)
            copy["agent_messages"] = serialize_messages(
                copy["agent_messages"], qdir, f"query_{qidx:03d}"
            )
            query_record = _json_safe(copy)
            (qdir / "query.json").write_text(
                json.dumps(query_record, indent=2, default=str), encoding="utf-8"
            )
        elif kind == "step":
            sidx = copy["step_index"]
            action = _safe_name(copy["action"])
            copy.pop("_decision_frame_rgb", None)
            post_name = f"step_{sidx:03d}_{action}.png"
            if _write_rgb(frames_dir / post_name, copy.pop("_post_step_rgb", None)):
                copy["post_step_frame"] = f"frames/{post_name}"

        transcript_out.append(_json_safe(copy))

    episode = _json_safe(
        {
            "success": result["success"],
            "steps_used": result["steps_used"],
            "end_reason": result["end_reason"],
            "query_count": result["query_count"],
            "config": result["config"],
            "maze_path": result["maze_path"],
            "task_spec": result["task_spec"],
            "initial_state": result["initial_state"],
            "final_state": result["final_state"],
            "transcript": transcript_out,
        }
    )
    path = out_dir / "episode.json"
    path.write_text(json.dumps(episode, indent=2, default=str), encoding="utf-8")

    kinds: dict[str, int] = {}
    for rec in transcript_out:
        kinds[rec["kind"]] = kinds.get(rec["kind"], 0) + 1
    (out_dir / "report.txt").write_text(
        "\n".join(
            [
                "Interface episode log",
                f"success={result['success']}",
                f"end_reason={result['end_reason']}",
                f"steps_used={result['steps_used']}",
                f"query_count={result['query_count']}",
                f"transcript_records={len(transcript_out)} ({', '.join(f'{k}={v}' for k, v in sorted(kinds.items()))})",
                f"out_dir={out_dir}",
            ]
        ),
        encoding="utf-8",
    )
    return path
