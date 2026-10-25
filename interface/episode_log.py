"""Episode log serialization — flush runner result to exhaustive on-disk artifacts."""

from __future__ import annotations

import base64
import hashlib
import json
import os
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


class ImageFiles:
    """Each distinct image written once under ``root``, keyed by its base64 payload.

    Full history resends every prior frame with each query; storing one file
    per image per query grows as steps squared. Queries instead reference the
    one file. ``register`` maps an already-written file (a step frame) so a
    query image with the same bytes reuses it; anything else lands in
    ``subdir`` under a content hash.
    """

    def __init__(self, root: Path, subdir: str = "images") -> None:
        self.root = root
        self.subdir = subdir
        self._paths: dict[str, str] = {}

    def register(self, b64: str, rel: str) -> None:
        self._paths.setdefault(b64, rel)

    def path_for(self, b64: str) -> str:
        """Path of the image relative to ``root``, writing it on first sight."""
        rel = self._paths.get(b64)
        if rel is None:
            name = hashlib.sha256(b64.encode("ascii")).hexdigest()[:16] + ".png"
            rel = f"{self.subdir}/{name}" if self.subdir else name
            path = self.root / rel
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + ".tmp")
                tmp.write_bytes(base64.b64decode(b64))
                os.replace(tmp, path)
            self._paths[b64] = rel
        return rel


def _write_rgb(
    path: Path, frame: bytes | np.ndarray | None, images: ImageFiles | None = None
) -> str | None:
    """Write a frame given as PNG bytes (the stepper's) or a raw array."""
    if frame is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    png = frame if isinstance(frame, bytes) else rgb_to_png_bytes(frame)
    path.write_bytes(png)
    if images is not None:
        images.register(
            base64.b64encode(png).decode("ascii"),
            path.relative_to(images.root).as_posix(),
        )
    return path.name


def serialize_message_content(
    content: Any, dest_dir: Path, file_prefix: str, images: ImageFiles | None = None
) -> Any:
    """JSON-safe content with each image replaced by a file reference.

    With ``images``, the file is shared across queries and ``file`` is relative
    to ``dest_dir``. Every other key is kept, so ``deserialize_message_content``
    rebuilds the exact blocks that were sent.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content

    blocks: list[dict[str, Any]] = []
    img_idx = 0
    for blk in content:
        if blk.get("type") == "text":
            blocks.append(dict(blk))
        elif blk.get("type") == "image_url":
            url = blk["image_url"]["url"]
            prefix, _, b64 = url.partition(";base64,")
            if images is None:
                img_name = f"{file_prefix}_img{img_idx:02d}.png"
                img_idx += 1
                (dest_dir / img_name).write_bytes(base64.b64decode(b64))
            else:
                img_name = os.path.relpath(images.root / images.path_for(b64), dest_dir)
            out = {"type": "image", "file": Path(img_name).as_posix()}
            out["media_type"] = prefix.removeprefix("data:")
            out.update((k, v) for k, v in blk.items() if k not in ("type", "image_url"))
            url_extra = {k: v for k, v in blk["image_url"].items() if k != "url"}
            if url_extra:
                out["image_url"] = url_extra
            blocks.append(out)
    return blocks


def serialize_messages(
    messages: list[dict], dest_dir: Path, prefix: str, images: ImageFiles | None = None
) -> list[dict]:
    return [
        {
            "role": msg["role"],
            "content": serialize_message_content(
                msg["content"], dest_dir, f"{prefix}_{msg['role']}_{i:02d}", images
            ),
        }
        for i, msg in enumerate(messages)
    ]


def deserialize_message_content(
    content: Any, src_dir: Path, cache: dict[Path, str] | None = None
) -> Any:
    """Inverse of ``serialize_message_content``: rebuild the blocks as sent.

    ``cache`` shares one base64 string per image file across calls.
    """
    if not isinstance(content, list):
        return content
    cache = {} if cache is None else cache
    blocks: list[Any] = []
    for blk in content:
        if isinstance(blk, dict) and blk.get("type") == "image" and "file" in blk:
            path = src_dir / blk["file"]
            b64 = cache.get(path)
            if b64 is None:
                b64 = cache[path] = base64.b64encode(path.read_bytes()).decode("utf-8")
            media_type = blk.get("media_type", "image/png")
            url = {"url": f"data:{media_type};base64,{b64}", **blk.get("image_url", {})}
            rebuilt = {"type": "image_url", "image_url": url}
            rebuilt.update(
                (k, v) for k, v in blk.items()
                if k not in ("type", "file", "media_type", "image_url")
            )
            blocks.append(rebuilt)
        else:
            blocks.append(blk)
    return blocks


def deserialize_messages(
    messages: list[dict], src_dir: Path, cache: dict[Path, str] | None = None
) -> list[dict]:
    cache = {} if cache is None else cache
    return [
        {**msg, "content": deserialize_message_content(msg["content"], src_dir, cache)}
        for msg in messages
    ]


def load_query_messages(query_dir: Path) -> list[dict]:
    """The agent messages a logged query sent, rebuilt from ``query.json`` and its images."""
    record = json.loads((query_dir / "query.json").read_text(encoding="utf-8"))
    return deserialize_messages(record["agent_messages"], query_dir)


def _pop_frame(rec: dict[str, Any], prefix: str) -> bytes | np.ndarray | None:
    png = rec.pop(f"{prefix}_png", None)
    rgb = rec.pop(f"{prefix}_rgb", None)
    return png if png is not None else rgb


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


def flush_episode_log(result: dict[str, Any], out_dir: Path, *, keep: tuple[str, ...] = ()) -> Path:
    """Write PNG frames and JSON-safe ``episode.json`` from a ``runner.run()`` result.

    ``out_dir`` is rewritten from scratch, except its top-level entries named in
    ``keep`` (a lockstep worker keeps its unit's checkpoint there until the
    result is safely uploaded).
    """
    if out_dir.exists() and keep:
        for child in out_dir.iterdir():
            if child.name in keep:
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    elif out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=bool(keep))
    frames_dir = out_dir / "frames"
    queries_dir = out_dir / "queries"
    # Query images reference the step frames (same PNG bytes) or, for anything
    # else, one shared file each, instead of one copy per query.
    images = ImageFiles(out_dir)

    transcript_out: list[dict[str, Any]] = []
    for rec in result["transcript"]:
        copy = dict(rec)
        kind = copy["kind"]

        if kind == "reset":
            frame = _pop_frame(copy, "_reset_frame")
            if _write_rgb(frames_dir / "reset.png", frame, images):
                copy["reset_frame"] = "frames/reset.png"
            if _write_rgb(frames_dir / "start_map.png", copy.pop("_start_map_rgb", None)):
                copy["start_map_frame"] = "frames/start_map.png"
        elif kind == "query":
            qidx = copy["query_index"]
            qdir = queries_dir / f"query_{qidx:03d}"
            qdir.mkdir(parents=True, exist_ok=True)
            copy["agent_messages"] = serialize_messages(
                copy["agent_messages"], qdir, f"query_{qidx:03d}", images
            )
            query_record = _json_safe(copy)
            (qdir / "query.json").write_text(
                json.dumps(query_record, indent=2, default=str), encoding="utf-8"
            )
        elif kind == "step":
            sidx = copy["step_index"]
            action = _safe_name(copy["action"])
            _pop_frame(copy, "_decision_frame")
            post_name = f"step_{sidx:03d}_{action}.png"
            if _write_rgb(frames_dir / post_name, _pop_frame(copy, "_post_step"), images):
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
    # Phase provenance (Task B3): stamp the two-tier pass and the caps the
    # episode ran under when the caller set them on ``result``, plus the
    # ``render`` block for a 3D run. Additive and optional — absent on
    # non-two-tier / 2D runs — and never part of any input hash.
    for key in ("pass", "max_tokens", "max_model_len", "phase_label", "render", "stall",
                "resumed_from", "resume"):
        value = result.get(key)
        if value is not None:
            episode[key] = _json_safe(value)
    # Atomic write: a crash mid-write must not leave a truncated episode.json
    # that a matching sidecar would treat as a valid cached episode.
    path = out_dir / "episode.json"
    tmp = out_dir / "episode.json.tmp"
    tmp.write_text(json.dumps(episode, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)

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
