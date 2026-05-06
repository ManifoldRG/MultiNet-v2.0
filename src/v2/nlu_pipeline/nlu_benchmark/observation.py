"""Observation builder for the NLU benchmark.

* **text_only** / **image_text** – The runner appends initial NL layout to the
  system message once per episode. Each user turn: ``render_user_observation_text``,
  last3 history, and live PNG when image is enabled.

* **image_only** – No initial NL map in system; live PNG each query; ``last3``
  history is multimodal: up to three prior **decision-frame** PNGs (view before each
  executed action) plus ``Action: …`` lines only — pose/outcome are left to the image.

``build_image_blocks`` adds PNGs whenever observation is not ``text_only`` (see ``runner._build_message``).
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import List, Literal, Optional

from nlu_benchmark.renderer import render_maze_image_png_bytes, render_user_observation_text


class _StepRecord:
    __slots__ = ("position", "facing", "action", "feedback", "png_raw")

    def __init__(self, position, facing, action, feedback, png_raw: Optional[bytes] = None):
        self.position = position
        self.facing   = facing
        self.action   = action
        self.feedback = feedback
        self.png_raw  = png_raw


class ObservationBuilder:
    """Builds what the model sees each step from config.observation + context_window."""

    def __init__(
        self,
        observation: Literal["text_only", "image_text", "image_only"],
        context_window: Literal["current", "last3"],
    ) -> None:
        self._observation    = observation
        self._context_window = context_window
        self._history: List[_StepRecord] = []

    def reset(self) -> None:
        self._history.clear()

    def render_decision_frame_png(self, state) -> Optional[bytes]:
        """PNG of the maze **before** ``env.step`` mutates ``state`` (``image_only`` only)."""
        if self._observation != "image_only":
            return None
        try:
            return render_maze_image_png_bytes(state)
        except Exception:
            return None

    def record(
        self,
        position,
        facing: str,
        action: str,
        feedback: str,
        *,
        decision_frame_png: Optional[bytes] = None,
    ) -> None:
        png_raw = decision_frame_png if self._observation == "image_only" else None
        self._history.append(_StepRecord(position, facing, action, feedback, png_raw))

    def history_content_blocks(self) -> List[dict]:
        """Multimedia tail for ``image_only`` + ``last3``: prior frames + action labels only."""
        if self._observation != "image_only" or self._context_window == "current" or not self._history:
            return []
        recs = self._history[-3:]
        blocks: List[dict] = []
        for rec in recs:
            if not rec.png_raw:
                continue
            b64 = base64.b64encode(rec.png_raw).decode("utf-8")
            blocks.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            blocks.append({"type": "text", "text": f"Action: {rec.action}\n\n"})
        if not blocks:
            return []
        intro = (
            "Recent steps (oldest first). Each image is the maze view from which the "
            "following action was chosen; infer pose and environment state from the image.\n\n"
        )
        return [{"type": "text", "text": intro}] + blocks

    def history_text(self) -> str:
        if (
            self._context_window == "current"
            or not self._history
            or self._observation == "image_only"
        ):
            return ""
        recs = self._history[-3:]
        lines = ["Recent history (last 3 steps, oldest first):"]
        for rec in recs:
            lines.append(
                f"  {rec.position} facing {rec.facing} -> {rec.action} -> {rec.feedback}"
            )
        return "\n".join(lines)

    def build_text(self, state) -> str:
        if self._observation == "image_only":
            return ""
        return render_user_observation_text(state)

    def build_image_blocks(self, state, maze_json_path: Optional[str]) -> List[dict]:
        if self._observation == "text_only":
            return []
        try:
            raw = render_maze_image_png_bytes(state)
        except Exception:
            raw = b""
        if raw:
            b64 = base64.b64encode(raw).decode("utf-8")
            return [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]
        b = _load_maze_png_block(maze_json_path)
        return [b] if b else []


def _load_maze_png_block(maze_json_path: Optional[str]) -> Optional[dict]:
    if not maze_json_path:
        return None
    p = Path(maze_json_path)
    img_path = p.parent / "pngs" / (p.stem + ".png")
    if not img_path.exists():
        return None
    b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
