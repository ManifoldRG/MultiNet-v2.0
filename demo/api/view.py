"""Serialize ``MiniGridPlaySession`` state for the web player."""

from __future__ import annotations

import base64
import io
import json

import numpy as np
from PIL import Image

from demo.compare import R1ResultCatalog, TaskComparison
from demo.r1_tasks import canonical_task_id
from demo.session import SETTINGS_AXES, MiniGridPlaySession, TASK_INSTRUCTION
from demo.theme import recolor_walls
from interface.action_space import EGOCENTRIC_ACTIONS, valid_actions


def _grid_image_b64(session: MiniGridPlaySession) -> str:
    rgb = recolor_walls(np.asarray(session.backend.render(), dtype=np.uint8))
    img = Image.fromarray(rgb[:, :, :3], mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def serialize_comparison(comparison: TaskComparison) -> dict:
    return {
        "taskId": comparison.task_id,
        "optimalSteps": comparison.optimal_steps,
        "models": [
            {
                "displayName": m.display_name,
                "success": m.success,
                "steps": m.steps,
                "summaryLine": m.summary_line,
            }
            for m in comparison.models
        ],
    }


def serialize_task(session: MiniGridPlaySession) -> dict:
    return {
        "taskId": canonical_task_id(session),
        "description": (session.task_spec.description or "").strip(),
        "instruction": TASK_INSTRUCTION,
        "difficultyTier": session.task_spec.difficulty_tier,
        "optimalSteps": session.optimal_steps,
        "maxSteps": session.state.max_steps,
        "taskIndex": session.task_index,
        "taskCount": len(session.task_list),
    }


def serialize_view(
    session: MiniGridPlaySession,
    *,
    catalog: R1ResultCatalog | None = None,
) -> dict:
    state = session.state
    view = {
        "gridImage": _grid_image_b64(session),
        "stepCount": state.step_count,
        "maxSteps": state.max_steps,
        "facing": state.agent_direction,
        "lastAction": session.last_action_name,
        "carrying": state.agent_carrying,
        "progress": [
            {
                "prefix": ev.prefix,
                "objectPhrase": ev.object_phrase,
                "suffix": ev.suffix,
                "color": ev.color,
                "icon": ev.icon,
            }
            for ev in session.event_log
        ],
        "done": session.episode_done,
        "success": session.episode_success,
        "endReason": session.end_reason,
        "actions": list(EGOCENTRIC_ACTIONS),
        "optimalSteps": session.optimal_steps,
        "displayReward": session.display_reward,
    }
    if session.episode_done:
        catalog = catalog or R1ResultCatalog()
        view["comparison"] = serialize_comparison(catalog.lookup(canonical_task_id(session)))
    return view


def is_allowed_action(session: MiniGridPlaySession, action: str) -> bool:
    return action in valid_actions(session.config.action_space)


def serialize_settings(session: MiniGridPlaySession, *, editable: bool = False) -> dict:
    """Tab-settings payload. Web R1 keeps ``editable=False``."""
    axes = [
        {
            "key": key_char,
            "attr": attr,
            "value": getattr(session.config, attr),
        }
        for key_char, attr, _choices in SETTINGS_AXES
    ]
    manifest_row = None
    if session.manifest_mode and session.task_path is not None:
        manifest_row = session.manifest_row_by_path.get(session.task_path)
    return {
        "editable": editable,
        "axes": axes,
        "manifestRow": manifest_row,
        "help": (
            "Frozen for R1 parity (matches interface.config.ExperimentConfig). "
            "Tab / Esc to close."
            if not editable
            else "Press a number to cycle a value. Tab / Esc to close."
        ),
    }


def serialize_model_view(session: MiniGridPlaySession) -> dict:
    return {
        "observation": session.config.observation,
        "contextWindow": session.config.context_window,
        "sections": [
            {"title": title, "text": text}
            for title, text in session._build_model_view_sections()
        ],
    }


def serialize_trajectory(session: MiniGridPlaySession) -> dict:
    """In-memory trajectory JSON (same fields as desktop ``--record``)."""
    return json.loads(json.dumps(session.trajectory_dict(), default=str))
