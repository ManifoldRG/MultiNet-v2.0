"""FastAPI surface for the MultiNet web MiniGrid player.

    uvicorn demo.api.app:app --reload --app-dir .
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from demo.api.registry import GameRegistry
from demo.api.view import (
    is_allowed_action,
    serialize_model_view,
    serialize_settings,
    serialize_task,
    serialize_trajectory,
    serialize_view,
)
from demo.fx import effects_for_dispatch
from demo.r1_tasks import list_r1_tasks
from demo.sounds import sfx_for_dispatch

# Flip to True to let Tab 1–5 cycle ExperimentConfig axes (web + API).
# Desktop mirrors this via MiniGridPlayerUI.settings_editable.
SETTINGS_EDITABLE = True

app = FastAPI(title="MultiNet MiniGrid Game API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = GameRegistry()


class StartBody(BaseModel):
    taskId: str


class ActionBody(BaseModel):
    action: str


class NavigateBody(BaseModel):
    delta: Literal[-1, 1]


class SettingBody(BaseModel):
    key: str


def _get(game_id: str):
    try:
        return registry.get(game_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown or expired gameId") from None


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/game/tasks")
def list_tasks() -> dict:
    return {"tasks": list_r1_tasks(catalog=registry.catalog)}


@app.post("/api/game/start")
def start_game(body: StartBody) -> dict:
    try:
        entry = registry.start(body.taskId)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
    return {
        "gameId": entry.game_id,
        "task": serialize_task(entry.session),
        "view": serialize_view(entry.session, catalog=registry.catalog),
        "settings": serialize_settings(entry.session, editable=SETTINGS_EDITABLE),
    }


@app.get("/api/game/{game_id}/settings")
def game_settings(game_id: str) -> dict:
    session = _get(game_id).session
    return serialize_settings(session, editable=SETTINGS_EDITABLE)


@app.post("/api/game/{game_id}/setting")
def game_setting(game_id: str, body: SettingBody) -> dict:
    """Cycle a settings axis. No-op while frozen for R1 parity."""
    session = _get(game_id).session
    if SETTINGS_EDITABLE:
        session._cycle_setting(body.key)
    return {
        "view": serialize_view(session, catalog=registry.catalog),
        "settings": serialize_settings(session, editable=SETTINGS_EDITABLE),
        "modelView": serialize_model_view(session),
    }


@app.get("/api/game/{game_id}/model-view")
def game_model_view(game_id: str) -> dict:
    return serialize_model_view(_get(game_id).session)


@app.get("/api/game/{game_id}/trajectory")
def game_trajectory(game_id: str) -> dict:
    return serialize_trajectory(_get(game_id).session)


@app.post("/api/game/{game_id}/action")
def game_action(game_id: str, body: ActionBody) -> dict:
    session = _get(game_id).session
    action = body.action.upper()
    if not is_allowed_action(session, action):
        raise HTTPException(status_code=400, detail=f"Invalid action {action!r}")
    sfx = None
    effects: list = []
    if not session.episode_done:
        prev_state = session.state
        events_before = len(session.event_log)
        prev_rgb = None
        try:
            if session.backend.env is not None:
                import numpy as np

                prev_rgb = np.asarray(session.backend.render(), dtype=np.uint8)
        except Exception:
            prev_rgb = None
        session._dispatch_token(action)
        sfx = sfx_for_dispatch(session, events_before)
        effects = effects_for_dispatch(
            session, action, prev_state, events_before, prev_rgb=prev_rgb
        )
    return {
        "view": serialize_view(session, catalog=registry.catalog),
        "sfx": sfx,
        "effects": effects,
    }


@app.post("/api/game/{game_id}/reset")
def game_reset(game_id: str) -> dict:
    session = _get(game_id).session
    session._reset_env()
    return {"view": serialize_view(session, catalog=registry.catalog), "sfx": "restart"}


@app.post("/api/game/{game_id}/navigate")
def game_navigate(game_id: str, body: NavigateBody) -> dict:
    session = _get(game_id).session
    session._load_adjacent_task(body.delta)
    return {
        "task": serialize_task(session),
        "view": serialize_view(session, catalog=registry.catalog),
        "sfx": "navigate",
    }
