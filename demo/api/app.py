"""FastAPI surface for the MultiNet web MiniGrid player.

    uvicorn demo.api.app:app --reload --app-dir .

Env:
  MULTINET_CORS_ORIGINS     comma-separated origins (default: local static servers)
  MULTINET_SETTINGS_EDITABLE  1/true to allow Tab 1–5 cycling (default: frozen)
  MULTINET_MAX_GAMES        live session cap (default: 64)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

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


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_origins(name: str, default: str) -> list[str]:
    raw = os.environ.get(name, default)
    return [part.strip() for part in raw.split(",") if part.strip()]


# Web R1 keeps ExperimentConfig frozen unless explicitly enabled.
SETTINGS_EDITABLE = _env_flag("MULTINET_SETTINGS_EDITABLE", default=False)

_CORS_ORIGINS = _env_origins(
    "MULTINET_CORS_ORIGINS",
    "http://127.0.0.1:5500,http://localhost:5500",
)

app = FastAPI(title="MultiNet MiniGrid Game API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

registry = GameRegistry(max_games=int(os.environ.get("MULTINET_MAX_GAMES", "64")))


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
async def game_setting(game_id: str, body: SettingBody) -> dict:
    """Cycle a settings axis. No-op while frozen for R1 parity."""
    entry = _get(game_id)
    async with entry.lock:
        session = entry.session
        if SETTINGS_EDITABLE:
            session._cycle_setting(body.key)
        return {
            "view": serialize_view(session, catalog=registry.catalog),
            "settings": serialize_settings(session, editable=SETTINGS_EDITABLE),
        }


@app.get("/api/game/{game_id}/model-view")
def game_model_view(game_id: str) -> dict:
    return serialize_model_view(_get(game_id).session)


@app.get("/api/game/{game_id}/trajectory")
def game_trajectory(game_id: str) -> dict:
    return serialize_trajectory(_get(game_id).session)


@app.post("/api/game/{game_id}/action")
async def game_action(game_id: str, body: ActionBody) -> dict:
    entry = _get(game_id)
    async with entry.lock:
        session = entry.session
        action = body.action.upper()
        if not is_allowed_action(session, action):
            raise HTTPException(status_code=400, detail=f"Invalid action {action!r}")
        sfx = None
        effects: list = []
        if not session.episode_done:
            prev_state = session.state
            events_before = len(session.event_log)
            prev_rgb = None
            if session.backend.env is not None:
                prev_rgb = np.asarray(session.backend.render(), dtype=np.uint8)
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
async def game_reset(game_id: str) -> dict:
    entry = _get(game_id)
    async with entry.lock:
        session = entry.session
        session._reset_env()
        return {
            "view": serialize_view(session, catalog=registry.catalog),
            "sfx": "restart",
        }


@app.post("/api/game/{game_id}/navigate")
async def game_navigate(game_id: str, body: NavigateBody) -> dict:
    entry = _get(game_id)
    async with entry.lock:
        session = entry.session
        session._load_adjacent_task(body.delta)
        return {
            "task": serialize_task(session),
            "view": serialize_view(session, catalog=registry.catalog),
            "sfx": "navigate",
        }
