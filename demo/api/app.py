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
from demo.api.view import is_allowed_action, serialize_task, serialize_view
from demo.r1_tasks import list_r1_tasks

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
    }


@app.post("/api/game/{game_id}/action")
def game_action(game_id: str, body: ActionBody) -> dict:
    session = _get(game_id).session
    action = body.action.upper()
    if not is_allowed_action(session, action):
        raise HTTPException(status_code=400, detail=f"Invalid action {action!r}")
    if not session.episode_done:
        session._dispatch_token(action)
    return {"view": serialize_view(session, catalog=registry.catalog)}


@app.post("/api/game/{game_id}/reset")
def game_reset(game_id: str) -> dict:
    session = _get(game_id).session
    session._reset_env()
    return {"view": serialize_view(session, catalog=registry.catalog)}


@app.post("/api/game/{game_id}/navigate")
def game_navigate(game_id: str, body: NavigateBody) -> dict:
    session = _get(game_id).session
    session._load_adjacent_task(body.delta)
    return {
        "task": serialize_task(session),
        "view": serialize_view(session, catalog=registry.catalog),
    }
