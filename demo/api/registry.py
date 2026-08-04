"""In-memory game session registry."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from demo.compare import R1ResultCatalog
from demo.r1_tasks import create_r1_play_session
from demo.session import MiniGridPlaySession


@dataclass
class GameEntry:
    game_id: str
    session: MiniGridPlaySession
    last_used: float = field(default_factory=time.time)


class GameRegistry:
    def __init__(self, ttl_seconds: float = 3600.0):
        self.ttl_seconds = ttl_seconds
        self.catalog = R1ResultCatalog()
        self._games: dict[str, GameEntry] = {}

    def start(self, task_id: str) -> GameEntry:
        self._purge()
        session = create_r1_play_session(task_id, catalog=self.catalog)
        entry = GameEntry(game_id=uuid.uuid4().hex, session=session)
        self._games[entry.game_id] = entry
        return entry

    def get(self, game_id: str) -> GameEntry:
        self._purge()
        entry = self._games[game_id]
        entry.last_used = time.time()
        return entry

    def _purge(self) -> None:
        now = time.time()
        for gid in [g for g, e in self._games.items() if now - e.last_used > self.ttl_seconds]:
            self._games.pop(gid).session.close()
