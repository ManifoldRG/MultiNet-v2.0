"""In-memory game session registry."""

from __future__ import annotations

import asyncio
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
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used: float = field(default_factory=time.time)


class GameRegistry:
    def __init__(self, ttl_seconds: float = 3600.0, max_games: int = 64):
        self.ttl_seconds = ttl_seconds
        self.max_games = max_games
        self.catalog = R1ResultCatalog()
        self._games: dict[str, GameEntry] = {}

    def start(self, task_id: str) -> GameEntry:
        self._purge()
        self._evict_overflow()
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

    def _evict_overflow(self) -> None:
        """Drop least-recently-used games so ``/start`` cannot grow without bound."""
        while len(self._games) >= self.max_games:
            oldest_id = min(self._games, key=lambda gid: self._games[gid].last_used)
            self._games.pop(oldest_id).session.close()
