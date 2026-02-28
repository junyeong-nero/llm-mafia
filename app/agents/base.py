from __future__ import annotations

from typing import Protocol

from app.engine.game_state import GameEvent


class Agent(Protocol):
    name: str
    model_id: str

    def speak(self, *, phase: str, turn: int, prompt: str, history: list[GameEvent] | None = None) -> str:
        ...
