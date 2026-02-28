from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Role(str, Enum):
    MAFIA = "mafia"
    POLICE = "police"
    DOCTOR = "doctor"
    CITIZEN = "citizen"


class Phase(str, Enum):
    SETUP = "setup"
    NIGHT = "night"
    DAY = "day"
    VOTE = "vote"
    END = "end"


@dataclass(frozen=True)
class Player:
    id: int
    name: str
    model_name: str
    model_id: str
    role: Role
    alive: bool = True


@dataclass(frozen=True)
class GameEvent:
    turn: int
    phase: Phase
    speaker: str
    kind: str
    content: str


@dataclass
class GameState:
    turn: int
    phase: Phase
    players: list[Player]
    events: list[GameEvent] = field(default_factory=list)
    winner: str | None = None

    def alive_players(self) -> list[Player]:
        return [player for player in self.players if player.alive]

    def alive_by_role(self, role: Role) -> list[Player]:
        return [player for player in self.players if player.alive and player.role == role]

    def replace_player(self, player_id: int, *, alive: bool) -> None:
        updated: list[Player] = []
        for player in self.players:
            if player.id == player_id:
                updated.append(
                    Player(
                        id=player.id,
                        name=player.name,
                        model_name=player.model_name,
                        model_id=player.model_id,
                        role=player.role,
                        alive=alive,
                    )
                )
            else:
                updated.append(player)
        self.players = updated
