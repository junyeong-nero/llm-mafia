from __future__ import annotations

import re
from typing import Callable

from src.engine.game_state import GameEvent, GameState, Phase, Player


ProgressCallback = Callable[[dict[str, object]], None]


def _append_system_event(
    events: list[GameEvent],
    *,
    turn: int,
    phase: Phase,
    kind: str,
    content: str,
) -> None:
    events.append(GameEvent(turn=turn, phase=phase, speaker="system", kind=kind, content=content))


def _player_name(state: GameState, player_id: int) -> str:
    for player in state.players:
        if player.id == player_id:
            return player.name
    return f"Unknown({player_id})"


def _player_naming_instruction(state: GameState, *, speaker: Player) -> str:
    alive_others = [player.name for player in state.alive_players() if player.id != speaker.id]
    if not alive_others:
        alive_others = [player.name for player in state.players if player.id != speaker.id]
    if alive_others:
        others_instruction = (
            f"Use only these player names when referring to others: {', '.join(alive_others)}. "
        )
    else:
        others_instruction = "There are no other players to name right now. "
    return (
        f"You are {speaker.name}. Use first person (I/me/my) for yourself. "
        f"Do not refer to yourself as {speaker.name}. "
        f"{others_instruction}"
        "Do not use numeric labels such as 'player 1', 'agent 2', or 'P3'."
    )


def _normalize_player_references(text: str, state: GameState) -> str:
    id_to_name = {player.id: player.name for player in state.players}

    def repl(match: re.Match[str]) -> str:
        player_id = int(match.group(1))
        return id_to_name.get(player_id, match.group(0))

    return re.sub(r"\b(?:player|agent|p)\s*#?(\d+)\b", repl, text, flags=re.IGNORECASE)


def _player_status_snapshot(state: GameState) -> list[dict[str, object]]:
    return [
        {"name": player.name, "model_name": player.model_name, "role": player.role.value, "alive": player.alive}
        for player in state.players
    ]


def _emit_progress(progress_callback: ProgressCallback | None, payload: dict[str, object]) -> None:
    if progress_callback is not None:
        progress_callback(payload)
