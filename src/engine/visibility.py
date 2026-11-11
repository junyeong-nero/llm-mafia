from __future__ import annotations

from dataclasses import replace

from src.engine.game_state import GameEvent, Player, Role


_MAFIA_PRIVATE_KINDS = {"mafia_chat", "mafia_vote", "mafia_vote_invalid", "mafia_consensus"}
_PERSONAL_KINDS = {"night_action", "night_action_invalid", "investigation_result"}
_INTERNAL_THOUGHT_KINDS = {"strategy", "speak_request_reason"}


def visible_history_for_player(history: list[GameEvent], player: Player) -> list[GameEvent]:
    visible: list[GameEvent] = []
    for event in history:
        if event.private_to is not None and event.private_to != player.name:
            continue
        # Fail closed for unscoped private events, including imported old logs.
        if event.kind in _PERSONAL_KINDS and event.private_to != player.name:
            continue
        if event.kind in _INTERNAL_THOUGHT_KINDS and event.speaker != player.name:
            continue
        if event.kind in _MAFIA_PRIVATE_KINDS and player.role != Role.MAFIA:
            continue
        if event.kind == "night_result":
            public_content = event.content
            for marker in ("mafia_target=", "doctor_target=", "police_target="):
                public_content = public_content.split(marker, maxsplit=1)[0].strip()
            event = replace(event, content=public_content)
        visible.append(event)
    return visible
