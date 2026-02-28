from __future__ import annotations

from collections import Counter

from app.engine.game_state import GameEvent, GameState, Role


def collect_metrics(state: GameState, events: list[GameEvent]) -> dict[str, object]:
    alive_roles = Counter(player.role.value for player in state.alive_players())
    vote_events = [event for event in events if event.kind == "vote_result"]
    night_kill_events = [event for event in events if event.kind == "night_result"]
    return {
        "winner": state.winner,
        "final_turn": state.turn,
        "alive_count": len(state.alive_players()),
        "alive_roles": dict(alive_roles),
        "vote_rounds": len(vote_events),
        "night_resolutions": len(night_kill_events),
        "mafia_alive": len(state.alive_by_role(Role.MAFIA)),
    }
