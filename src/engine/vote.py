from __future__ import annotations

from src.engine.game_state import GameState


def resolve_vote(state: GameState, *, ballots: dict[int, int]) -> int | None:
    alive = state.alive_players()
    if len(alive) <= 1:
        return None
    alive_ids = {player.id for player in alive}
    votes: dict[int, int] = {player.id: 0 for player in alive}
    for voter_id, target_id in ballots.items():
        if voter_id not in alive_ids:
            continue
        if target_id not in alive_ids:
            continue
        if voter_id == target_id:
            continue
        votes[target_id] += 1

    top_count = max(votes.values())
    if top_count == 0:
        return None
    top_candidates = [pid for pid, count in votes.items() if count == top_count]
    if len(top_candidates) > 1:
        return None
    return top_candidates[0]
