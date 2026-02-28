from __future__ import annotations

import random

from app.engine.game_state import GameState


def resolve_vote(state: GameState, *, seed: int) -> int | None:
    alive = state.alive_players()
    if len(alive) <= 1:
        return None
    rng = random.Random(seed + state.turn * 13)
    votes: dict[int, int] = {p.id: 0 for p in alive}
    for voter in alive:
        candidates = [p.id for p in alive if p.id != voter.id]
        choice = rng.choice(candidates)
        votes[choice] += 1

    top_count = max(votes.values())
    top_candidates = [pid for pid, count in votes.items() if count == top_count]
    if len(top_candidates) > 1:
        return None
    return top_candidates[0]
