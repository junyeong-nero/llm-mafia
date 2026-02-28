from __future__ import annotations

from src.engine.game_state import Phase


def next_phase(phase: Phase) -> Phase:
    if phase == Phase.SETUP:
        return Phase.NIGHT
    if phase == Phase.NIGHT:
        return Phase.DAY
    if phase == Phase.DAY:
        return Phase.VOTE
    if phase == Phase.VOTE:
        return Phase.NIGHT
    return Phase.END
