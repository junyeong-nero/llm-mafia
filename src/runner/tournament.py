from __future__ import annotations

from dataclasses import dataclass

from src.config import AppConfig

from src.runner.single_match import MatchResult, run_single_match


@dataclass(frozen=True)
class TournamentResult:
    matches: list[MatchResult]
    win_counts: dict[str, int]


def run_tournament(config: AppConfig, *, rounds: int, seed: int = 42) -> TournamentResult:
    matches: list[MatchResult] = []
    win_counts: dict[str, int] = {}
    for idx in range(rounds):
        result = run_single_match(config, seed=seed + idx)
        matches.append(result)
        winner = result.state.winner or "draw"
        win_counts[winner] = win_counts.get(winner, 0) + 1
    return TournamentResult(matches=matches, win_counts=win_counts)
