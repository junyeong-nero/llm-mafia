from __future__ import annotations

from dataclasses import dataclass

from src.engine.game_state import GameState, Role


@dataclass
class WinStats:
    appearances: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    def record(self, *, won: bool, draw: bool) -> None:
        self.appearances += 1
        if draw:
            self.draws += 1
        elif won:
            self.wins += 1
        else:
            self.losses += 1

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "appearances": self.appearances,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.wins / self.appearances if self.appearances else None,
        }


def collect_model_stats(states: list[GameState]) -> dict[str, dict[str, object]]:
    """Count a player's team outcome, including eliminated players and draws."""
    totals: dict[str, WinStats] = {}
    by_role: dict[str, dict[str, WinStats]] = {}
    by_team: dict[str, dict[str, WinStats]] = {}
    model_ids: dict[str, str] = {}
    for state in states:
        if state.winner not in {"mafia", "citizen", "draw"}:
            raise ValueError("cannot score an unfinished match")
        for player in state.players:
            name = player.model_name
            if name not in totals:
                totals[name] = WinStats()
                by_role[name] = {role.value: WinStats() for role in Role}
                by_team[name] = {team: WinStats() for team in ("mafia", "citizen")}
                model_ids[name] = player.model_id
            elif model_ids[name] != player.model_id:
                raise ValueError(f"model name maps to multiple model IDs: {name}")
            team = "mafia" if player.role == Role.MAFIA else "citizen"
            for stats in (totals[name], by_role[name][player.role.value], by_team[name][team]):
                stats.record(won=state.winner == team, draw=state.winner == "draw")
    return {
        name: {
            "model": model_ids[name],
            **stats.to_dict(),
            "by_role": {role: counts.to_dict() for role, counts in by_role[name].items()},
            "by_team": {team: counts.to_dict() for team, counts in by_team[name].items()},
        }
        for name, stats in totals.items()
    }


def comparison_report(model_stats: dict[str, dict[str, object]]) -> str:
    lines = [
        "모델별 팀 승률 (승리한 진영의 플레이어 수 / 전체 플레이어 출전 수; 무승부 포함)",
        "model | 출전 | 승 | 패 | 무 | 승률",
    ]
    for name, stats in model_stats.items():
        rate = stats["win_rate"]
        rate_text = f"{rate:.1%}" if isinstance(rate, (int, float)) else "N/A"
        lines.append(
            f"{name} | {stats['appearances']} | {stats['wins']} | "
            f"{stats['losses']} | {stats['draws']} | {rate_text}"
        )
    lines.append("같은 경기의 출전 결과는 독립 표본이 아닙니다. 소규모 결과로 모델 우열을 확정하지 마세요.")
    return "\n".join(lines)
