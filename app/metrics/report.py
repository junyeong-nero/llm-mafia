from __future__ import annotations


def to_report_text(metrics: dict[str, object]) -> str:
    return "\n".join(
        [
            "Match Summary",
            f"- winner: {metrics.get('winner')}",
            f"- final_turn: {metrics.get('final_turn')}",
            f"- alive_count: {metrics.get('alive_count')}",
            f"- vote_rounds: {metrics.get('vote_rounds')}",
            f"- night_resolutions: {metrics.get('night_resolutions')}",
        ]
    )
