from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from src.config import AppConfig
from src.io.event_log import build_output_dir, write_summary_json
from src.metrics.comparison import collect_model_stats
from src.runner.match_runner import MatchResult, RunnerType, run_match
from src.runner.single_match import ProgressCallback


@dataclass(frozen=True)
class TournamentResult:
    matches: list[MatchResult]
    win_counts: dict[str, int]
    model_stats: dict[str, dict[str, object]]
    summary_path: Path


def run_tournament(
    config: AppConfig,
    *,
    rounds: int,
    seed: int = 42,
    max_rounds: int = 10,
    runner: RunnerType = "graph",
    balanced: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> TournamentResult:
    if rounds <= 0:
        raise ValueError("rounds must be > 0")
    if max_rounds <= 0:
        raise ValueError("max_rounds must be > 0")
    model_count = len(config.llm.models)
    if balanced:
        if model_count < 2:
            raise ValueError("balanced comparison requires at least two models")
        if len({model.count for model in config.llm.models}) != 1:
            raise ValueError("balanced comparison requires equal model counts")
        if rounds % model_count:
            raise ValueError("rounds must be a multiple of the number of models for balanced comparison")

    output_dir = build_output_dir(Path("logs") / "tournaments")
    matches: list[MatchResult] = []
    win_counts: dict[str, int] = {}
    records: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "status": "running",
        "config": asdict(config),
        "seed": seed,
        "requested_matches": rounds,
        "completed_matches": 0,
        "max_rounds": max_rounds,
        "runner": runner,
        "balanced": balanced,
        "language": "ko",
        "scoring": "team wins per player appearance, including eliminated players; draws stay in denominator",
        "win_counts": win_counts,
        "matches": records,
        "model_stats": {},
    }
    summary_path = write_summary_json(summary, output_dir)
    for index in range(rounds):
        offset = index % model_count if balanced else 0
        match_seed = seed + (index // model_count if balanced else index)
        rotated = config.llm.models[offset:] + config.llm.models[:offset]
        match_config = replace(config, llm=replace(config.llm, models=rotated))
        try:
            if progress_callback is not None:
                progress_callback({
                    "kind": "tournament_match",
                    "match": index + 1,
                    "total": rounds,
                    "seed": match_seed,
                    "rotation": offset,
                    "summary_path": str(summary_path),
                })
            result = run_match(
                match_config, seed=match_seed, max_rounds=max_rounds,
                runner=runner, progress_callback=progress_callback,
            )
        except Exception as exc:
            summary.update(status="failed", failed_match=index + 1, error=str(exc))
            write_summary_json(summary, output_dir)
            raise RuntimeError(f"Tournament stopped; partial results: {summary_path}: {exc}") from exc

        matches.append(result)
        winner = result.state.winner or "draw"
        win_counts[winner] = win_counts.get(winner, 0) + 1
        records.append({
            "match": index + 1,
            "seed": match_seed,
            "rotation": offset,
            "model_order": [model.name for model in rotated],
            "winner": winner,
            "summary_path": str(result.summary_path),
            "events_path": str(result.events_path),
            "invalid_votes": sum(event.kind in {"day_vote_invalid", "mafia_vote_invalid"} for event in result.events),
        })
        summary["completed_matches"] = len(matches)
        summary["model_stats"] = collect_model_stats([match.state for match in matches])
        summary["status"] = "completed" if len(matches) == rounds else "running"
        write_summary_json(summary, output_dir)

    return TournamentResult(
        matches=matches,
        win_counts=win_counts,
        model_stats=collect_model_stats([match.state for match in matches]),
        summary_path=summary_path,
    )
