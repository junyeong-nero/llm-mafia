from __future__ import annotations

from typing import Literal

from src.config import AppConfig
from src.runner.graph_runner import run_graph_match
from src.runner.single_match import MatchResult, ProgressCallback, run_single_match


RunnerType = Literal["legacy", "graph"]


def run_match(
    config: AppConfig,
    *,
    seed: int | None = None,
    max_rounds: int = 10,
    progress_callback: ProgressCallback | None = None,
    runner: RunnerType = "graph",
) -> MatchResult:
    if runner == "graph":
        return run_graph_match(
            config,
            seed=seed,
            max_rounds=max_rounds,
            progress_callback=progress_callback,
        )
    return run_single_match(
        config,
        seed=seed,
        max_rounds=max_rounds,
        progress_callback=progress_callback,
    )
