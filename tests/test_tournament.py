from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.config import load_config
from src.engine.game_state import GameState, Phase, Player, Role
from src.engine.rules import build_players
from src.metrics.comparison import collect_model_stats
from src.providers.chat_client import ProviderError
from src.runner.tournament import run_tournament


def _state(winner: str | None, *, swap_roles: bool = False) -> GameState:
    roles = [Role.MAFIA, Role.CITIZEN] if not swap_roles else [Role.DOCTOR, Role.MAFIA]
    return GameState(turn=2, phase=Phase.END, winner=winner, players=[
        Player(id=1, name="Alex", model_name="a", model_id="a/id", role=roles[0], alive=False),
        Player(id=2, name="Emma", model_name="b", model_id="b/id", role=roles[1], alive=True),
    ])


def test_model_win_rate_counts_team_outcome_dead_players_and_draws() -> None:
    stats = collect_model_stats([_state("mafia"), _state("citizen", swap_roles=True), _state("draw")])
    assert stats["a"]["appearances"] == 3
    assert stats["a"]["wins"] == 2
    assert stats["a"]["losses"] == 0
    assert stats["a"]["draws"] == 1
    assert stats["a"]["win_rate"] == pytest.approx(2 / 3)
    assert stats["a"]["by_role"]["mafia"]["wins"] == 1
    assert stats["a"]["by_role"]["doctor"]["wins"] == 1
    assert stats["a"]["by_role"]["police"]["win_rate"] is None
    assert stats["a"]["by_team"]["citizen"]["appearances"] == 1
    assert stats["b"]["losses"] == 2
    assert stats["b"]["win_rate"] == 0


def test_unfinished_matches_are_not_scored() -> None:
    with pytest.raises(ValueError, match="unfinished match"):
        collect_model_stats([_state(None)])


def test_empty_stats_are_not_reported_as_zero_percent() -> None:
    assert collect_model_stats([]) == {}


def _stub_output(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr("src.runner.tournament.build_output_dir", lambda base_dir: tmp_path)


def test_balanced_rotation_equalizes_roles_seats_and_seeds(monkeypatch: Any, tmp_path: Path) -> None:
    _stub_output(monkeypatch, tmp_path)
    config = load_config(Path("config.compare.yaml"))
    calls: list[dict[str, object]] = []
    roles_by_model: dict[str, Counter] = defaultdict(Counter)
    seats_by_model: dict[str, Counter] = defaultdict(Counter)

    def fake_match(config: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        role_counts = {role: getattr(config.game.roles, role.value) for role in Role}
        slots = [(model.name, model.model) for model in config.llm.models for _ in range(model.count)]
        players = build_players(role_counts, slots, seed=kwargs["seed"])
        for player in players:
            roles_by_model[player.model_name][player.role] += 1
            seats_by_model[player.model_name][player.id] += 1
        return SimpleNamespace(
            state=GameState(turn=2, phase=Phase.END, winner="citizen", players=players),
            events=[], summary_path=tmp_path / "match.json", events_path=tmp_path / "events.jsonl",
        )

    monkeypatch.setattr("src.runner.tournament.run_match", fake_match)
    result = run_tournament(config, rounds=6, seed=7, max_rounds=4, runner="legacy", balanced=True)
    names = [model.name for model in config.llm.models]
    assert roles_by_model[names[0]] == roles_by_model[names[1]] == roles_by_model[names[2]]
    assert seats_by_model[names[0]] == seats_by_model[names[1]] == seats_by_model[names[2]]
    assert [call["seed"] for call in calls] == [7, 7, 7, 8, 8, 8]
    assert all(call["max_rounds"] == 4 and call["runner"] == "legacy" for call in calls)
    assert result.win_counts == {"citizen": 6}
    report = json.loads(result.summary_path.read_text())
    assert report["status"] == "completed"
    assert report["completed_matches"] == 6
    assert report["language"] == "ko"
    assert report["model_stats"] == result.model_stats


@pytest.mark.parametrize("rounds", [0, -1, 1, 2, 4])
def test_balanced_tournament_rejects_invalid_match_counts(rounds: int) -> None:
    config = load_config(Path("config.compare.yaml"))
    with pytest.raises(ValueError, match="rounds"):
        run_tournament(config, rounds=rounds, balanced=True)


def test_balanced_tournament_rejects_unequal_model_counts() -> None:
    config = load_config(Path("config.compare.yaml"))
    models = [replace(model, count=index + 1) for index, model in enumerate(config.llm.models)]
    config = replace(config, llm=replace(config.llm, models=models))
    with pytest.raises(ValueError, match="equal model counts"):
        run_tournament(config, rounds=3, balanced=True)


def test_failed_match_is_excluded_and_completed_results_are_saved(monkeypatch: Any, tmp_path: Path) -> None:
    _stub_output(monkeypatch, tmp_path)
    config = load_config(Path("config.compare.yaml"))
    calls = 0

    def fake_match(*args: Any, **kwargs: Any) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ProviderError("rate limited")
        return SimpleNamespace(
            state=_state("mafia"), events=[],
            summary_path=tmp_path / "match.json", events_path=tmp_path / "events.jsonl",
        )

    monkeypatch.setattr("src.runner.tournament.run_match", fake_match)
    with pytest.raises(RuntimeError, match="partial results"):
        run_tournament(config, rounds=3, balanced=True)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["completed_matches"] == 1
    assert summary["failed_match"] == 2
    assert summary["win_counts"] == {"mafia": 1}
    assert summary["model_stats"]["a"]["appearances"] == 1
    assert calls == 2
