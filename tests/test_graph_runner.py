from __future__ import annotations

from itertools import count
from pathlib import Path
from typing import Any

from src.config import AppConfig, GameConfig, LLMConfig, ModelConfig, RoleConfig
from src.engine.game_state import Phase
from src.runner.graph_runner import run_graph_match
from src.runner.match_runner import run_match
from src.runner.single_match import run_single_match


def _build_config() -> AppConfig:
    return AppConfig(
        game=GameConfig(
            player_count=4,
            roles=RoleConfig(mafia=1, police=0, doctor=0, citizen=3),
            day_max_speeches_per_player=2,
        ),
        llm=LLMConfig(
            provider="openrouter",
            models=[
                ModelConfig(name="m1", model="model/one", count=2),
                ModelConfig(name="m2", model="model/two", count=2),
            ],
        ),
    )


def _stub_output_dirs(monkeypatch: Any, tmp_path: Path) -> None:
    sequence = count()

    def fake_build_output_dir(base_dir: Path | None = None) -> Path:
        target = tmp_path / f"run-{next(sequence)}"
        target.mkdir(parents=True, exist_ok=False)
        return target

    monkeypatch.setattr("src.runner.single_match.build_output_dir", fake_build_output_dir)
    monkeypatch.setattr("src.runner.single_match._build_provider_client", lambda config, progress_callback=None: None)


def _event_signature(result: Any) -> list[tuple[int, object, str, str, str]]:
    return [
        (event.turn, event.phase, event.speaker, event.kind, event.content)
        for event in result.events
    ]


def _assert_progress_payload_shape(payloads: list[dict[str, object]]) -> None:
    assert payloads
    for payload in payloads:
        assert isinstance(payload["kind"], str)
        if payload["kind"] != "provider_retry":
            assert isinstance(payload.get("turn"), int)
            assert isinstance(payload.get("phase"), str)
        if "players_status" in payload:
            players_status = payload["players_status"]
            assert isinstance(players_status, list)
            for player in players_status:
                assert isinstance(player, dict)
                assert {"name", "model_name", "role", "alive"} <= set(player)
        if payload["kind"] in {"day_vote", "mafia_vote"}:
            assert isinstance(payload.get("target_name"), str)
        if payload["kind"] == "speech_queue":
            speech_queue = payload.get("speech_queue")
            assert isinstance(speech_queue, list)
            assert all(isinstance(item, str) for item in speech_queue)


def test_graph_runner_completes_full_match(monkeypatch: Any, tmp_path: Path) -> None:
    _stub_output_dirs(monkeypatch, tmp_path)
    progress_payloads: list[dict[str, object]] = []

    result = run_graph_match(
        _build_config(),
        seed=7,
        max_rounds=3,
        progress_callback=progress_payloads.append,
    )

    assert result.state.phase == Phase.END
    assert result.state.winner in {"citizen", "mafia", "draw"}
    assert result.events[-1].kind == "game_end"
    assert result.output_dir.exists()
    assert result.events_path.exists()
    assert result.summary_path.exists()
    assert progress_payloads[0]["kind"] == "setup"
    assert any(payload["kind"] == "phase" for payload in progress_payloads)
    assert progress_payloads[-1]["kind"] == "game_end"
    _assert_progress_payload_shape(progress_payloads)


def test_graph_runner_matches_legacy_for_seeded_match(monkeypatch: Any, tmp_path: Path) -> None:
    _stub_output_dirs(monkeypatch, tmp_path)
    legacy_progress: list[dict[str, object]] = []
    graph_progress: list[dict[str, object]] = []
    config = _build_config()

    legacy_result = run_single_match(
        config,
        seed=11,
        max_rounds=3,
        progress_callback=legacy_progress.append,
    )
    graph_result = run_graph_match(
        config,
        seed=11,
        max_rounds=3,
        progress_callback=graph_progress.append,
    )

    assert graph_result.state.winner == legacy_result.state.winner
    assert graph_result.metrics == legacy_result.metrics
    assert _event_signature(graph_result) == _event_signature(legacy_result)
    assert graph_progress == legacy_progress
    _assert_progress_payload_shape(graph_progress)


def test_graph_runner_matches_legacy_across_multiple_seeds(monkeypatch: Any, tmp_path: Path) -> None:
    _stub_output_dirs(monkeypatch, tmp_path)
    config = _build_config()

    for seed in (1, 7, 11):
        legacy_result = run_single_match(config, seed=seed, max_rounds=3)
        graph_result = run_graph_match(config, seed=seed, max_rounds=3)

        assert graph_result.state.winner == legacy_result.state.winner
        assert graph_result.metrics == legacy_result.metrics
        assert _event_signature(graph_result) == _event_signature(legacy_result)


def test_run_match_dispatches_to_requested_runner(monkeypatch: Any) -> None:
    config = _build_config()
    calls: list[str] = []

    def fake_legacy(
        config: AppConfig,
        *,
        seed: int | None = None,
        max_rounds: int = 10,
        progress_callback: Any = None,
    ) -> str:
        calls.append(f"legacy:{seed}:{max_rounds}:{progress_callback is not None}")
        return "legacy"

    def fake_graph(
        config: AppConfig,
        *,
        seed: int | None = None,
        max_rounds: int = 10,
        progress_callback: Any = None,
    ) -> str:
        calls.append(f"graph:{seed}:{max_rounds}:{progress_callback is not None}")
        return "graph"

    monkeypatch.setattr("src.runner.match_runner.run_single_match", fake_legacy)
    monkeypatch.setattr("src.runner.match_runner.run_graph_match", fake_graph)

    assert run_match(config, seed=3, max_rounds=5, runner="legacy") == "legacy"
    assert run_match(config, seed=4, max_rounds=6, progress_callback=list.append, runner="graph") == "graph"
    assert calls == ["legacy:3:5:False", "graph:4:6:True"]
