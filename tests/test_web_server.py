from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.web.match_session import MatchSession
from src.web.server import create_app

CONFIG_YAML = """
game:
    player_count: 4
    day_max_speeches_per_player: 2
    roles:
        mafia: 1
        police: 0
        doctor: 0
        citizen: 3

llm:
    provider: openrouter
    models:
        - name: m1
          model: model/one
          count: 2
        - name: m2
          model: model/two
          count: 2
"""


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_YAML, encoding="utf-8")
    return path


@pytest.fixture
def offline_match(monkeypatch: Any, tmp_path: Path) -> Path:
    sequence = iter(range(100))

    def fake_build_output_dir(base_dir: Path | None = None) -> Path:
        target = tmp_path / f"run-{next(sequence)}"
        target.mkdir(parents=True, exist_ok=False)
        return target

    monkeypatch.setattr("src.runner.single_match.build_output_dir", fake_build_output_dir)
    monkeypatch.setattr(
        "src.runner.single_match._build_provider_client",
        lambda config, progress_callback=None: None,
    )
    return _write_config(tmp_path)


def _fresh_client(config_path: Path) -> tuple[TestClient, MatchSession]:
    session = MatchSession()
    app = create_app(session)
    return TestClient(app), session


def test_index_page_serves_dashboard_html() -> None:
    client, _ = _fresh_client(Path("unused"))
    response = client.get("/")
    html = response.text

    assert response.status_code == 200
    assert "LLM Mafia" in html
    assert "Live chat stream" in html
    assert "/static/icons.js" in html
    assert "/static/app.js" in html


def test_static_assets_are_served() -> None:
    client, _ = _fresh_client(Path("unused"))

    assert client.get("/static/app.js").status_code == 200
    assert "function icon(" in client.get("/static/icons.js").text
    style = client.get("/static/style.css")
    assert style.status_code == 200
    assert ".console-card" in style.text


def test_match_state_reports_idle_before_start() -> None:
    client, _ = _fresh_client(Path("unused"))
    payload = client.get("/api/match/state").json()

    assert payload["status"] == "idle"
    assert payload["error_message"] is None
    assert payload["view"]["current"] == "Preparing game"
    assert "request" not in payload


def test_start_with_missing_config_returns_400(tmp_path: Path) -> None:
    client, _ = _fresh_client(tmp_path / "missing.yaml")
    response = client.post(
        "/api/match/start",
        json={"config_path": str(tmp_path / "nope.yaml")},
    )

    assert response.status_code == 400
    assert "Configuration error" in response.json()["detail"]


def test_start_with_invalid_runner_returns_422(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    client, _ = _fresh_client(config_path)
    response = client.post(
        "/api/match/start",
        json={"config_path": str(config_path), "runner": "turbo"},
    )

    assert response.status_code == 422


def test_start_with_malformed_config_returns_400(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("game:\n  player_count: 4\n", encoding="utf-8")
    client, _ = _fresh_client(bad)
    response = client.post("/api/match/start", json={"config_path": str(bad)})

    assert response.status_code == 400
    assert "Configuration error" in response.json()["detail"]


def test_basic_config_is_valid_for_session_start(tmp_path: Path) -> None:
    _write_config(tmp_path)
    from src.config import load_config

    config = load_config(tmp_path / "config.yaml")
    assert config.game.player_count == 4
    assert config.llm.models[0].name == "m1"


def test_start_runs_match_to_completion(
    offline_match: Path,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    client, session = _fresh_client(config_path)

    response = client.post(
        "/api/match/start",
        json={"config_path": str(config_path), "seed": 7, "max_rounds": 2},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "running"
    assert payload["request"] == {
        "config_path": str(config_path),
        "seed": 7,
        "max_rounds": 2,
        "runner": "graph",
    }

    session.wait()

    snapshot = client.get("/api/match/state").json()
    assert snapshot["status"] == "complete"
    assert snapshot["result"]["output_dir"]
    assert Path(snapshot["result"]["events_path"]).exists()
    assert Path(snapshot["result"]["summary_path"]).exists()

    streamed = client.get("/api/match/stream").text
    assert '"type": "match_complete"' in streamed
    assert '"kind": "setup"' in streamed


def test_start_rejects_second_run_while_running(
    offline_match: Path,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    client, session = _fresh_client(config_path)

    release = threading.Event()
    started = threading.Event()

    def blocking_run_match(config: Any, **kwargs: Any):
        started.set()
        release.wait(timeout=10)
        raise RuntimeError("released")

    monkeypatch.setattr("src.web.match_session.run_match", blocking_run_match)

    first = client.post("/api/match/start", json={"config_path": str(config_path)})
    assert first.status_code == 200
    assert started.wait(timeout=5)

    second = client.post("/api/match/start", json={"config_path": str(config_path)})
    assert second.status_code == 409

    release.set()
    session.wait()
