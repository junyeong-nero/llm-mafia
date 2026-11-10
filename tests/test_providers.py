from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest
import requests
import yaml

from src.config import load_config
from src.providers.chat_client import ChatSettings, ProviderError
from src.providers.exaone_client import ExaoneClient, load_exaone_settings
from src.providers.factory import build_provider_clients
from src.providers.openrouter_client import OpenRouterClient
from src.providers.settings import load_api_key
from src.runner.single_match import _build_match_state


def _response(data: object, status: int = 200) -> SimpleNamespace:
    return SimpleNamespace(status_code=status, headers={}, text="upstream error", json=lambda: data)


def test_exaone_uses_official_api_and_content_not_reasoning(monkeypatch: Any) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_post(url: str, **kwargs: Any) -> SimpleNamespace:
        calls.append((url, kwargs))
        return _response({
            "model": "k-exaone_v2.5",
            "choices": [{"message": {"content": "한국어 답변", "reasoning_content": "private reasoning"}, "finish_reason": "stop"}],
        })

    monkeypatch.setattr(requests, "post", fake_post)
    settings = ChatSettings(api_key="private-test-key", max_attempts=1)
    client = ExaoneClient(settings)
    result = client.chat_completion(model="k-exaone_v2.5", messages=[{"role": "user", "content": "안녕"}])
    assert result["text"] == "한국어 답변"
    url, kwargs = calls[0]
    assert url == "https://api.exaone.ai/v1/chat/completions"
    assert kwargs["headers"] == {"Authorization": "Bearer private-test-key", "Content-Type": "application/json"}
    assert kwargs["json"] == {
        "model": "k-exaone_v2.5", "messages": [{"role": "user", "content": "안녕"}],
        "stream": False, "reasoning_effort": "low", "max_tokens": 4096,
    }
    assert "private-test-key" not in repr(settings)


@pytest.mark.parametrize("data, error", [
    ([], "JSON object"),
    ({"choices": []}, "missing choices"),
    ({"choices": [{"message": {"content": None}}]}, "empty completion"),
    ({"choices": [{"message": {"reasoning_content": "thinking"}}]}, "empty completion"),
    ({"choices": [{"message": {"content": "VOTE:"}, "finish_reason": "length"}]}, "truncated"),
])
def test_invalid_completions_fail_instead_of_fake_responses(monkeypatch: Any, data: object, error: str) -> None:
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _response(data))
    with pytest.raises(ProviderError, match=error):
        ExaoneClient(ChatSettings(api_key="test", max_attempts=1)).chat_completion(model="k-exaone_v2.5", messages=[])


def test_retry_after_and_model_identity_in_retry_event(monkeypatch: Any) -> None:
    retry = _response({}, status=429)
    retry.headers = {"Retry-After": "2"}
    responses = iter([retry, _response({"choices": [{"message": {"content": "응답"}}]})])
    delays: list[float] = []
    events: list[dict[str, object]] = []
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr("src.providers.chat_client.time.sleep", delays.append)
    client = ExaoneClient(ChatSettings(api_key="test", max_attempts=2), on_retry=events.append)
    assert client.chat_completion(model="k-exaone_v2.5", messages=[])["text"] == "응답"
    assert delays == [2.0]
    assert events[0]["model"] == "k-exaone_v2.5"
    assert events[0]["status_code"] == 429


@pytest.fixture
def clean_credentials(monkeypatch: Any) -> None:
    monkeypatch.setattr("src.providers.settings.load_project_env_file", lambda: None)
    for key in ("EXAONE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv(f"{key}_COMMAND", raising=False)


def test_environment_key_takes_precedence_over_command(monkeypatch: Any, clean_credentials: None) -> None:
    monkeypatch.setenv("EXAONE_API_KEY", "env-key")
    monkeypatch.setenv("EXAONE_API_KEY_COMMAND", "must-not-run")
    assert load_exaone_settings().api_key == "env-key"


def test_keychain_command_is_captured_without_shell(monkeypatch: Any, clean_credentials: None) -> None:
    monkeypatch.setenv("EXAONE_API_KEY_COMMAND", "security find-generic-password -s pi-exaone-api-key -w")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(args: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append((args, kwargs))
        return SimpleNamespace(stdout="keychain-secret\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert load_api_key("EXAONE_API_KEY") == "keychain-secret"
    args, kwargs = calls[0]
    assert args == ["security", "find-generic-password", "-s", "pi-exaone-api-key", "-w"]
    assert kwargs["capture_output"] is True
    assert kwargs.get("shell", False) is False


def test_failed_credential_command_does_not_leak_output(monkeypatch: Any, clean_credentials: None) -> None:
    monkeypatch.setenv("EXAONE_API_KEY_COMMAND", "security -w")

    def fail(*args: Any, **kwargs: Any) -> None:
        raise subprocess.CalledProcessError(1, "security", output="secret", stderr="secret")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(ProviderError) as error:
        load_exaone_settings()
    assert str(error.value) == "EXAONE_API_KEY_COMMAND failed"
    assert error.value.__suppress_context__ is True


def test_missing_credentials_stop_match(monkeypatch: Any, clean_credentials: None) -> None:
    config = load_config(Path("config.yaml"))
    with pytest.raises(ProviderError, match="EXAONE_API_KEY is not set"):
        _build_match_state(config, seed=42, max_rounds=3, progress_callback=None)


def test_mixed_match_routes_each_player_to_its_provider(monkeypatch: Any, clean_credentials: None) -> None:
    monkeypatch.setenv("EXAONE_API_KEY", "exaone-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test")
    config = load_config(Path("config.compare.yaml"))
    clients = build_provider_clients(config)
    assert isinstance(clients["exaone"], ExaoneClient)
    assert isinstance(clients["openrouter"], OpenRouterClient)
    state, _, runtime = _build_match_state(config, seed=42, max_rounds=3, progress_callback=None)
    for player in state.players:
        agent = runtime.agents[player.id]
        expected = ExaoneClient if player.model_id == "k-exaone_v2.5" else OpenRouterClient
        assert isinstance(agent.client, expected)
        assert agent.model_id == player.model_id


@pytest.mark.parametrize("change, error", [
    ("provider", "provider must be"),
    ("model_provider", "provider must be"),
    ("duplicate_name", "duplicate model name"),
])
def test_config_rejects_invalid_provider_and_ambiguous_names(tmp_path: Path, change: str, error: str) -> None:
    raw = yaml.safe_load(Path("config.compare.yaml").read_text())
    if change == "provider":
        raw["llm"]["provider"] = "unknown"
    elif change == "model_provider":
        raw["llm"]["models"][0]["provider"] = "unknown"
    else:
        raw["llm"]["models"][1]["name"] = raw["llm"]["models"][0]["name"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=error):
        load_config(path)
