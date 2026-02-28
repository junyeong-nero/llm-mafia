from __future__ import annotations

from typing import Any

import requests

from src.providers.openrouter_client import OpenRouterClient, OpenRouterError, OpenRouterSettings


class _FakeResponse:
    def __init__(self, status_code: int, data: dict[str, object] | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._data = data
        self.text = text
        self.headers: dict[str, str] = {}

    def json(self) -> dict[str, object]:
        if self._data is None:
            raise ValueError("invalid json")
        return self._data


def test_chat_completion_tries_next_model_on_404(monkeypatch: Any) -> None:
    settings = OpenRouterSettings(api_key="test", max_attempts=1)
    client = OpenRouterClient(settings)
    calls: list[str] = []

    def fake_post(*args: Any, **kwargs: Any) -> _FakeResponse:
        payload = kwargs["json"]
        model = str(payload["model"])
        calls.append(model)
        if model == "bad/model":
            return _FakeResponse(404, text="model not found")
        return _FakeResponse(
            200,
            data={
                "model": model,
                "choices": [
                    {
                        "message": {"content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    monkeypatch.setattr(requests, "post", fake_post)

    result = client.chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        model="bad/model",
        fallback_models=["good/model"],
    )

    assert result["text"] == "hello"
    assert calls == ["bad/model", "good/model"]


def test_chat_completion_raises_openrouter_error_on_invalid_json(monkeypatch: Any) -> None:
    settings = OpenRouterSettings(api_key="test", max_attempts=1)
    client = OpenRouterClient(settings)

    def fake_post(*args: Any, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(200, data=None)

    monkeypatch.setattr(requests, "post", fake_post)

    try:
        client.chat_completion(messages=[{"role": "user", "content": "hi"}], model="good/model")
    except OpenRouterError as exc:
        assert "invalid JSON response" in str(exc)
    else:
        raise AssertionError("Expected OpenRouterError for invalid JSON response")
