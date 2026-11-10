from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable

import requests


RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatSettings:
    api_key: str = field(repr=False)
    timeout_seconds: float = 120.0
    max_attempts: int = 4
    max_tokens: int = 4096


class ChatClient:
    """Shared non-streaming transport for OpenAI-compatible chat APIs."""

    def __init__(
        self,
        settings: ChatSettings,
        *,
        url: str,
        headers: dict[str, str] | None = None,
        request_options: dict[str, object] | None = None,
        on_retry: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self._settings = settings
        self._url = url
        self._headers = headers or {}
        self._request_options = request_options or {}
        self._on_retry = on_retry

    def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        fallback_models: list[str] | None = None,
    ) -> dict[str, object]:
        if not model:
            raise ProviderError("model must not be empty")

        headers = {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
            **self._headers,
        }
        # Fallback is opt-in for transport callers; game agents never enable it.
        model_candidates = [model, *(fallback_models or [])]
        last_error: str | None = None
        for current_model in model_candidates:
            payload: dict[str, object] = {
                "messages": messages,
                "stream": False,
                "max_tokens": self._settings.max_tokens,
                **self._request_options,
                "model": current_model,
            }
            result, last_error, try_next = self._attempt_completion_with_model(
                headers=headers, payload=payload, model_name=current_model,
            )
            if result is not None:
                return result
            if not try_next:
                break
        raise ProviderError(f"model={model}: {last_error or 'chat completion failed'}")

    def _attempt_completion_with_model(
        self,
        *,
        headers: dict[str, str],
        payload: dict[str, object],
        model_name: str,
    ) -> tuple[dict[str, object] | None, str | None, bool]:
        last_error = "chat completion failed"
        for attempt in range(self._settings.max_attempts):
            retry_after: str | None = None
            status_code: int | None = None
            try:
                response = requests.post(
                    self._url,
                    headers=headers,
                    json=payload,
                    timeout=self._settings.timeout_seconds,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = f"network failure: {exc}"
                reason = "network_failure"
            else:
                status_code = response.status_code
                if status_code in RETRYABLE_STATUSES:
                    last_error = f"http {status_code}: {response.text[:500]}"
                    reason = "retryable_http_status"
                    retry_after = response.headers.get("Retry-After")
                elif status_code != 200:
                    return None, f"http {status_code}: {response.text[:500]}", status_code in {400, 404}
                else:
                    try:
                        data = response.json()
                    except ValueError as exc:
                        last_error = f"invalid JSON response: {exc}"
                        reason = "invalid_json"
                    else:
                        result, last_error, reason = _parse_completion(data)
                        if result is not None:
                            return result, None, False
                        if reason == "truncated_content":
                            return None, last_error, False

            if attempt < self._settings.max_attempts - 1:
                delay = float(retry_after) if retry_after and retry_after.isdigit() else _backoff_delay(attempt)
                if self._on_retry is not None:
                    self._on_retry({
                        "model": model_name,
                        "attempt": attempt + 1,
                        "max_attempts": self._settings.max_attempts,
                        "reason": reason,
                        "detail": last_error,
                        "status_code": status_code,
                        "delay_seconds": delay,
                    })
                time.sleep(delay)
        return None, last_error, False


def _parse_completion(data: object) -> tuple[dict[str, object] | None, str, str]:
    if not isinstance(data, dict):
        return None, "response must be a JSON object", "invalid_json"
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, "missing choices in response", "missing_choices"
    first_choice = choices[0]
    message = first_choice.get("message") if isinstance(first_choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    finish_reason = first_choice.get("finish_reason") if isinstance(first_choice, dict) else None
    if finish_reason == "length":
        return None, "completion was truncated by the token limit", "truncated_content"
    if not isinstance(content, str) or not content.strip():
        return None, "empty completion content", "empty_content"
    return {
        "text": content,
        "model": data.get("model"),
        "finish_reason": finish_reason,
        "raw": data,
    }, "", ""


def _backoff_delay(attempt: int) -> float:
    return min(8.0, 0.5 * (2**attempt)) + random.uniform(0.0, 0.25)
