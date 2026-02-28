from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests


RETRYABLE_STATUSES = {408, 429, 500, 502, 503}


class OpenRouterError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenRouterSettings:
    api_key: str
    referer: str | None = None
    title: str | None = None
    timeout_seconds: float = 45.0
    max_attempts: int = 4


def load_openrouter_settings() -> OpenRouterSettings:
    _load_project_env_file()
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise OpenRouterError("OPENROUTER_API_KEY is not set")
    return OpenRouterSettings(
        api_key=api_key,
        referer=os.getenv("OPENROUTER_HTTP_REFERER"),
            title=os.getenv("OPENROUTER_APP_TITLE", "llm-mafia"),
    )


class OpenRouterClient:
    def __init__(
        self,
        settings: OpenRouterSettings,
        *,
        on_retry: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self._settings = settings
        self._url = "https://openrouter.ai/api/v1/chat/completions"
        self._on_retry = on_retry

    def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        fallback_models: list[str] | None = None,
    ) -> dict[str, object]:
        if not model:
            raise OpenRouterError("model must not be empty")

        model_candidates = [model]
        if fallback_models:
            model_candidates.extend(fallback_models)

        headers = {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
        }
        if self._settings.referer:
            headers["HTTP-Referer"] = self._settings.referer
        if self._settings.title:
            headers["X-OpenRouter-Title"] = self._settings.title

        payload_base: dict[str, object] = {
            "messages": messages,
            "stream": False,
        }

        last_error: str | None = None
        for model_index, current_model in enumerate(model_candidates):
            payload = dict(payload_base)
            payload["model"] = current_model
            if model_index == 0 and len(model_candidates) > 1:
                payload["models"] = model_candidates

            result, last_error, should_try_next_model = self._attempt_completion_with_model(
                headers=headers,
                payload=payload,
                model_name=current_model,
            )
            if result is not None:
                return result
            if should_try_next_model:
                continue
            break

        raise OpenRouterError(last_error or "openrouter call failed")

    def _attempt_completion_with_model(
        self,
        *,
        headers: dict[str, str],
        payload: dict[str, object],
        model_name: str,
    ) -> tuple[dict[str, object] | None, str | None, bool]:
        last_error: str | None = None
        for attempt in range(self._settings.max_attempts):
            try:
                response = requests.post(
                    self._url,
                    headers=headers,
                    json=payload,
                    timeout=self._settings.timeout_seconds,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = str(exc)
                if attempt < self._settings.max_attempts - 1:
                    delay_seconds = _backoff_delay(attempt)
                    self._emit_retry(
                        {
                            "attempt": attempt + 1,
                            "max_attempts": self._settings.max_attempts,
                            "reason": "network_failure",
                            "detail": last_error,
                            "delay_seconds": delay_seconds,
                        }
                    )
                    time.sleep(delay_seconds)
                    continue
                return None, f"network failure: {last_error}", False

            if response.status_code in RETRYABLE_STATUSES:
                last_error = f"http {response.status_code}"
                if attempt < self._settings.max_attempts - 1:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        delay_seconds = float(retry_after)
                    else:
                        delay_seconds = _backoff_delay(attempt)
                    self._emit_retry(
                        {
                            "attempt": attempt + 1,
                            "max_attempts": self._settings.max_attempts,
                            "reason": "retryable_http_status",
                            "detail": last_error,
                            "status_code": response.status_code,
                            "delay_seconds": delay_seconds,
                        }
                    )
                    time.sleep(delay_seconds)
                    continue
                return None, f"retryable failure exhausted: {last_error}", False

            if response.status_code != 200:
                error_message = f"http {response.status_code}: {response.text[:500]}"
                should_try_next_model = response.status_code in {400, 404}
                if should_try_next_model:
                    prefixed_error = f"model={model_name} failed ({error_message})"
                    return None, prefixed_error, True
                return None, error_message, False

            try:
                data = response.json()
            except ValueError as exc:
                last_error = f"invalid JSON response: {exc}"
                if attempt < self._settings.max_attempts - 1:
                    delay_seconds = _backoff_delay(attempt)
                    self._emit_retry(
                        {
                            "attempt": attempt + 1,
                            "max_attempts": self._settings.max_attempts,
                            "reason": "invalid_json",
                            "detail": last_error,
                            "delay_seconds": delay_seconds,
                        }
                    )
                    time.sleep(delay_seconds)
                    continue
                return None, last_error, False

            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                last_error = "missing choices in response"
                if attempt < self._settings.max_attempts - 1:
                    delay_seconds = _backoff_delay(attempt)
                    self._emit_retry(
                        {
                            "attempt": attempt + 1,
                            "max_attempts": self._settings.max_attempts,
                            "reason": "missing_choices",
                            "detail": last_error,
                            "delay_seconds": delay_seconds,
                        }
                    )
                    time.sleep(delay_seconds)
                    continue
                return None, last_error, False

            first_choice = choices[0]
            message = first_choice.get("message") if isinstance(first_choice, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                last_error = "empty completion content"
                if attempt < self._settings.max_attempts - 1:
                    delay_seconds = _backoff_delay(attempt)
                    self._emit_retry(
                        {
                            "attempt": attempt + 1,
                            "max_attempts": self._settings.max_attempts,
                            "reason": "empty_content",
                            "detail": last_error,
                            "delay_seconds": delay_seconds,
                        }
                    )
                    time.sleep(delay_seconds)
                    continue
                return None, last_error, False

            return {
                "text": content,
                "model": data.get("model"),
                "finish_reason": first_choice.get("finish_reason") if isinstance(first_choice, dict) else None,
                "raw": data,
            }, None, False

        return None, last_error or "openrouter call failed", False

    def _emit_retry(self, event: dict[str, object]) -> None:
        if self._on_retry is None:
            return
        self._on_retry(event)


def _backoff_delay(attempt: int) -> float:
    return min(8.0, 0.5 * (2**attempt)) + random.uniform(0.0, 0.25)


def _load_project_env_file() -> None:
    env_path = _project_env_path()
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _project_env_path() -> Path:
    return Path(__file__).resolve().parents[2] / ".env"
