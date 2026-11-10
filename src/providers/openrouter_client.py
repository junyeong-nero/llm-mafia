from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from src.providers.chat_client import ChatClient, ChatSettings, ProviderError
from src.providers.settings import load_api_key


# Keep the public exception name for existing OpenRouter callers.
OpenRouterError = ProviderError


@dataclass(frozen=True)
class OpenRouterSettings(ChatSettings):
    referer: str | None = None
    title: str | None = None


def load_openrouter_settings() -> OpenRouterSettings:
    api_key = load_api_key("OPENROUTER_API_KEY")
    return OpenRouterSettings(
        api_key=api_key,
        referer=os.getenv("OPENROUTER_HTTP_REFERER"),
        title=os.getenv("OPENROUTER_APP_TITLE", "llm-mafia"),
    )


class OpenRouterClient(ChatClient):
    def __init__(
        self,
        settings: OpenRouterSettings,
        *,
        on_retry: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        headers: dict[str, str] = {}
        if settings.referer:
            headers["HTTP-Referer"] = settings.referer
        if settings.title:
            headers["X-OpenRouter-Title"] = settings.title
        super().__init__(
            settings,
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            on_retry=on_retry,
        )
