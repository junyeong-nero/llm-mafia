from __future__ import annotations

from typing import Callable

from src.providers.chat_client import ChatClient, ChatSettings
from src.providers.settings import load_api_key


def load_exaone_settings() -> ChatSettings:
    return ChatSettings(api_key=load_api_key("EXAONE_API_KEY"))


class ExaoneClient(ChatClient):
    def __init__(
        self,
        settings: ChatSettings,
        *,
        on_retry: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        super().__init__(
            settings,
            url="https://api.exaone.ai/v1/chat/completions",
            request_options={"reasoning_effort": "low"},
            on_retry=on_retry,
        )
