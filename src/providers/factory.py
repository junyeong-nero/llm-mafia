from __future__ import annotations

from typing import Callable

from src.config import AppConfig
from src.providers.chat_client import ChatClient
from src.providers.exaone_client import ExaoneClient, load_exaone_settings
from src.providers.openrouter_client import OpenRouterClient, load_openrouter_settings


def build_provider_clients(
    config: AppConfig,
    *,
    on_retry: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, ChatClient]:
    clients: dict[str, ChatClient] = {}
    for model in config.llm.models:
        provider = model.provider or config.llm.provider
        if provider in clients:
            continue
        if provider == "exaone":
            clients[provider] = ExaoneClient(load_exaone_settings(), on_retry=on_retry)
        elif provider == "openrouter":
            clients[provider] = OpenRouterClient(load_openrouter_settings(), on_retry=on_retry)
        else:
            raise ValueError(f"unsupported provider: {provider}")
    return clients
