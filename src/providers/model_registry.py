from __future__ import annotations

from src.config import AppConfig


def build_model_slots(config: AppConfig) -> list[tuple[str, str]]:
    slots: list[tuple[str, str]] = []
    for model in config.llm.models:
        for _ in range(model.count):
            slots.append((model.name, model.model))
    return slots
