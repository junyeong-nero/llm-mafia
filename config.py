from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RoleConfig:
    mafia: int
    police: int
    doctor: int
    citizen: int


@dataclass(frozen=True)
class GameConfig:
    player_count: int
    roles: RoleConfig
    day_max_speeches_per_player: int


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model: str
    count: int


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    models: list[ModelConfig]


@dataclass(frozen=True)
class AppConfig:
    game: GameConfig
    llm: LLMConfig


def _expect_mapping(data: object, key: str) -> dict[str, object]:
    if not isinstance(data, dict):
        raise ValueError(f"'{key}' must be a mapping")
    return data


def _expect_int(data: dict[str, object], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        raise ValueError(f"'{key}' must be an integer")
    return value


def _expect_non_negative_int(data: dict[str, object], key: str) -> int:
    value = _expect_int(data, key)
    if value < 0:
        raise ValueError(f"'{key}' must be >= 0")
    return value


def _expect_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' must be a non-empty string")
    return value


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in '{path}': {exc}") from exc

    root = _expect_mapping(raw, "root")
    game_raw = _expect_mapping(root.get("game"), "game")
    llm_raw = _expect_mapping(root.get("llm"), "llm")

    roles_raw = _expect_mapping(game_raw.get("roles"), "game.roles")
    roles = RoleConfig(
        mafia=_expect_non_negative_int(roles_raw, "mafia"),
        police=_expect_non_negative_int(roles_raw, "police"),
        doctor=_expect_non_negative_int(roles_raw, "doctor"),
        citizen=_expect_non_negative_int(roles_raw, "citizen"),
    )
    player_count = _expect_int(game_raw, "player_count")
    if player_count <= 0:
        raise ValueError("'player_count' must be > 0")

    role_total = roles.mafia + roles.police + roles.doctor + roles.citizen
    if role_total != player_count:
        raise ValueError(
            "sum of roles must match game.player_count "
            f"(player_count={player_count}, roles_total={role_total})"
        )

    day_max_speeches_per_player = game_raw.get("day_max_speeches_per_player", 2)
    if not isinstance(day_max_speeches_per_player, int):
        raise ValueError("'day_max_speeches_per_player' must be an integer")
    if day_max_speeches_per_player <= 0:
        raise ValueError("'day_max_speeches_per_player' must be > 0")

    models_raw = llm_raw.get("models")
    if not isinstance(models_raw, list) or not models_raw:
        raise ValueError("'llm.models' must be a non-empty list")

    models: list[ModelConfig] = []
    model_total = 0
    for index, model_raw in enumerate(models_raw):
        model_map = _expect_mapping(model_raw, f"llm.models[{index}]")
        model = ModelConfig(
            name=_expect_str(model_map, "name"),
            model=_expect_str(model_map, "model"),
            count=_expect_int(model_map, "count"),
        )
        if model.count <= 0:
            raise ValueError(f"llm.models[{index}].count must be > 0")
        models.append(model)
        model_total += model.count

    if model_total != player_count:
        raise ValueError(
            "sum of llm model counts must match game.player_count "
            f"(player_count={player_count}, model_total={model_total})"
        )

    llm = LLMConfig(provider=_expect_str(llm_raw, "provider"), models=models)
    if llm.provider != "openrouter":
        raise ValueError("'llm.provider' must be 'openrouter'")

    game = GameConfig(
        player_count=player_count,
        roles=roles,
        day_max_speeches_per_player=day_max_speeches_per_player,
    )
    return AppConfig(game=game, llm=llm)
