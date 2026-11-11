from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.engine.game_state import Role


_PROMPT_DIR = Path(__file__).resolve().parent


def role_persona(role: Role) -> str:
    return "\n\n".join([
        _read_prompt(f"persona_{role.value}.txt"),
        _read_prompt("game_rules.txt"),
        _read_prompt("language.txt"),
    ])


@lru_cache(maxsize=None)
def _read_prompt(filename: str) -> str:
    return (_PROMPT_DIR / filename).read_text(encoding="utf-8").strip()
