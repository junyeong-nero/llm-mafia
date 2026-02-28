from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.engine.game_state import Role


_PROMPT_DIR = Path(__file__).resolve().parent


def role_persona(role: Role) -> str:
    if role == Role.MAFIA:
        return _read_prompt("persona_mafia.txt")
    if role == Role.POLICE:
        return _read_prompt("persona_police.txt")
    if role == Role.DOCTOR:
        return _read_prompt("persona_doctor.txt")
    return _read_prompt("persona_citizen.txt")


@lru_cache(maxsize=None)
def _read_prompt(filename: str) -> str:
    return (_PROMPT_DIR / filename).read_text(encoding="utf-8").strip()
