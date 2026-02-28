from __future__ import annotations

import random

from src.engine.game_state import GameState, Phase, Player, Role


COMMON_AGENT_NAMES = [
    "Alex",
    "Emma",
    "Noah",
    "Olivia",
    "Liam",
    "Ava",
    "Ethan",
    "Mia",
    "Lucas",
    "Sophia",
    "Mason",
    "Isabella",
    "Logan",
    "Amelia",
    "James",
    "Charlotte",
]


def build_players(
    role_counts: dict[Role, int],
    model_slots: list[tuple[str, str]],
    *,
    seed: int,
) -> list[Player]:
    roles: list[Role] = []
    for role, count in role_counts.items():
        roles.extend([role] * count)

    if len(roles) != len(model_slots):
        raise ValueError("role count must match model slots")

    rng = random.Random(seed)
    rng.shuffle(roles)

    players: list[Player] = []
    assigned_names: dict[str, int] = {}
    for idx, ((model_name, model_id), role) in enumerate(zip(model_slots, roles, strict=True), start=1):
        base_name = COMMON_AGENT_NAMES[(idx - 1) % len(COMMON_AGENT_NAMES)]
        assigned_names[base_name] = assigned_names.get(base_name, 0) + 1
        suffix = assigned_names[base_name]
        agent_name = base_name if suffix == 1 else f"{base_name}{suffix}"
        players.append(
            Player(
                id=idx,
                name=agent_name,
                model_name=model_name,
                model_id=model_id,
                role=role,
            )
        )
    return players


def check_winner(state: GameState) -> str | None:
    mafia_count = len(state.alive_by_role(Role.MAFIA))
    citizen_count = len([p for p in state.alive_players() if p.role != Role.MAFIA])
    if mafia_count == 0:
        return "citizen"
    if mafia_count >= citizen_count:
        return "mafia"
    return None


def resolve_night(
    state: GameState,
    *,
    seed: int,
    mafia_target: int | None = None,
) -> tuple[int | None, int | None, int | None, int | None]:
    rng = random.Random(seed + state.turn)
    alive = state.alive_players()
    alive_non_mafia = [p for p in alive if p.role != Role.MAFIA]
    if not alive_non_mafia:
        return None, None, None, None

    alive_non_mafia_ids = {player.id for player in alive_non_mafia}
    selected_mafia_target = mafia_target if mafia_target in alive_non_mafia_ids else rng.choice(alive_non_mafia).id
    doctor_players = state.alive_by_role(Role.DOCTOR)
    doctor_target = rng.choice(alive).id if doctor_players else None
    police_players = state.alive_by_role(Role.POLICE)
    police_target = rng.choice(alive).id if police_players else None

    killed = None if doctor_target == selected_mafia_target else selected_mafia_target
    return killed, selected_mafia_target, doctor_target, police_target


def phase_label(phase: Phase) -> str:
    if phase == Phase.DAY:
        return "day"
    if phase == Phase.NIGHT:
        return "night"
    if phase == Phase.VOTE:
        return "vote"
    if phase == Phase.SETUP:
        return "setup"
    return "end"
