from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class NightResolution:
    killed: int | None
    mafia_target: int | None
    investigations: dict[int, bool]


def resolve_night(
    state: GameState,
    *,
    seed: int,
    mafia_target: int | None = None,
    doctor_targets: dict[int, int] | None = None,
    police_targets: dict[int, int] | None = None,
) -> NightResolution:
    alive_by_id = {player.id: player for player in state.alive_players()}
    doctor_targets = doctor_targets or {}
    police_targets = police_targets or {}
    for role, targets in ((Role.DOCTOR, doctor_targets), (Role.POLICE, police_targets)):
        for actor_id, target_id in targets.items():
            actor = alive_by_id.get(actor_id)
            if actor is None or actor.role != role:
                raise ValueError(f"{role.value} action requires an alive {role.value}")
            if target_id not in alive_by_id or (role == Role.POLICE and actor_id == target_id):
                raise ValueError(f"invalid {role.value} target: {target_id}")

    # All actions take effect from the same pre-elimination state.
    investigations = {
        police_id: alive_by_id[target_id].role == Role.MAFIA
        for police_id, target_id in police_targets.items()
    }
    candidates = [player.id for player in alive_by_id.values() if player.role != Role.MAFIA]
    if not candidates or not state.alive_by_role(Role.MAFIA):
        return NightResolution(killed=None, mafia_target=None, investigations=investigations)
    # Preserve the existing mafia no-consensus fallback; never invent doctor/police actions.
    selected_target = mafia_target if mafia_target in candidates else random.Random(seed + state.turn).choice(candidates)
    killed = None if selected_target in doctor_targets.values() else selected_target
    return NightResolution(killed=killed, mafia_target=selected_target, investigations=investigations)


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
