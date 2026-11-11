from __future__ import annotations

from collections import Counter
import json
import random
import re

from src.agents.llm_agent import LLMAgent
from src.engine.game_state import GameEvent, GameState, Phase, Role
from src.engine.rules import resolve_night
from src.engine.visibility import visible_history_for_player
from src.runner.common import (
    ProgressCallback,
    _append_system_event,
    _emit_progress,
    _normalize_player_references,
    _player_name,
    _player_naming_instruction,
    _player_status_snapshot,
)


def _append_night_phase_talk(
    state: GameState,
    agents: dict[int, LLMAgent],
    events: list[GameEvent],
    *,
    progress_callback: ProgressCallback | None = None,
) -> None:
    alive_mafia_players = state.alive_by_role(Role.MAFIA)
    for player in alive_mafia_players:
        naming_instruction = _player_naming_instruction(state, speaker=player)
        teammate_names = [teammate.name for teammate in alive_mafia_players if teammate.id != player.id]
        if teammate_names:
            teammate_instruction = (
                f"Your confirmed mafia teammates are: {', '.join(teammate_names)}. "
                "Treat them as allies in this private chat."
            )
        else:
            teammate_instruction = "You are the only surviving mafia member right now."
        agent = agents[player.id]
        prompt = (
            f"Night turn {state.turn}. You are in private mafia chat. "
            f"You are {player.name}. "
            "Coordinate with teammates discreetly. "
            "Respond in exactly two lines. "
            "Line 1 format: CHAT: <your plan and its reason; admit when evidence is unavailable>. "
            "Line 2 format: VOTE_JSON: {\"target\": \"<exact player name>\"}. "
            "The target must be exactly one alive non-mafia player name. "
            "Do not invent past discussion or votes; the first night precedes all discussion. "
            f"{teammate_instruction} "
            f"{naming_instruction}"
        )
        _emit_progress(progress_callback, {
            "kind": "agent_thinking", "turn": state.turn, "phase": state.phase.value,
            "speaker": player.name, "message": f"{player.name} is thinking.",
        })
        history = visible_history_for_player(events, player)
        agent.refresh_memory(
            turn=state.turn, visible_history=history,
            alive_player_names=[alive.name for alive in state.alive_players()],
        )
        text = agent.speak(phase="night", turn=state.turn, prompt=prompt, history=history)
        text = _normalize_player_references(text, state)
        chat_text, voted_target_id, vote_error = _parse_mafia_vote_json(text, state)
        events.append(GameEvent(
            turn=state.turn, phase=state.phase, speaker=player.name, kind="mafia_chat", content=chat_text,
        ))
        if voted_target_id is not None:
            voted_target_name = _player_name(state, voted_target_id)
            events.append(GameEvent(
                turn=state.turn, phase=state.phase, speaker=player.name, kind="mafia_vote",
                content=json.dumps({"target_id": voted_target_id, "target_name": voted_target_name}),
            ))
            _emit_progress(progress_callback, {
                "kind": "mafia_vote", "turn": state.turn, "phase": state.phase.value,
                "speaker": player.name, "message": f"{player.name} voted to eliminate {voted_target_name}.",
                "target_name": voted_target_name,
            })
        else:
            events.append(GameEvent(
                turn=state.turn, phase=state.phase, speaker=player.name,
                kind="mafia_vote_invalid", content=vote_error or "invalid vote payload",
            ))
        _emit_progress(progress_callback, {
            "kind": "agent_spoke", "turn": state.turn, "phase": state.phase.value,
            "speaker": player.name, "message": f"{player.name} finished speaking.",
        })
        _emit_progress(progress_callback, {
            "kind": "mafia_chat", "turn": state.turn, "phase": state.phase.value,
            "speaker": player.name, "message": chat_text,
        })
        for teammate in alive_mafia_players:
            agents[teammate.id].refresh_memory(
                turn=state.turn, visible_history=visible_history_for_player(events, teammate),
                alive_player_names=[alive.name for alive in state.alive_players()],
            )


def _append_mafia_consensus_event(
    state: GameState,
    events: list[GameEvent],
    *,
    consensus_target: int | None,
    consensus_reason: str,
    progress_callback: ProgressCallback | None = None,
) -> str:
    if consensus_target is not None:
        content = f"Mafia consensus target: {_player_name(state, consensus_target)}."
    else:
        content = f"Mafia consensus not reached: {consensus_reason}."
    _append_system_event(events, turn=state.turn, phase=state.phase, kind="mafia_consensus", content=content)
    _emit_progress(progress_callback, {
        "kind": "mafia_consensus", "turn": state.turn, "phase": state.phase.value,
        "message": content, "players_status": _player_status_snapshot(state),
    })
    return content


def _collect_special_night_actions(
    state: GameState,
    agents: dict[int, LLMAgent],
    events: list[GameEvent],
    *,
    progress_callback: ProgressCallback | None,
) -> tuple[dict[int, int], dict[int, int]]:
    doctor_targets: dict[int, int] = {}
    police_targets: dict[int, int] = {}
    history_before_actions = list(events)
    for role, action, targets in (
        (Role.POLICE, "investigate", police_targets),
        (Role.DOCTOR, "protect", doctor_targets),
    ):
        for player in state.alive_by_role(role):
            candidates = [
                candidate for candidate in state.alive_players()
                if role == Role.DOCTOR or candidate.id != player.id
            ]
            if not candidates:
                continue
            prefix = action.upper()
            instruction = (
                "Choose one other living player to investigate. You will privately learn only whether they are mafia."
                if role == Role.POLICE else
                "Choose one living player to protect tonight. You may protect yourself, including on consecutive nights."
            )
            prompt = (
                f"Night turn {state.turn}. You are {player.name}. {instruction}\n"
                f"Valid targets: {', '.join(candidate.name for candidate in candidates)}.\n"
                "Use the provided records; if no evidence exists, make an exploratory choice without inventing evidence.\n"
                f"Reply in exactly one line: {prefix}: <exact target name>. No explanation."
            )
            _emit_progress(progress_callback, {
                "kind": "agent_thinking", "turn": state.turn, "phase": Phase.NIGHT.value,
                "speaker": player.name, "message": f"{player.name} is choosing a private night action.",
            })
            response = agents[player.id].speak(
                phase=action, turn=state.turn, prompt=prompt,
                history=visible_history_for_player(history_before_actions, player),
            ).strip()
            response_prefix, separator, name = response.partition(":")
            target_by_name = {candidate.name.casefold(): candidate.id for candidate in candidates}
            target_id = target_by_name.get(name.strip().casefold()) if separator and response_prefix.upper() == prefix else None
            if target_id is None:
                kind = "night_action_invalid"
                payload: dict[str, object] = {"action": action, "error": "expected an exact living target with the action prefix"}
            else:
                targets[player.id] = target_id
                kind = "night_action"
                payload = {"action": action, "target_id": target_id, "target_name": _player_name(state, target_id)}
            events.append(GameEvent(
                turn=state.turn, phase=Phase.NIGHT, speaker=player.name, kind=kind,
                content=json.dumps(payload, ensure_ascii=False), private_to=player.name,
            ))
            _emit_progress(progress_callback, {
                "kind": "agent_spoke", "turn": state.turn, "phase": Phase.NIGHT.value,
                "speaker": player.name, "message": f"{player.name} finished a private night action.",
            })
    return doctor_targets, police_targets


def _apply_night_resolution(
    state: GameState,
    events: list[GameEvent],
    *,
    agents: dict[int, LLMAgent],
    seed: int,
    mafia_target: int | None,
    progress_callback: ProgressCallback | None = None,
) -> str:
    doctor_targets, police_targets = _collect_special_night_actions(
        state, agents, events, progress_callback=progress_callback,
    )
    result = resolve_night(
        state, seed=seed, mafia_target=mafia_target,
        doctor_targets=doctor_targets, police_targets=police_targets,
    )
    for police_id, is_mafia in result.investigations.items():
        target_id = police_targets[police_id]
        events.append(GameEvent(
            turn=state.turn, phase=Phase.NIGHT, speaker="system", kind="investigation_result",
            content=json.dumps({
                "target_id": target_id, "target_name": _player_name(state, target_id), "is_mafia": is_mafia,
            }),
            private_to=_player_name(state, police_id),
        ))
    if result.killed is not None:
        state.replace_player(result.killed, alive=False)
        content = f"Night result: {_player_name(state, result.killed)} was eliminated."
    else:
        content = "Night result: no one was eliminated."
    _append_system_event(events, turn=state.turn, phase=state.phase, kind="night_result", content=content)
    _emit_progress(progress_callback, {
        "kind": "night_result", "turn": state.turn, "phase": state.phase.value,
        "message": content, "players_status": _player_status_snapshot(state),
    })
    return content


def _resolve_mafia_consensus_target(
    state: GameState,
    events: list[GameEvent],
    *,
    seed: int,
) -> tuple[int | None, str]:
    alive_mafia = state.alive_by_role(Role.MAFIA)
    if not alive_mafia:
        return None, "no alive mafia"
    alive_non_mafia_ids = {player.id for player in state.alive_players() if player.role != Role.MAFIA}
    if not alive_non_mafia_ids:
        return None, "no alive non-mafia targets"
    mafia_speakers = {player.name for player in alive_mafia}
    relevant_events = [
        event for event in events
        if event.turn == state.turn and event.phase == Phase.NIGHT
        and event.kind == "mafia_vote" and event.speaker in mafia_speakers
    ]
    if not relevant_events:
        return None, "no mafia_vote events"
    votes_by_speaker: dict[str, int] = {}
    for event in relevant_events:
        try:
            payload = json.loads(event.content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        target_id = payload.get("target_id")
        if isinstance(target_id, int) and target_id in alive_non_mafia_ids:
            votes_by_speaker[event.speaker] = target_id
    if not votes_by_speaker:
        return None, "no valid mafia_vote payload"
    tally = Counter(votes_by_speaker.values())
    top_vote_count = max(tally.values())
    top_target_ids = sorted([candidate_id for candidate_id, votes in tally.items() if votes == top_vote_count])
    if len(top_target_ids) > 1:
        selected_target_id = random.Random(seed + state.turn).choice(top_target_ids)
        return selected_target_id, f"tie random among {len(top_target_ids)} targets"
    if top_vote_count >= len(alive_mafia) // 2 + 1:
        return top_target_ids[0], f"majority {top_vote_count}/{len(alive_mafia)}"
    return None, "no majority consensus"


def _parse_mafia_vote_json(text: str, state: GameState) -> tuple[str, int | None, str | None]:
    alive_non_mafia = [player for player in state.alive_players() if player.role != Role.MAFIA]
    name_to_id = {player.name.lower(): player.id for player in alive_non_mafia}
    chat_text = text.strip()
    vote_payload_text: str | None = None
    for line in text.splitlines():
        if line.upper().startswith("VOTE_JSON:"):
            vote_payload_text = line.split(":", maxsplit=1)[1].strip()
        if line.upper().startswith("CHAT:"):
            candidate_chat = line.split(":", maxsplit=1)[1].strip()
            if candidate_chat:
                chat_text = candidate_chat
    if vote_payload_text is None:
        fence_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if fence_match:
            vote_payload_text = fence_match.group(1)
    if vote_payload_text is None:
        inline_match = re.search(r"\{\s*\"target\"\s*:\s*\"[^\"]+\"\s*\}", text)
        if inline_match:
            vote_payload_text = inline_match.group(0)
    if vote_payload_text is None:
        return chat_text, None, "missing VOTE_JSON payload"
    try:
        payload = json.loads(vote_payload_text)
    except json.JSONDecodeError:
        return chat_text, None, "invalid JSON payload"
    if not isinstance(payload, dict):
        return chat_text, None, "vote payload must be an object"
    raw_target = payload.get("target")
    if not isinstance(raw_target, str) or not raw_target.strip():
        return chat_text, None, "target must be a non-empty string"
    target_id = name_to_id.get(raw_target.strip().lower())
    if target_id is None:
        return chat_text, None, "target is not an alive non-mafia player"
    return chat_text, target_id, None
