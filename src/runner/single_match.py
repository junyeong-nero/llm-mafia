from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from typing import Callable

from src.config import AppConfig

from src.agents.llm_agent import LLMAgent
from src.engine.game_state import GameEvent, GameState, Phase, Player, Role
from src.engine.phase import next_phase
from src.engine.rules import build_players, check_winner, resolve_night
from src.engine.vote import resolve_vote
from src.io.event_log import build_output_dir, state_to_dict, write_events_jsonl, write_summary_json
from src.metrics.collector import collect_metrics
from src.providers.openrouter_client import OpenRouterClient, OpenRouterError, load_openrouter_settings
from src.runner.speech_queue import SpeechQueue


@dataclass(frozen=True)
class MatchResult:
    state: GameState
    events: list[GameEvent]
    metrics: dict[str, object]
    output_dir: Path
    events_path: Path
    summary_path: Path


ProgressCallback = Callable[[dict[str, object]], None]


@dataclass
class MatchRuntime:
    config: AppConfig
    match_seed: int
    max_rounds: int
    progress_callback: ProgressCallback | None
    agents: dict[int, LLMAgent]
    latest_night_result: str = "Night result: no prior record."


def run_single_match(
    config: AppConfig,
    *,
    seed: int | None = None,
    max_rounds: int = 10,
    progress_callback: ProgressCallback | None = None,
) -> MatchResult:
    state, events, runtime = _build_match_state(config, seed=seed, max_rounds=max_rounds, progress_callback=progress_callback)
    _record_setup(state, events, progress_callback=runtime.progress_callback)

    while state.turn <= runtime.max_rounds:
        _advance_match_phase(state, progress_callback=runtime.progress_callback)
        if state.phase == Phase.END:
            break

        if state.phase == Phase.NIGHT and _handle_night_phase(state, runtime, events):
            break

        if state.phase == Phase.DAY:
            _handle_day_phase(state, runtime, events)

        if state.phase == Phase.VOTE and _handle_vote_phase(state, runtime, events):
            break

    _ensure_match_winner(state, events, progress_callback=runtime.progress_callback)
    return _finalize_match(state, events)


def _build_match_state(
    config: AppConfig,
    *,
    seed: int | None,
    max_rounds: int,
    progress_callback: ProgressCallback | None,
) -> tuple[GameState, list[GameEvent], MatchRuntime]:
    match_seed = seed if seed is not None else random.SystemRandom().randrange(0, 2**63)

    role_counts = {
        Role.MAFIA: config.game.roles.mafia,
        Role.POLICE: config.game.roles.police,
        Role.DOCTOR: config.game.roles.doctor,
        Role.CITIZEN: config.game.roles.citizen,
    }
    model_slots: list[tuple[str, str]] = []
    for model in config.llm.models:
        model_slots.extend([(model.name, model.model)] * model.count)

    players = build_players(role_counts, model_slots, seed=match_seed)
    state = GameState(turn=1, phase=Phase.SETUP, players=players)
    events: list[GameEvent] = []
    provider_client = _build_provider_client(config, progress_callback=progress_callback)
    fallback_models = [model.model for model in config.llm.models]
    agents = {
        player.id: LLMAgent(
            name=player.name,
            model_id=player.model_id,
            role=player.role,
            client=provider_client,
            fallback_models=[mid for mid in fallback_models if mid != player.model_id],
        )
        for player in players
    }
    runtime = MatchRuntime(
        config=config,
        match_seed=match_seed,
        max_rounds=max_rounds,
        progress_callback=progress_callback,
        agents=agents,
    )
    return state, events, runtime


def _record_setup(
    state: GameState,
    events: list[GameEvent],
    *,
    progress_callback: ProgressCallback | None,
) -> None:
    content = f"Game started with {len(state.players)} players."
    _append_system_event(events, turn=state.turn, phase=state.phase, kind="setup", content=content)
    _emit_progress(
        progress_callback,
        {
            "kind": "setup",
            "turn": state.turn,
            "phase": state.phase.value,
            "message": content,
            "players_status": _player_status_snapshot(state),
        },
    )


def _advance_match_phase(
    state: GameState,
    *,
    progress_callback: ProgressCallback | None,
) -> None:
    state.phase = next_phase(state.phase)
    _emit_progress(
        progress_callback,
        {
            "kind": "phase",
            "turn": state.turn,
            "phase": state.phase.value,
            "message": f"Entering {state.phase.value} phase on turn {state.turn}.",
            "players_status": _player_status_snapshot(state),
        },
    )


def _handle_night_phase(
    state: GameState,
    runtime: MatchRuntime,
    events: list[GameEvent],
) -> bool:
    _append_night_phase_talk(
        state,
        runtime.agents,
        events,
        progress_callback=runtime.progress_callback,
    )
    consensus_target, consensus_reason = _resolve_mafia_consensus_target(
        state,
        events,
        seed=runtime.match_seed,
    )
    _append_mafia_consensus_event(
        state,
        events,
        consensus_target=consensus_target,
        consensus_reason=consensus_reason,
        progress_callback=runtime.progress_callback,
    )
    runtime.latest_night_result = _apply_night_resolution(
        state,
        events,
        seed=runtime.match_seed,
        mafia_target=consensus_target,
        progress_callback=runtime.progress_callback,
    )
    winner_after_night = check_winner(state)
    return _set_winner_if_needed(
        state,
        events,
        winner=winner_after_night,
        progress_callback=runtime.progress_callback,
    )


def _handle_day_phase(
    state: GameState,
    runtime: MatchRuntime,
    events: list[GameEvent],
) -> None:
    _append_day_phase_talk(
        state,
        runtime.agents,
        events,
        night_result=runtime.latest_night_result,
        day_max_speeches_per_player=runtime.config.game.day_max_speeches_per_player,
        progress_callback=runtime.progress_callback,
    )


def _handle_vote_phase(
    state: GameState,
    runtime: MatchRuntime,
    events: list[GameEvent],
) -> bool:
    ballots = _collect_day_vote_ballots(
        state,
        runtime.agents,
        events,
        progress_callback=runtime.progress_callback,
    )
    _resolve_vote_result(
        state,
        events,
        ballots=ballots,
        progress_callback=runtime.progress_callback,
    )
    winner = check_winner(state)
    if _set_winner_if_needed(state, events, winner=winner, progress_callback=runtime.progress_callback):
        return True
    state.turn += 1
    return False


def _set_winner_if_needed(
    state: GameState,
    events: list[GameEvent],
    *,
    winner: str | None,
    progress_callback: ProgressCallback | None,
) -> bool:
    if winner is None:
        return False
    state.winner = winner
    state.phase = Phase.END
    content = f"Winner: {winner}"
    _append_system_event(events, turn=state.turn, phase=Phase.END, kind="game_end", content=content)
    _emit_progress(
        progress_callback,
        {
            "kind": "game_end",
            "turn": state.turn,
            "phase": Phase.END.value,
            "message": content,
            "players_status": _player_status_snapshot(state),
        },
    )
    return True


def _ensure_match_winner(
    state: GameState,
    events: list[GameEvent],
    *,
    progress_callback: ProgressCallback | None,
) -> None:
    if state.winner is not None:
        return
    fallback_winner = check_winner(state) or "draw"
    _set_winner_if_needed(
        state,
        events,
        winner=fallback_winner,
        progress_callback=progress_callback,
    )


def _finalize_match(state: GameState, events: list[GameEvent]) -> MatchResult:
    state.events = events
    metrics = collect_metrics(state, events)
    output_dir = build_output_dir()
    events_path = write_events_jsonl(events, output_dir)
    summary = {
        "state": state_to_dict(state),
        "metrics": metrics,
    }
    summary_path = write_summary_json(summary, output_dir)
    return MatchResult(
        state=state,
        events=events,
        metrics=metrics,
        output_dir=output_dir,
        events_path=events_path,
        summary_path=summary_path,
    )


def _append_system_event(
    events: list[GameEvent],
    *,
    turn: int,
    phase: Phase,
    kind: str,
    content: str,
) -> None:
    events.append(GameEvent(turn=turn, phase=phase, speaker="system", kind=kind, content=content))


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
            "Line 1 format: CHAT: <one concrete observation and one short reason>. "
            "Line 2 format: VOTE_JSON: {\"target\": \"<exact player name>\"}. "
            "The target must be exactly one alive non-mafia player name. "
            f"{teammate_instruction} "
            f"{naming_instruction}"
        )
        _emit_progress(
            progress_callback,
            {
                "kind": "agent_thinking",
                "turn": state.turn,
                "phase": state.phase.value,
                "speaker": player.name,
                "message": f"{player.name} is thinking.",
            },
        )
        history = _visible_history_for_player(events, player)
        agent.refresh_memory(
            turn=state.turn,
            visible_history=history,
            alive_player_names=[alive.name for alive in state.alive_players()],
            inference_mode="night",
        )
        text = agent.speak(phase="night", turn=state.turn, prompt=prompt, history=history)
        text = _normalize_player_references(text, state)
        chat_text, voted_target_id, vote_error = _parse_mafia_vote_json(text, state)
        events.append(
            GameEvent(
                turn=state.turn,
                phase=state.phase,
                speaker=player.name,
                kind="mafia_chat",
                content=chat_text,
            )
        )
        if voted_target_id is not None:
            voted_target_name = _player_name(state, voted_target_id)
            vote_payload = json.dumps({"target_id": voted_target_id, "target_name": voted_target_name})
            events.append(
                GameEvent(
                    turn=state.turn,
                    phase=state.phase,
                    speaker=player.name,
                    kind="mafia_vote",
                    content=vote_payload,
                )
            )
            _emit_progress(
                progress_callback,
                {
                    "kind": "mafia_vote",
                    "turn": state.turn,
                    "phase": state.phase.value,
                    "speaker": player.name,
                    "message": f"{player.name} voted to eliminate {voted_target_name}.",
                    "target_name": voted_target_name,
                },
            )
        else:
            events.append(
                GameEvent(
                    turn=state.turn,
                    phase=state.phase,
                    speaker=player.name,
                    kind="mafia_vote_invalid",
                    content=vote_error or "invalid vote payload",
                )
            )
        _emit_progress(
            progress_callback,
            {
                "kind": "agent_spoke",
                "turn": state.turn,
                "phase": state.phase.value,
                "speaker": player.name,
                "message": f"{player.name} finished speaking.",
            },
        )
        _emit_progress(
            progress_callback,
            {
                "kind": "mafia_chat",
                "turn": state.turn,
                "phase": state.phase.value,
                "speaker": player.name,
                "message": chat_text,
            },
        )
        _refresh_memory_for_players(
            state=state,
            agents=agents,
            events=events,
            players=alive_mafia_players,
            inference_mode="night",
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
        consensus_content = f"Mafia consensus target: {_player_name(state, consensus_target)}."
    else:
        consensus_content = f"Mafia consensus not reached: {consensus_reason}."
    _append_system_event(
        events,
        turn=state.turn,
        phase=state.phase,
        kind="mafia_consensus",
        content=consensus_content,
    )
    _emit_progress(
        progress_callback,
        {
            "kind": "mafia_consensus",
            "turn": state.turn,
            "phase": state.phase.value,
            "message": consensus_content,
            "players_status": _player_status_snapshot(state),
        },
    )
    return consensus_content


def _apply_night_resolution(
    state: GameState,
    events: list[GameEvent],
    *,
    seed: int,
    mafia_target: int | None,
    progress_callback: ProgressCallback | None = None,
) -> str:
    killed, _mafia_target, _doctor_target, _police_target = resolve_night(
        state,
        seed=seed,
        mafia_target=mafia_target,
    )
    if killed is not None:
        state.replace_player(killed, alive=False)
        content = f"Night result: {_player_name(state, killed)} was eliminated."
    else:
        content = "Night result: no one was eliminated."
    _append_system_event(
        events,
        turn=state.turn,
        phase=state.phase,
        kind="night_result",
        content=content,
    )
    _emit_progress(
        progress_callback,
        {
            "kind": "night_result",
            "turn": state.turn,
            "phase": state.phase.value,
            "message": content,
            "players_status": _player_status_snapshot(state),
        },
    )
    return content


def _append_day_phase_talk(
    state: GameState,
    agents: dict[int, LLMAgent],
    events: list[GameEvent],
    *,
    night_result: str,
    day_max_speeches_per_player: int,
    progress_callback: ProgressCallback | None = None,
) -> None:
    queue = SpeechQueue()
    strategies: dict[int, str] = {}
    alive_players = state.alive_players()
    speeches_by_player: dict[int, int] = {player.id: 0 for player in alive_players}
    max_speeches_per_player = day_max_speeches_per_player
    _refresh_day_memory(state, agents, events)
    _collect_day_strategies_and_initial_requests(
        state,
        agents,
        events,
        night_result=night_result,
        strategies=strategies,
        queue=queue,
        speeches_by_player=speeches_by_player,
        max_speeches_per_player=max_speeches_per_player,
        progress_callback=progress_callback,
    )
    _refresh_memory_for_players(
        state=state,
        agents=agents,
        events=events,
        players=alive_players,
            inference_mode="day",
    )
    if _emit_initial_speech_queue_state(state, events, queue, progress_callback=progress_callback):
        return
    while not queue.is_empty():
        player_id, speech = _run_next_day_speech(
            state,
            agents,
            events,
            queue=queue,
            strategies=strategies,
            speeches_by_player=speeches_by_player,
            max_speeches_per_player=max_speeches_per_player,
            night_result=night_result,
            progress_callback=progress_callback,
        )
        if player_id is None or speech is None:
            continue
        _collect_day_followups(
            state,
            agents,
            events,
            queue=queue,
            speeches_by_player=speeches_by_player,
            max_speeches_per_player=max_speeches_per_player,
            current_speaker_id=player_id,
            speech=speech,
            night_result=night_result,
            progress_callback=progress_callback,
        )
        _refresh_day_memory(state, agents, events)
    _emit_progress(
        progress_callback,
        {
            "kind": "speech_queue",
            "turn": state.turn,
            "phase": Phase.DAY.value,
            "message": "All queued speeches finished.",
            "speech_queue": [],
        },
    )


def _refresh_day_memory(
    state: GameState,
    agents: dict[int, LLMAgent],
    events: list[GameEvent],
) -> None:
    _refresh_memory_for_players(
        state=state,
        agents=agents,
        events=events,
        players=state.alive_players(),
        inference_mode="day",
    )


def _enqueue_speaker_request(
    queue: SpeechQueue,
    *,
    player_id: int,
    speeches_by_player: dict[int, int],
    max_speeches_per_player: int,
) -> bool:
    if speeches_by_player.get(player_id, 0) >= max_speeches_per_player:
        return False
    if player_id in queue.items:
        return False
    queue.enqueue(player_id)
    return True


def _collect_day_strategies_and_initial_requests(
    state: GameState,
    agents: dict[int, LLMAgent],
    events: list[GameEvent],
    *,
    night_result: str,
    strategies: dict[int, str],
    queue: SpeechQueue,
    speeches_by_player: dict[int, int],
    max_speeches_per_player: int,
    progress_callback: ProgressCallback | None = None,
) -> None:
    for player in state.alive_players():
        naming_instruction = _player_naming_instruction(state, speaker=player)
        agent = agents[player.id]
        history = _visible_history_for_player(events, player)
        strategy_prompt = (
            f"{night_result}\n"
            "You survived this night. In one short sentence, state your survival strategy with one evidence clue and one suspicion target."
            f" {naming_instruction}"
        )
        _emit_progress(
            progress_callback,
            {
                "kind": "agent_thinking",
                "turn": state.turn,
                "phase": Phase.DAY.value,
                "speaker": player.name,
                "message": f"{player.name} is preparing strategy.",
            },
        )
        strategy = agent.speak(phase="day", turn=state.turn, prompt=strategy_prompt, history=history).strip()
        strategy = _normalize_player_references(strategy, state)
        strategies[player.id] = strategy
        events.append(
            GameEvent(
                turn=state.turn,
                phase=Phase.DAY,
                speaker=player.name,
                kind="strategy",
                content=strategy,
            )
        )
        request_prompt = agent.build_speak_request_prompt(
            night_result=night_result,
            strategy=strategy,
            naming_instruction=naming_instruction,
        )
        request_history = _visible_history_for_player(events, player)
        request_text = agent.speak(
            phase="day",
            turn=state.turn,
            prompt=request_prompt,
            history=request_history,
        ).strip()
        request_text = _normalize_player_references(request_text, state)
        requested, reason = _parse_speech_request(request_text)
        _append_speech_request_events(
            events,
            turn=state.turn,
            speaker_name=player.name,
            requested=requested,
            reason=reason,
        )
        if requested:
            _enqueue_speaker_request(
                queue,
                player_id=player.id,
                speeches_by_player=speeches_by_player,
                max_speeches_per_player=max_speeches_per_player,
            )


def _append_speech_request_events(
    events: list[GameEvent],
    *,
    turn: int,
    speaker_name: str,
    requested: bool,
    reason: str,
) -> None:
    request_label = "REQUEST" if requested else "PASS"
    events.append(
        GameEvent(
            turn=turn,
            phase=Phase.DAY,
            speaker=speaker_name,
            kind="speak_request",
            content=request_label,
        )
    )
    events.append(
        GameEvent(
            turn=turn,
            phase=Phase.DAY,
            speaker=speaker_name,
            kind="speak_request_reason",
            content=reason,
        )
    )


def _emit_initial_speech_queue_state(
    state: GameState,
    events: list[GameEvent],
    queue: SpeechQueue,
    *,
    progress_callback: ProgressCallback | None = None,
) -> bool:
    if queue.is_empty():
        events.append(
            GameEvent(
                turn=state.turn,
                phase=Phase.DAY,
                speaker="system",
                kind="speech_queue",
                content="No players requested speaking.",
            )
        )
        _emit_progress(
            progress_callback,
            {
                "kind": "speech_queue",
                "turn": state.turn,
                "phase": Phase.DAY.value,
                "message": "No players requested speaking.",
                "speech_queue": [],
            },
        )
        return True
    queue_names = [_player_name(state, player_id) for player_id in queue.items]
    events.append(
        GameEvent(
            turn=state.turn,
            phase=Phase.DAY,
            speaker="system",
            kind="speech_queue",
            content=f"Speech queue initialized with {len(queue)} players.",
        )
    )
    _emit_progress(
        progress_callback,
        {
            "kind": "speech_queue",
            "turn": state.turn,
            "phase": Phase.DAY.value,
            "message": f"Speech queue initialized with {len(queue_names)} players.",
            "speech_queue": queue_names,
        },
    )
    return False


def _run_next_day_speech(
    state: GameState,
    agents: dict[int, LLMAgent],
    events: list[GameEvent],
    *,
    queue: SpeechQueue,
    strategies: dict[int, str],
    speeches_by_player: dict[int, int],
    max_speeches_per_player: int,
    night_result: str,
    progress_callback: ProgressCallback | None = None,
) -> tuple[int | None, str | None]:
    player_id = queue.dequeue()
    if player_id is None:
        return None, None
    alive_by_id = {player.id: player for player in state.alive_players()}
    player = alive_by_id.get(player_id)
    if player is None:
        return None, None
    if speeches_by_player.get(player_id, 0) >= max_speeches_per_player:
        return None, None
    remaining_queue = [_player_name(state, queued_player_id) for queued_player_id in queue.items]
    _emit_progress(
        progress_callback,
        {
            "kind": "speech_queue",
            "turn": state.turn,
            "phase": Phase.DAY.value,
            "speaker": player.name,
            "message": f"Now speaking: {player.name}",
            "speech_queue": remaining_queue,
        },
    )
    agent = agents[player_id]
    strategy = strategies.get(player_id, "")
    history = _visible_history_for_player(events, player)
    speech_prompt = _build_day_speech_prompt(
        player=player,
        speech_number=speeches_by_player.get(player_id, 0) + 1,
        max_speeches_per_player=max_speeches_per_player,
        night_result=night_result,
        strategy=strategy,
        self_speech_context=_build_self_speech_context(events, player),
        naming_instruction=_player_naming_instruction(state, speaker=player),
    )
    _emit_progress(
        progress_callback,
        {
            "kind": "agent_thinking",
            "turn": state.turn,
            "phase": Phase.DAY.value,
            "speaker": player.name,
            "message": f"{player.name} is preparing public statement.",
        },
    )
    speech = agent.speak(phase="day", turn=state.turn, prompt=speech_prompt, history=history)
    speech = _normalize_player_references(speech, state)
    events.append(
        GameEvent(
            turn=state.turn,
            phase=Phase.DAY,
            speaker=player.name,
            kind="speech",
            content=speech,
        )
    )
    _emit_progress(
        progress_callback,
        {
            "kind": "agent_spoke",
            "turn": state.turn,
            "phase": Phase.DAY.value,
            "speaker": player.name,
            "message": f"{player.name} posted public statement.",
        },
    )
    _emit_progress(
        progress_callback,
        {
            "kind": "speech",
            "turn": state.turn,
            "phase": Phase.DAY.value,
            "speaker": player.name,
            "message": speech,
        },
    )
    speeches_by_player[player_id] = speeches_by_player.get(player_id, 0) + 1
    return player_id, speech


def _collect_day_followups(
    state: GameState,
    agents: dict[int, LLMAgent],
    events: list[GameEvent],
    *,
    queue: SpeechQueue,
    speeches_by_player: dict[int, int],
    max_speeches_per_player: int,
    current_speaker_id: int,
    speech: str,
    night_result: str,
    progress_callback: ProgressCallback | None = None,
) -> None:
    current_speaker_name = _player_name(state, current_speaker_id)
    for candidate in state.alive_players():
        if candidate.id == current_speaker_id:
            continue
        if speeches_by_player.get(candidate.id, 0) >= max_speeches_per_player:
            continue
        if candidate.id in queue.items:
            continue
        candidate_agent = agents[candidate.id]
        candidate_history = _visible_history_for_player(events, candidate)
        followup_prompt = candidate_agent.build_followup_request_prompt(
            night_result=night_result,
            speaker_name=current_speaker_name,
            speech=speech,
            naming_instruction=_player_naming_instruction(state, speaker=candidate),
        )
        followup_text = candidate_agent.speak(
            phase="day",
            turn=state.turn,
            prompt=followup_prompt,
            history=candidate_history,
        ).strip()
        followup_text = _normalize_player_references(followup_text, state)
        followup_requested, followup_reason = _parse_speech_request(followup_text)
        _append_speech_request_events(
            events,
            turn=state.turn,
            speaker_name=candidate.name,
            requested=followup_requested,
            reason=followup_reason,
        )
        if followup_requested:
            _enqueue_speaker_request(
                queue,
                player_id=candidate.id,
                speeches_by_player=speeches_by_player,
                max_speeches_per_player=max_speeches_per_player,
            )
    _emit_progress(
        progress_callback,
        {
            "kind": "speech_queue",
            "turn": state.turn,
            "phase": Phase.DAY.value,
            "message": f"Follow-up requests processed after {current_speaker_name}.",
            "speech_queue": [_player_name(state, queued_player_id) for queued_player_id in queue.items],
        },
    )


def _resolve_vote_result(
    state: GameState,
    events: list[GameEvent],
    *,
    ballots: dict[int, int],
    progress_callback: ProgressCallback | None = None,
) -> str:
    voted_out = resolve_vote(state, ballots=ballots)
    if voted_out is None:
        vote_content = "Vote result: tie, no elimination."
    else:
        state.replace_player(voted_out, alive=False)
        vote_content = f"Vote result: {_player_name(state, voted_out)} eliminated."
    _append_system_event(
        events,
        turn=state.turn,
        phase=state.phase,
        kind="vote_result",
        content=vote_content,
    )
    _emit_progress(
        progress_callback,
        {
            "kind": "vote_result",
            "turn": state.turn,
            "phase": state.phase.value,
            "message": vote_content,
            "players_status": _player_status_snapshot(state),
        },
    )
    return vote_content


def _player_name(state: GameState, player_id: int) -> str:
    for player in state.players:
        if player.id == player_id:
            return player.name
    return f"Unknown({player_id})"


def _player_naming_instruction(state: GameState, *, speaker: Player) -> str:
    alive_others = [player.name for player in state.alive_players() if player.id != speaker.id]
    if not alive_others:
        alive_others = [player.name for player in state.players if player.id != speaker.id]
    if alive_others:
        others_instruction = (
            f"Use only these player names when referring to others: {', '.join(alive_others)}. "
        )
    else:
        others_instruction = "There are no other players to name right now. "
    return (
        f"You are {speaker.name}. Use first person (I/me/my) for yourself. "
        f"Do not refer to yourself as {speaker.name}. "
        f"{others_instruction}"
        "Do not use numeric labels such as 'player 1', 'agent 2', or 'P3'."
    )


def _normalize_player_references(text: str, state: GameState) -> str:
    id_to_name = {player.id: player.name for player in state.players}

    def repl(match: re.Match[str]) -> str:
        player_id = int(match.group(1))
        return id_to_name.get(player_id, match.group(0))

    normalized = re.sub(r"\b(?:player|agent|p)\s*#?(\d+)\b", repl, text, flags=re.IGNORECASE)
    return normalized


def _player_status_snapshot(state: GameState) -> list[dict[str, object]]:
    return [
        {
            "name": player.name,
            "model_name": player.model_name,
            "role": player.role.value,
            "alive": player.alive,
        }
        for player in state.players
    ]


def _build_provider_client(
    config: AppConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> OpenRouterClient | None:
    if config.llm.provider != "openrouter":
        return None
    try:
        settings = load_openrouter_settings()
    except OpenRouterError:
        return None
    return OpenRouterClient(settings, on_retry=lambda event: _emit_provider_retry(progress_callback, event))


def _parse_speech_request(request_text: str) -> tuple[bool, str]:
    stripped = request_text.strip()
    normalized = stripped.upper()

    if normalized.startswith("REQUEST"):
        reason = stripped.split(":", maxsplit=1)[1].strip() if ":" in stripped else ""
        return True, reason or "wants to speak"

    if normalized.startswith("PASS"):
        reason = stripped.split(":", maxsplit=1)[1].strip() if ":" in stripped else ""
        return False, reason or "no reason provided"

    reason = stripped
    if "PASS" in normalized:
        return False, reason or "no reason provided"
    if "REQUEST" in normalized:
        return True, reason or "wants to speak"
    requested = bool(request_text.strip())
    default_reason = reason or "implicit request"
    return requested, default_reason


def _collect_day_vote_ballots(
    state: GameState,
    agents: dict[int, LLMAgent],
    events: list[GameEvent],
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[int, int]:
    ballots: dict[int, int] = {}
    alive_player_names = [player.name for player in state.alive_players()]
    for voter in state.alive_players():
        naming_instruction = _player_naming_instruction(state, speaker=voter)
        voter_agent = agents[voter.id]
        visible_history = _visible_history_for_player(events, voter)
        voter_agent.refresh_memory(
            turn=state.turn,
            visible_history=visible_history,
            alive_player_names=alive_player_names,
            inference_mode="day",
        )
        self_speech_context = voter_agent.build_own_dialogue_context()
        belief_context = voter_agent.build_belief_context(alive_player_names=alive_player_names)
        vote_prompt = voter_agent.build_day_vote_prompt(
            self_speech_context=self_speech_context,
            belief_context=belief_context,
            naming_instruction=naming_instruction,
        )
        _emit_progress(
            progress_callback,
            {
                "kind": "agent_thinking",
                "turn": state.turn,
                "phase": Phase.VOTE.value,
                "speaker": voter.name,
                "message": f"{voter.name} is deciding day vote.",
            },
        )
        vote_text = voter_agent.speak(phase="vote", turn=state.turn, prompt=vote_prompt, history=None).strip()
        vote_text = _normalize_player_references(vote_text, state)
        voted_target_id, vote_error = _parse_day_vote(vote_text, state=state, voter=voter)
        if voted_target_id is None:
            events.append(
                GameEvent(
                    turn=state.turn,
                    phase=Phase.VOTE,
                    speaker=voter.name,
                    kind="day_vote_invalid",
                    content=vote_error or "invalid day vote response",
                )
            )
            continue
        voted_target_name = _player_name(state, voted_target_id)
        ballots[voter.id] = voted_target_id
        vote_payload = json.dumps({"target_id": voted_target_id, "target_name": voted_target_name})
        events.append(
            GameEvent(
                turn=state.turn,
                phase=Phase.VOTE,
                speaker=voter.name,
                kind="day_vote",
                content=vote_payload,
            )
        )
        _emit_progress(
            progress_callback,
            {
                "kind": "day_vote",
                "turn": state.turn,
                "phase": Phase.VOTE.value,
                "speaker": voter.name,
                "message": f"{voter.name} voted to eliminate {voted_target_name}.",
                "target_name": voted_target_name,
            },
        )
    return ballots


def _build_self_speech_context(events: list[GameEvent], voter: Player) -> str:
    own_messages = [
        event.content.strip()
        for event in events
        if event.phase == Phase.DAY and event.speaker == voter.name and event.kind in {"strategy", "speech"}
    ]
    if not own_messages:
        return "- (no prior public statements)"
    lines: list[str] = []
    for idx, message in enumerate(own_messages, start=1):
        lines.append(f"- #{idx}: {message}")
    return "\n".join(lines)


def _build_day_speech_prompt(
    *,
    player: Player,
    speech_number: int,
    max_speeches_per_player: int,
    night_result: str,
    strategy: str,
    self_speech_context: str,
    naming_instruction: str,
) -> str:
    return (
        f"{night_result}\n"
        f"You are {player.name}. This is your speech #{speech_number} out of {max_speeches_per_player} allowed speeches today.\n"
        f"Your strategy: {strategy}\n"
        "Your own public statements so far:\n"
        f"{self_speech_context}\n"
        "Speak in first person (I/me/my). Do not refer to yourself by your own name. "
        "Give one public statement with one concrete clue and one next suspicion target. "
        "Do not repeat prior wording; add one new point. "
        f"{naming_instruction}"
    )


def _parse_day_vote(vote_text: str, *, state: GameState, voter: Player) -> tuple[int | None, str | None]:
    alive_non_self = [candidate for candidate in state.alive_players() if candidate.id != voter.id]
    if not alive_non_self:
        return None, "no alive target candidates"
    name_to_id = {candidate.name.lower(): candidate.id for candidate in alive_non_self}

    normalized = vote_text.strip()
    if not normalized:
        return None, "empty vote response"

    if ":" in normalized:
        prefix, remainder = normalized.split(":", maxsplit=1)
        if prefix.strip().upper() == "VOTE":
            normalized = remainder.strip()

    target_id = name_to_id.get(normalized.lower())
    if target_id is not None:
        return target_id, None

    for candidate in alive_non_self:
        if re.search(rf"\b{re.escape(candidate.name)}\b", normalized, flags=re.IGNORECASE):
            return candidate.id, None
    return None, "target is not an alive non-self player"


def _resolve_mafia_consensus_target(
    state: GameState,
    events: list[GameEvent],
    *,
    seed: int,
) -> tuple[int | None, str]:
    alive_mafia = state.alive_by_role(Role.MAFIA)
    if not alive_mafia:
        return None, "no alive mafia"

    alive_non_mafia = [player for player in state.alive_players() if player.role != Role.MAFIA]
    if not alive_non_mafia:
        return None, "no alive non-mafia targets"
    alive_non_mafia_ids = {player.id for player in alive_non_mafia}

    mafia_speakers = {player.name for player in alive_mafia}
    relevant_events = [
        event
        for event in events
        if event.turn == state.turn
        and event.phase == Phase.NIGHT
        and event.kind == "mafia_vote"
        and event.speaker in mafia_speakers
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
        rng = random.Random(seed + state.turn)
        selected_target_id = rng.choice(top_target_ids)
        return selected_target_id, f"tie random among {len(top_target_ids)} targets"

    majority_threshold = len(alive_mafia) // 2 + 1
    top_target_id = top_target_ids[0]
    if top_vote_count >= majority_threshold:
        return top_target_id, f"majority {top_vote_count}/{len(alive_mafia)}"

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


def _visible_history_for_player(history: list[GameEvent], player: Player) -> list[GameEvent]:
    if player.role == Role.MAFIA:
        return history

    visible_history: list[GameEvent] = []
    for event in history:
        if event.kind == "mafia_consensus":
            continue
        if event.phase == Phase.NIGHT and event.kind in {"mafia_chat", "mafia_vote", "mafia_vote_invalid"}:
            continue

        if event.kind == "night_result" and "mafia_target=" in event.content:
            public_content = event.content.split("mafia_target=", maxsplit=1)[0].strip()
            visible_history.append(
                GameEvent(
                    turn=event.turn,
                    phase=event.phase,
                    speaker=event.speaker,
                    kind=event.kind,
                    content=public_content,
                )
            )
            continue

        visible_history.append(event)

    return visible_history


def _refresh_memory_for_players(
    *,
    state: GameState,
    agents: dict[int, LLMAgent],
    events: list[GameEvent],
    players: list[Player],
    inference_mode: str,
) -> None:
    alive_player_names = [alive.name for alive in state.alive_players()]
    for player in players:
        if not player.alive:
            continue
        history = _visible_history_for_player(events, player)
        agents[player.id].refresh_memory(
            turn=state.turn,
            visible_history=history,
            alive_player_names=alive_player_names,
            inference_mode=inference_mode,
        )


def _emit_provider_retry(
    progress_callback: ProgressCallback | None,
    retry_event: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "kind": "provider_retry",
        "message": "Provider call is retrying.",
    }
    payload.update(retry_event)
    _emit_progress(progress_callback, payload)


def _emit_progress(
    progress_callback: ProgressCallback | None,
    payload: dict[str, object],
) -> None:
    if progress_callback is None:
        return
    progress_callback(payload)
