from __future__ import annotations

from dataclasses import dataclass
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


def run_single_match(
    config: AppConfig,
    *,
    seed: int | None = None,
    max_rounds: int = 10,
    progress_callback: ProgressCallback | None = None,
) -> MatchResult:
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
        p.id: LLMAgent(
            name=p.name,
            model_id=p.model_id,
            role=p.role,
            client=provider_client,
            fallback_models=[mid for mid in fallback_models if mid != p.model_id],
        )
        for p in players
    }

    events.append(
        GameEvent(
            turn=state.turn,
            phase=state.phase,
            speaker="system",
            kind="setup",
            content=f"Game started with {len(players)} players.",
        )
    )
    _emit_progress(
        progress_callback,
        {
            "kind": "setup",
            "turn": state.turn,
            "phase": state.phase.value,
            "message": f"Game started with {len(players)} players.",
            "players_status": _player_status_snapshot(state),
        },
    )
    latest_night_result = "Night result: no prior record."

    while state.turn <= max_rounds:
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
        if state.phase == Phase.END:
            break

        if state.phase == Phase.NIGHT:
            _append_night_phase_talk(
                state,
                agents,
                events,
                progress_callback=progress_callback,
            )
            killed, doctor_target, police_target = resolve_night(state, seed=match_seed)
            if killed is not None:
                state.replace_player(killed, alive=False)
                killed_name = _player_name(state, killed)
                content = f"Night result: {killed_name} was eliminated."
            else:
                content = "Night result: no one was eliminated."
            latest_night_result = content
            events.append(
                GameEvent(
                    turn=state.turn,
                    phase=state.phase,
                    speaker="system",
                    kind="night_result",
                    content=f"{content} doctor_target={doctor_target} police_target={police_target}",
                )
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
            winner_after_night = check_winner(state)
            if winner_after_night is not None:
                state.winner = winner_after_night
                state.phase = Phase.END
                events.append(
                    GameEvent(
                        turn=state.turn,
                        phase=Phase.END,
                        speaker="system",
                        kind="game_end",
                        content=f"Winner: {winner_after_night}",
                    )
                )
                _emit_progress(
                    progress_callback,
                    {
                        "kind": "game_end",
                        "turn": state.turn,
                        "phase": Phase.END.value,
                        "message": f"Winner: {winner_after_night}",
                        "players_status": _player_status_snapshot(state),
                    },
                )
                break

        if state.phase == Phase.DAY:
            _append_day_phase_talk(
                state,
                agents,
                events,
                night_result=latest_night_result,
                day_max_speeches_per_player=config.game.day_max_speeches_per_player,
                progress_callback=progress_callback,
            )

        if state.phase == Phase.VOTE:
            voted_out = resolve_vote(state, seed=match_seed)
            if voted_out is None:
                vote_content = "Vote result: tie, no elimination."
            else:
                state.replace_player(voted_out, alive=False)
                vote_content = f"Vote result: {_player_name(state, voted_out)} eliminated."
            events.append(
                GameEvent(
                    turn=state.turn,
                    phase=state.phase,
                    speaker="system",
                    kind="vote_result",
                    content=vote_content,
                )
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
            winner = check_winner(state)
            if winner is not None:
                state.winner = winner
                state.phase = Phase.END
                events.append(
                    GameEvent(
                        turn=state.turn,
                        phase=Phase.END,
                        speaker="system",
                        kind="game_end",
                        content=f"Winner: {winner}",
                    )
                )
                _emit_progress(
                    progress_callback,
                    {
                        "kind": "game_end",
                        "turn": state.turn,
                        "phase": Phase.END.value,
                        "message": f"Winner: {winner}",
                        "players_status": _player_status_snapshot(state),
                    },
                )
                break
            state.turn += 1

    if state.winner is None:
        state.winner = check_winner(state) or "draw"
        state.phase = Phase.END
        events.append(
            GameEvent(
                turn=state.turn,
                phase=Phase.END,
                speaker="system",
                kind="game_end",
                content=f"Winner: {state.winner}",
            )
        )
        _emit_progress(
            progress_callback,
            {
                "kind": "game_end",
                "turn": state.turn,
                "phase": Phase.END.value,
                "message": f"Winner: {state.winner}",
                "players_status": _player_status_snapshot(state),
            },
        )

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


def _append_night_phase_talk(
    state: GameState,
    agents: dict[int, LLMAgent],
    events: list[GameEvent],
    *,
    progress_callback: ProgressCallback | None = None,
) -> None:
    naming_instruction = _player_naming_instruction(state)
    for player in state.alive_by_role(Role.MAFIA):
        agent = agents[player.id]
        prompt = (
            f"Night turn {state.turn}. You are in private mafia chat. "
            "Coordinate with teammates discreetly. "
            "Include exactly one concrete observation and one next target with a short reason. "
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
        text = agent.speak(phase="night", turn=state.turn, prompt=prompt, history=history)
        text = _normalize_player_references(text, state)
        events.append(
            GameEvent(
                turn=state.turn,
                phase=state.phase,
                speaker=player.name,
                kind="mafia_chat",
                content=text,
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
                "message": text,
            },
        )


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
    alive_by_id = {player.id: player for player in alive_players}
    speeches_by_player: dict[int, int] = {player.id: 0 for player in alive_players}
    max_speeches_per_player = day_max_speeches_per_player
    naming_instruction = _player_naming_instruction(state)

    def _enqueue_requester(player_id: int) -> None:
        if speeches_by_player.get(player_id, 0) >= max_speeches_per_player:
            return
        if player_id in queue.items:
            return
        queue.enqueue(player_id)

    for player in alive_players:
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
        request_text = agent.speak(phase="day", turn=state.turn, prompt=request_prompt, history=history).strip()
        request_text = _normalize_player_references(request_text, state)
        requested, reason = _parse_speech_request(request_text)
        request_label = "REQUEST" if requested else "PASS"
        events.append(
            GameEvent(
                turn=state.turn,
                phase=Phase.DAY,
                speaker=player.name,
                kind="speak_request",
                content=request_label,
            )
        )
        events.append(
            GameEvent(
                turn=state.turn,
                phase=Phase.DAY,
                speaker=player.name,
                kind="speak_request_reason",
                content=reason,
            )
        )
        if requested:
            _enqueue_requester(player.id)

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
        return

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

    while not queue.is_empty():
        player_id = queue.dequeue()
        if player_id is None:
            break
        player = alive_by_id.get(player_id)
        if player is None:
            continue
        if speeches_by_player.get(player_id, 0) >= max_speeches_per_player:
            continue
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
        speech_prompt = (
            f"{night_result}\n"
            f"Your strategy: {strategy}\n"
            "Now give your public statement to all players. Include one specific evidence clue and one next suspicion target. "
            f"{naming_instruction}"
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

        for candidate in state.alive_players():
            if candidate.id == player_id:
                continue
            if speeches_by_player.get(candidate.id, 0) >= max_speeches_per_player:
                continue
            if candidate.id in queue.items:
                continue
            candidate_agent = agents[candidate.id]
            candidate_history = _visible_history_for_player(events, candidate)
            followup_prompt = candidate_agent.build_followup_request_prompt(
                night_result=night_result,
                speaker_name=player.name,
                speech=speech,
                naming_instruction=naming_instruction,
            )
            followup_text = candidate_agent.speak(
                phase="day",
                turn=state.turn,
                prompt=followup_prompt,
                history=candidate_history,
            ).strip()
            followup_text = _normalize_player_references(followup_text, state)
            followup_requested, followup_reason = _parse_speech_request(followup_text)
            followup_label = "REQUEST" if followup_requested else "PASS"
            events.append(
                GameEvent(
                    turn=state.turn,
                    phase=Phase.DAY,
                    speaker=candidate.name,
                    kind="speak_request",
                    content=followup_label,
                )
            )
            events.append(
                GameEvent(
                    turn=state.turn,
                    phase=Phase.DAY,
                    speaker=candidate.name,
                    kind="speak_request_reason",
                    content=followup_reason,
                )
            )
            if followup_requested:
                _enqueue_requester(candidate.id)

        _emit_progress(
            progress_callback,
            {
                "kind": "speech_queue",
                "turn": state.turn,
                "phase": Phase.DAY.value,
                "message": f"Follow-up requests processed after {player.name}.",
                "speech_queue": [_player_name(state, queued_player_id) for queued_player_id in queue.items],
            },
        )

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


def _player_name(state: GameState, player_id: int) -> str:
    for player in state.players:
        if player.id == player_id:
            return player.name
    return f"Unknown({player_id})"


def _player_naming_instruction(state: GameState) -> str:
    alive_names = ", ".join(player.name for player in state.alive_players())
    if not alive_names:
        alive_names = ", ".join(player.name for player in state.players)
    return (
        f"Use only these player names when referring to others: {alive_names}. "
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
    normalized = request_text.upper()
    reason = request_text.strip()
    if ":" in request_text:
        _, raw_reason = request_text.split(":", maxsplit=1)
        cleaned = raw_reason.strip()
        if cleaned:
            reason = cleaned
    if "PASS" in normalized:
        return False, reason or "no reason provided"
    if "REQUEST" in normalized:
        return True, reason or "wants to speak"
    requested = bool(request_text.strip())
    default_reason = reason or "implicit request"
    return requested, default_reason


def _visible_history_for_player(history: list[GameEvent], player: Player) -> list[GameEvent]:
    if player.role == Role.MAFIA:
        return history
    return [event for event in history if not (event.phase == Phase.NIGHT and event.kind == "mafia_chat")]


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
