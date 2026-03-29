from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from src.config import AppConfig
from src.engine.game_state import GameEvent, GameState, Phase
from src.runner.single_match import (
    MatchResult,
    MatchRuntime,
    ProgressCallback,
    _advance_match_phase,
    _append_mafia_consensus_event,
    _append_night_phase_talk,
    _build_match_state,
    _collect_day_followups,
    _collect_day_strategies_and_initial_requests,
    _collect_day_vote_ballots,
    _emit_initial_speech_queue_state,
    _ensure_match_winner,
    _finalize_match,
    _refresh_day_memory,
    _record_setup,
    _resolve_mafia_consensus_target,
    _resolve_vote_result,
    _run_next_day_speech,
    _set_winner_if_needed,
    _apply_night_resolution,
)
from src.runner.speech_queue import SpeechQueue


class MafiaGraphState(TypedDict):
    game_state: GameState
    events: list[GameEvent]
    latest_night_result: str
    speech_queue: list[int]
    speeches_by_player: dict[int, int]
    strategies: dict[int, str]
    ballots: dict[int, int]
    current_speaker_id: int | None
    current_speech: str | None
    queue_initialized: bool
    queue_finished: bool
    night_consensus_target: int | None
    night_consensus_reason: str


@dataclass(frozen=True)
class GraphRuntime:
    match_runtime: MatchRuntime


def run_graph_match(
    config: AppConfig,
    *,
    seed: int | None = None,
    max_rounds: int = 10,
    progress_callback: ProgressCallback | None = None,
) -> MatchResult:
    game_state, events, runtime = _build_match_state(
        config,
        seed=seed,
        max_rounds=max_rounds,
        progress_callback=progress_callback,
    )
    graph = _build_match_graph(GraphRuntime(match_runtime=runtime))
    final_state = graph.invoke(
        {
            "game_state": game_state,
            "events": events,
            "latest_night_result": runtime.latest_night_result,
            "speech_queue": [],
            "speeches_by_player": {},
            "strategies": {},
            "ballots": {},
            "current_speaker_id": None,
            "current_speech": None,
            "queue_initialized": False,
            "queue_finished": False,
            "night_consensus_target": None,
            "night_consensus_reason": "",
        }
    )
    runtime.latest_night_result = final_state["latest_night_result"]
    _ensure_match_winner(
        final_state["game_state"],
        final_state["events"],
        progress_callback=runtime.progress_callback,
    )
    return _finalize_match(final_state["game_state"], final_state["events"])


def _build_match_graph(graph_runtime: GraphRuntime):
    builder = StateGraph(MafiaGraphState)
    builder.add_node("setup_match", _setup_match_node(graph_runtime))
    builder.add_node("advance_phase", _advance_phase_node(graph_runtime))
    builder.add_node("night_private_chat", _night_private_chat_node(graph_runtime))
    builder.add_node("night_consensus", _night_consensus_node(graph_runtime))
    builder.add_node("night_resolution", _night_resolution_node(graph_runtime))
    builder.add_node("day_prepare_memory", _day_prepare_memory_node())
    builder.add_node("day_collect_initial_requests", _day_collect_initial_requests_node(graph_runtime))
    builder.add_node("day_run_next_speech", _day_run_next_speech_node(graph_runtime))
    builder.add_node("day_collect_followups", _day_collect_followups_node(graph_runtime))
    builder.add_node("vote_collect_ballots", _vote_collect_ballots_node(graph_runtime))
    builder.add_node("vote_resolve", _vote_resolve_node(graph_runtime))
    builder.add_node("check_winner", _check_winner_node(graph_runtime))
    builder.add_node("finalize_state", _finalize_state_node(graph_runtime))

    builder.add_edge(START, "setup_match")
    builder.add_edge("setup_match", "advance_phase")
    builder.add_conditional_edges("advance_phase", _route_after_advance_phase)
    builder.add_edge("night_private_chat", "night_consensus")
    builder.add_edge("night_consensus", "night_resolution")
    builder.add_edge("night_resolution", "check_winner")
    builder.add_edge("day_prepare_memory", "day_collect_initial_requests")
    builder.add_conditional_edges("day_collect_initial_requests", _route_after_day_collect_initial_requests)
    builder.add_conditional_edges("day_run_next_speech", _route_after_day_run_next_speech)
    builder.add_edge("day_collect_followups", "day_run_next_speech")
    builder.add_edge("vote_collect_ballots", "vote_resolve")
    builder.add_edge("vote_resolve", "check_winner")
    builder.add_conditional_edges("check_winner", _route_after_check_winner(graph_runtime))
    builder.add_edge("finalize_state", END)
    return builder.compile()


def _setup_match_node(graph_runtime: GraphRuntime):
    runtime = graph_runtime.match_runtime

    def setup_match(state: MafiaGraphState) -> MafiaGraphState:
        _record_setup(state["game_state"], state["events"], progress_callback=runtime.progress_callback)
        return _snapshot_state(state)

    return setup_match


def _advance_phase_node(graph_runtime: GraphRuntime):
    runtime = graph_runtime.match_runtime

    def advance_phase(state: MafiaGraphState) -> MafiaGraphState:
        _reset_phase_state(state)
        _advance_match_phase(state["game_state"], progress_callback=runtime.progress_callback)
        return _snapshot_state(state)

    return advance_phase


def _night_private_chat_node(graph_runtime: GraphRuntime):
    runtime = graph_runtime.match_runtime

    def night_private_chat(state: MafiaGraphState) -> MafiaGraphState:
        _append_night_phase_talk(
            state["game_state"],
            runtime.agents,
            state["events"],
            progress_callback=runtime.progress_callback,
        )
        return _snapshot_state(state)

    return night_private_chat


def _night_consensus_node(graph_runtime: GraphRuntime):
    runtime = graph_runtime.match_runtime

    def night_consensus(state: MafiaGraphState) -> MafiaGraphState:
        consensus_target, consensus_reason = _resolve_mafia_consensus_target(
            state["game_state"],
            state["events"],
            seed=runtime.match_seed,
        )
        state["night_consensus_target"] = consensus_target
        state["night_consensus_reason"] = consensus_reason
        _append_mafia_consensus_event(
            state["game_state"],
            state["events"],
            consensus_target=consensus_target,
            consensus_reason=consensus_reason,
            progress_callback=runtime.progress_callback,
        )
        return _snapshot_state(state)

    return night_consensus


def _night_resolution_node(graph_runtime: GraphRuntime):
    runtime = graph_runtime.match_runtime

    def night_resolution(state: MafiaGraphState) -> MafiaGraphState:
        state["latest_night_result"] = _apply_night_resolution(
            state["game_state"],
            state["events"],
            seed=runtime.match_seed,
            mafia_target=state["night_consensus_target"],
            progress_callback=runtime.progress_callback,
        )
        return _snapshot_state(state)

    return night_resolution


def _day_prepare_memory_node():
    def day_prepare_memory(state: MafiaGraphState) -> MafiaGraphState:
        game_state = state["game_state"]
        state["speeches_by_player"] = {player.id: 0 for player in game_state.alive_players()}
        state["strategies"] = {}
        state["speech_queue"] = []
        state["queue_initialized"] = False
        state["queue_finished"] = False
        state["current_speaker_id"] = None
        state["current_speech"] = None
        return _snapshot_state(state)

    return day_prepare_memory


def _day_collect_initial_requests_node(graph_runtime: GraphRuntime):
    runtime = graph_runtime.match_runtime

    def day_collect_initial_requests(state: MafiaGraphState) -> MafiaGraphState:
        queue = SpeechQueue(items=list(state["speech_queue"]))
        _refresh_day_memory(state["game_state"], runtime.agents, state["events"])
        _collect_day_strategies_and_initial_requests(
            state["game_state"],
            runtime.agents,
            state["events"],
            night_result=state["latest_night_result"],
            strategies=state["strategies"],
            queue=queue,
            speeches_by_player=state["speeches_by_player"],
            max_speeches_per_player=runtime.config.game.day_max_speeches_per_player,
            progress_callback=runtime.progress_callback,
        )
        _refresh_day_memory(state["game_state"], runtime.agents, state["events"])
        state["speech_queue"] = list(queue.items)
        state["queue_finished"] = _emit_initial_speech_queue_state(
            state["game_state"],
            state["events"],
            queue,
            progress_callback=runtime.progress_callback,
        )
        state["queue_initialized"] = True
        state["speech_queue"] = list(queue.items)
        return _snapshot_state(state)

    return day_collect_initial_requests


def _day_run_next_speech_node(graph_runtime: GraphRuntime):
    runtime = graph_runtime.match_runtime

    def day_run_next_speech(state: MafiaGraphState) -> MafiaGraphState:
        queue = SpeechQueue(items=list(state["speech_queue"]))
        player_id, speech = _run_next_day_speech(
            state["game_state"],
            runtime.agents,
            state["events"],
            queue=queue,
            strategies=state["strategies"],
            speeches_by_player=state["speeches_by_player"],
            max_speeches_per_player=runtime.config.game.day_max_speeches_per_player,
            night_result=state["latest_night_result"],
            progress_callback=runtime.progress_callback,
        )
        state["speech_queue"] = list(queue.items)
        state["current_speaker_id"] = player_id
        state["current_speech"] = speech
        return _snapshot_state(state)

    return day_run_next_speech


def _day_collect_followups_node(graph_runtime: GraphRuntime):
    runtime = graph_runtime.match_runtime

    def day_collect_followups(state: MafiaGraphState) -> MafiaGraphState:
        player_id = state["current_speaker_id"]
        speech = state["current_speech"]
        if player_id is None or speech is None:
            return _snapshot_state(state)
        queue = SpeechQueue(items=list(state["speech_queue"]))
        _collect_day_followups(
            state["game_state"],
            runtime.agents,
            state["events"],
            queue=queue,
            speeches_by_player=state["speeches_by_player"],
            max_speeches_per_player=runtime.config.game.day_max_speeches_per_player,
            current_speaker_id=player_id,
            speech=speech,
            night_result=state["latest_night_result"],
            progress_callback=runtime.progress_callback,
        )
        _refresh_day_memory(state["game_state"], runtime.agents, state["events"])
        state["speech_queue"] = list(queue.items)
        state["current_speaker_id"] = None
        state["current_speech"] = None
        if not state["speech_queue"]:
            state["queue_finished"] = True
            from src.runner.single_match import _emit_progress

            _emit_progress(
                runtime.progress_callback,
                {
                    "kind": "speech_queue",
                    "turn": state["game_state"].turn,
                    "phase": Phase.DAY.value,
                    "message": "All queued speeches finished.",
                    "speech_queue": [],
                },
            )
        return _snapshot_state(state)

    return day_collect_followups


def _vote_collect_ballots_node(graph_runtime: GraphRuntime):
    runtime = graph_runtime.match_runtime

    def vote_collect_ballots(state: MafiaGraphState) -> MafiaGraphState:
        state["ballots"] = _collect_day_vote_ballots(
            state["game_state"],
            runtime.agents,
            state["events"],
            progress_callback=runtime.progress_callback,
        )
        return _snapshot_state(state)

    return vote_collect_ballots


def _vote_resolve_node(graph_runtime: GraphRuntime):
    runtime = graph_runtime.match_runtime

    def vote_resolve(state: MafiaGraphState) -> MafiaGraphState:
        _resolve_vote_result(
            state["game_state"],
            state["events"],
            ballots=state["ballots"],
            progress_callback=runtime.progress_callback,
        )
        state["ballots"] = {}
        return _snapshot_state(state)

    return vote_resolve


def _check_winner_node(graph_runtime: GraphRuntime):
    runtime = graph_runtime.match_runtime

    def check_winner(state: MafiaGraphState) -> MafiaGraphState:
        winner = state["game_state"].winner
        if winner is None:
            from src.engine.rules import check_winner as resolve_winner

            winner = resolve_winner(state["game_state"])
        if not _set_winner_if_needed(
            state["game_state"],
            state["events"],
            winner=winner,
            progress_callback=runtime.progress_callback,
        ) and state["game_state"].phase == Phase.VOTE:
            state["game_state"].turn += 1
        return _snapshot_state(state)

    return check_winner


def _finalize_state_node(graph_runtime: GraphRuntime):
    runtime = graph_runtime.match_runtime

    def finalize_state(state: MafiaGraphState) -> MafiaGraphState:
        _ensure_match_winner(
            state["game_state"],
            state["events"],
            progress_callback=runtime.progress_callback,
        )
        return _snapshot_state(state)

    return finalize_state


def _route_after_advance_phase(state: MafiaGraphState) -> str:
    phase = state["game_state"].phase
    if phase == Phase.NIGHT:
        return "night_private_chat"
    if phase == Phase.DAY:
        return "day_prepare_memory"
    if phase == Phase.VOTE:
        return "vote_collect_ballots"
    return "finalize_state"


def _route_after_day_collect_initial_requests(state: MafiaGraphState) -> str:
    if state["queue_finished"]:
        return "advance_phase"
    return "day_run_next_speech"


def _route_after_day_run_next_speech(state: MafiaGraphState) -> str:
    if state["current_speaker_id"] is None or state["current_speech"] is None:
        if not state["speech_queue"]:
            return "advance_phase"
        return "day_run_next_speech"
    return "day_collect_followups"


def _route_after_check_winner(graph_runtime: GraphRuntime):
    runtime = graph_runtime.match_runtime

    def route_after_check_winner(state: MafiaGraphState) -> Literal["advance_phase", "finalize_state"]:
        game_state = state["game_state"]
        if game_state.winner is not None:
            return "finalize_state"
        if game_state.turn > runtime.max_rounds:
            return "finalize_state"
        return "advance_phase"

    return route_after_check_winner


def _reset_phase_state(state: MafiaGraphState) -> None:
    state["ballots"] = {}
    state["current_speaker_id"] = None
    state["current_speech"] = None
    state["queue_initialized"] = False
    state["queue_finished"] = False
    state["night_consensus_target"] = None
    state["night_consensus_reason"] = ""
    if state["game_state"].phase != Phase.DAY:
        state["speech_queue"] = []
        state["speeches_by_player"] = {}
        state["strategies"] = {}


def _snapshot_state(state: MafiaGraphState) -> MafiaGraphState:
    return {
        "game_state": state["game_state"],
        "events": state["events"],
        "latest_night_result": state["latest_night_result"],
        "speech_queue": list(state["speech_queue"]),
        "speeches_by_player": dict(state["speeches_by_player"]),
        "strategies": dict(state["strategies"]),
        "ballots": dict(state["ballots"]),
        "current_speaker_id": state["current_speaker_id"],
        "current_speech": state["current_speech"],
        "queue_initialized": state["queue_initialized"],
        "queue_finished": state["queue_finished"],
        "night_consensus_target": state["night_consensus_target"],
        "night_consensus_reason": state["night_consensus_reason"],
    }
