from __future__ import annotations

from src.agents.llm_agent import LLMAgent
from src.engine.game_state import GameEvent, GameState, Phase, Player, Role
from src.engine.vote import resolve_vote
from src.runner.single_match import _build_self_speech_context, _parse_day_vote


def _build_state() -> GameState:
    players = [
        Player(id=1, name="Liam", model_name="m", model_id="m", role=Role.CITIZEN, alive=True),
        Player(id=2, name="Emma", model_name="m", model_id="m", role=Role.CITIZEN, alive=True),
        Player(id=3, name="Ava", model_name="m", model_id="m", role=Role.MAFIA, alive=True),
    ]
    return GameState(turn=2, phase=Phase.VOTE, players=players)


def test_build_self_speech_context_includes_only_own_day_statements() -> None:
    voter = Player(id=1, name="Liam", model_name="m", model_id="m", role=Role.CITIZEN, alive=True)
    events = [
        GameEvent(turn=1, phase=Phase.DAY, speaker="Liam", kind="strategy", content="I suspect Ava."),
        GameEvent(turn=1, phase=Phase.DAY, speaker="Liam", kind="speech", content="Ava changed stories."),
        GameEvent(turn=1, phase=Phase.DAY, speaker="Emma", kind="speech", content="I trust Ava."),
        GameEvent(turn=1, phase=Phase.NIGHT, speaker="Liam", kind="mafia_chat", content="hidden"),
    ]

    context = _build_self_speech_context(events, voter)

    assert "I suspect Ava." in context
    assert "Ava changed stories." in context
    assert "I trust Ava." not in context
    assert "hidden" not in context


def test_parse_day_vote_accepts_vote_prefix_and_name() -> None:
    state = _build_state()
    voter = state.players[0]

    target_id, error = _parse_day_vote("VOTE: Emma", state=state, voter=voter)

    assert target_id == 2
    assert error is None


def test_parse_day_vote_rejects_self_vote() -> None:
    state = _build_state()
    voter = state.players[0]

    target_id, error = _parse_day_vote("VOTE: Liam", state=state, voter=voter)

    assert target_id is None
    assert error == "target is not an alive non-self player"


def test_build_day_vote_prompt_includes_role_belief_context() -> None:
    agent = LLMAgent(name="Liam", model_id="m", role=Role.CITIZEN)
    alive_player_names = ["Liam", "Emma", "Ava"]
    visible_history = [
        GameEvent(turn=1, phase=Phase.DAY, speaker="Liam", kind="strategy", content="I suspect Ava."),
        GameEvent(turn=1, phase=Phase.DAY, speaker="Liam", kind="speech", content="Ava changed stories."),
        GameEvent(turn=1, phase=Phase.DAY, speaker="Emma", kind="speech", content="Ava is suspicious."),
    ]
    agent.refresh_memory(turn=2, visible_history=visible_history, alive_player_names=alive_player_names)

    prompt = agent.build_day_vote_prompt(
        self_speech_context=agent.build_own_dialogue_context(),
        belief_context=agent.build_belief_context(alive_player_names=alive_player_names),
        naming_instruction="Use only these player names when referring to others: Liam, Emma, Ava.",
    )

    assert "Role inferences (visible only):" in prompt
    assert "Ava | M:" in prompt
    assert "VOTE: <exact alive player name>" in prompt
    assert "You are Liam. Use first person (I/me/my). Do not refer to yourself as Liam." in prompt
    assert "{belief_context}" not in prompt


def test_resolve_vote_returns_top_candidate_when_unique() -> None:
    state = _build_state()
    ballots = {
        1: 2,
        2: 3,
        3: 2,
    }

    voted_out = resolve_vote(state, ballots=ballots)

    assert voted_out == 2


def test_resolve_vote_ignores_invalid_ballots_and_handles_draw() -> None:
    state = _build_state()
    ballots = {
        1: 1,
        2: 999,
    }

    voted_out = resolve_vote(state, ballots=ballots)

    assert voted_out is None
