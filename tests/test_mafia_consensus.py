from __future__ import annotations

from src.engine.game_state import GameEvent, GameState, Phase, Player, Role
from src.engine.rules import resolve_night
from src.runner.night_phase import _parse_mafia_vote_json, _resolve_mafia_consensus_target


def _build_state() -> GameState:
    players = [
        Player(id=1, name="Liam", model_name="m", model_id="m", role=Role.MAFIA, alive=True),
        Player(id=2, name="Ethan", model_name="m", model_id="m", role=Role.MAFIA, alive=True),
        Player(id=3, name="Emma", model_name="m", model_id="m", role=Role.CITIZEN, alive=True),
        Player(id=4, name="Ava", model_name="m", model_id="m", role=Role.CITIZEN, alive=True),
    ]
    return GameState(turn=1, phase=Phase.NIGHT, players=players)


def test_parse_mafia_vote_json_extracts_chat_and_target() -> None:
    state = _build_state()
    text = (
        "CHAT: A를 눈여겨 보고 있지만 오늘은 C를 정리하자.\n"
        'VOTE_JSON: {"target": "Emma"}'
    )

    chat_text, target_id, error = _parse_mafia_vote_json(text, state)

    assert chat_text == "A를 눈여겨 보고 있지만 오늘은 C를 정리하자."
    assert target_id == 3
    assert error is None


def test_resolve_mafia_consensus_returns_majority_target() -> None:
    state = _build_state()
    events = [
        GameEvent(
            turn=1,
            phase=Phase.NIGHT,
            speaker="Liam",
            kind="mafia_vote",
            content='{"target_id": 3, "target_name": "Emma"}',
        ),
        GameEvent(
            turn=1,
            phase=Phase.NIGHT,
            speaker="Ethan",
            kind="mafia_vote",
            content='{"target_id": 3, "target_name": "Emma"}',
        ),
    ]

    target, reason = _resolve_mafia_consensus_target(state, events, seed=11)

    assert target == 3
    assert reason.startswith("majority")


def test_resolve_mafia_consensus_breaks_tie_randomly_between_top_targets() -> None:
    state = _build_state()
    events = [
        GameEvent(
            turn=1,
            phase=Phase.NIGHT,
            speaker="Liam",
            kind="mafia_vote",
            content='{"target_id": 3, "target_name": "Emma"}',
        ),
        GameEvent(
            turn=1,
            phase=Phase.NIGHT,
            speaker="Ethan",
            kind="mafia_vote",
            content='{"target_id": 4, "target_name": "Ava"}',
        ),
    ]

    target, reason = _resolve_mafia_consensus_target(state, events, seed=11)

    assert target in {3, 4}
    assert reason.startswith("tie random")


def test_parse_mafia_vote_json_rejects_non_alive_target() -> None:
    state = _build_state()
    state.replace_player(3, alive=False)

    chat_text, target_id, error = _parse_mafia_vote_json('VOTE_JSON: {"target": "Emma"}', state)

    assert chat_text == 'VOTE_JSON: {"target": "Emma"}'
    assert target_id is None
    assert error == "target is not an alive non-mafia player"


def test_resolve_night_applies_consensus_target_when_provided() -> None:
    state = _build_state()

    result = resolve_night(state, seed=7, mafia_target=3)

    assert result.mafia_target == 3
    assert result.killed == 3
    assert result.investigations == {}


def test_resolve_night_falls_back_when_no_consensus_target() -> None:
    state = _build_state()

    result = resolve_night(state, seed=7, mafia_target=None)

    assert result.mafia_target in {3, 4}
    assert result.killed == result.mafia_target
    assert result.investigations == {}


def test_night_vote_tie_randomly_kills_one_of_tied_targets() -> None:
    state = _build_state()
    events = [
        GameEvent(
            turn=1,
            phase=Phase.NIGHT,
            speaker="Liam",
            kind="mafia_vote",
            content='{"target_id": 3, "target_name": "Emma"}',
        ),
        GameEvent(
            turn=1,
            phase=Phase.NIGHT,
            speaker="Ethan",
            kind="mafia_vote",
            content='{"target_id": 4, "target_name": "Ava"}',
        ),
    ]

    selected_target, reason = _resolve_mafia_consensus_target(state, events, seed=11)
    result = resolve_night(state, seed=11, mafia_target=selected_target)

    assert selected_target in {3, 4}
    assert reason.startswith("tie random")
    assert result.mafia_target == selected_target
    assert result.killed == selected_target
    assert result.investigations == {}
