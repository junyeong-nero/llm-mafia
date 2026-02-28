from __future__ import annotations

from src.engine.game_state import GameEvent, Phase, Player, Role
from src.runner.single_match import _visible_history_for_player


def test_visible_history_for_non_mafia_hides_private_night_events_and_targets() -> None:
    citizen = Player(id=1, name="Emma", model_name="m", model_id="m", role=Role.CITIZEN, alive=True)
    events = [
        GameEvent(turn=1, phase=Phase.NIGHT, speaker="Ethan", kind="mafia_chat", content="private mafia chat"),
        GameEvent(
            turn=1,
            phase=Phase.NIGHT,
            speaker="Ethan",
            kind="mafia_vote",
            content='{"target_id": 2, "target_name": "Noah"}',
        ),
        GameEvent(
            turn=1,
            phase=Phase.NIGHT,
            speaker="system",
            kind="mafia_consensus",
            content="Mafia consensus target: Noah.",
        ),
        GameEvent(
            turn=1,
            phase=Phase.NIGHT,
            speaker="system",
            kind="night_result",
            content="Night result: Alex was eliminated. mafia_target=1 doctor_target=2 police_target=3",
        ),
        GameEvent(turn=1, phase=Phase.DAY, speaker="Emma", kind="speech", content="I suspect Noah."),
    ]

    visible = _visible_history_for_player(events, citizen)

    visible_kinds = {event.kind for event in visible}
    assert "mafia_chat" not in visible_kinds
    assert "mafia_vote" not in visible_kinds
    assert "mafia_consensus" not in visible_kinds

    night_results = [event for event in visible if event.kind == "night_result"]
    assert len(night_results) == 1
    assert night_results[0].content == "Night result: Alex was eliminated."


def test_visible_history_for_mafia_keeps_private_events() -> None:
    mafia = Player(id=7, name="Ethan", model_name="m", model_id="m", role=Role.MAFIA, alive=True)
    events = [
        GameEvent(turn=1, phase=Phase.NIGHT, speaker="Ethan", kind="mafia_chat", content="private mafia chat"),
        GameEvent(
            turn=1,
            phase=Phase.NIGHT,
            speaker="system",
            kind="mafia_consensus",
            content="Mafia consensus target: Noah.",
        ),
        GameEvent(
            turn=1,
            phase=Phase.NIGHT,
            speaker="system",
            kind="night_result",
            content="Night result: Alex was eliminated. mafia_target=1 doctor_target=2 police_target=3",
        ),
    ]

    visible = _visible_history_for_player(events, mafia)

    assert visible == events
