from __future__ import annotations

import json
from typing import Any

import pytest

from src.engine.game_state import GameEvent, GameState, Phase, Player, Role
from src.engine.rules import resolve_night
from src.runner.single_match import _apply_night_resolution, _visible_history_for_player


def _state() -> GameState:
    roles = [Role.MAFIA, Role.POLICE, Role.DOCTOR, Role.CITIZEN, Role.POLICE, Role.DOCTOR]
    names = ["Alex", "Emma", "Noah", "Olivia", "Liam", "Ava"]
    return GameState(turn=1, phase=Phase.NIGHT, players=[
        Player(id=index, name=name, model_name="test", model_id="test", role=role)
        for index, (name, role) in enumerate(zip(names, roles, strict=True), start=1)
    ])


class ActionAgent:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def speak(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return self.reply


def _agents() -> dict[int, ActionAgent]:
    return {
        2: ActionAgent("INVESTIGATE: Alex"),
        3: ActionAgent("PROTECT: Olivia"),
        5: ActionAgent("INVESTIGATE: Noah"),
        6: ActionAgent("PROTECT: Ava"),
    }


def test_explicit_protection_blocks_attack_and_investigations_return_only_mafia_status() -> None:
    state = _state()
    result = resolve_night(
        state, seed=42, mafia_target=4,
        doctor_targets={3: 4, 6: 6}, police_targets={2: 1, 5: 3},
    )
    assert result.killed is None
    assert result.mafia_target == 4
    assert result.investigations == {2: True, 5: False}
    assert all(player.alive for player in state.players)


def test_omitted_special_actions_are_not_randomly_generated() -> None:
    result = resolve_night(_state(), seed=42, mafia_target=4)
    assert result.killed == 4
    assert result.investigations == {}


def test_doctor_can_protect_self() -> None:
    result = resolve_night(_state(), seed=42, mafia_target=3, doctor_targets={3: 3})
    assert result.killed is None


@pytest.mark.parametrize("doctor_targets,police_targets", [
    ({4: 3}, {}),  # citizen cannot protect
    ({3: 999}, {}),
    ({}, {4: 1}),  # citizen cannot investigate
    ({}, {2: 2}),  # investigating oneself adds no information
    ({}, {2: 999}),
])
def test_engine_rejects_invalid_special_actions(doctor_targets: dict[int, int], police_targets: dict[int, int]) -> None:
    with pytest.raises(ValueError):
        resolve_night(_state(), seed=42, mafia_target=4, doctor_targets=doctor_targets, police_targets=police_targets)


def test_dead_role_cannot_act_and_dead_player_cannot_be_targeted() -> None:
    state = _state()
    state.replace_player(3, alive=False)
    with pytest.raises(ValueError):
        resolve_night(state, seed=42, doctor_targets={3: 4})
    with pytest.raises(ValueError):
        resolve_night(state, seed=42, police_targets={2: 3})


def test_runner_requests_actions_and_delivers_results_only_to_each_investigator() -> None:
    state = _state()
    agents = _agents()
    events = [GameEvent(turn=1, phase=Phase.NIGHT, speaker="Alex", kind="mafia_chat", content="secret attack plan")]
    progress: list[dict[str, object]] = []
    content = _apply_night_resolution(
        state, events, agents=agents, seed=42, mafia_target=4, progress_callback=progress.append,
    )
    assert content == "Night result: no one was eliminated."
    assert agents[2].calls[0]["phase"] == "investigate"
    assert agents[3].calls[0]["phase"] == "protect"
    for agent in agents.values():
        assert len(agent.calls) == 1
        assert not any(event.kind == "mafia_chat" for event in agent.calls[0]["history"])
        assert not any(event.kind == "investigation_result" for event in agent.calls[0]["history"])

    results = [event for event in events if event.kind == "investigation_result"]
    assert len(results) == 2
    assert {event.private_to for event in results} == {"Emma", "Liam"}
    for player in state.players:
        visible = _visible_history_for_player(events, player)
        own_results = [event for event in visible if event.kind == "investigation_result"]
        if player.role == Role.POLICE:
            assert len(own_results) == 1
            assert own_results[0].private_to == player.name
            payload = json.loads(own_results[0].content)
            assert payload["is_mafia"] is (player.name == "Emma")
            assert "role" not in payload  # non-mafia does not reveal doctor/citizen/police
        else:
            assert own_results == []
    assert not any("investigation_result" in str(payload) or "is_mafia" in str(payload) for payload in progress)


def test_attacked_police_acts_before_death_and_result_stays_private() -> None:
    state = _state()
    agents = _agents()
    events: list[GameEvent] = []
    _apply_night_resolution(state, events, agents=agents, seed=42, mafia_target=2)
    assert not state.players[1].alive
    assert len(agents[2].calls) == 1
    result = next(event for event in events if event.kind == "investigation_result" and event.private_to == "Emma")
    assert json.loads(result.content)["is_mafia"] is True
    assert result not in _visible_history_for_player(events, state.players[0])


def test_dead_special_roles_are_not_queried() -> None:
    state = _state()
    state.replace_player(2, alive=False)
    state.replace_player(3, alive=False)
    agents = {5: ActionAgent("INVESTIGATE: Alex"), 6: ActionAgent("PROTECT: Olivia")}
    events: list[GameEvent] = []
    _apply_night_resolution(state, events, agents=agents, seed=42, mafia_target=4)
    assert all(len(agent.calls) == 1 for agent in agents.values())
    assert not any(event.private_to in {"Emma", "Noah"} for event in events)
    assert state.players[3].alive


def test_next_night_receives_only_own_previous_investigation_results() -> None:
    state = _state()
    agents = _agents()
    events: list[GameEvent] = []
    _apply_night_resolution(state, events, agents=agents, seed=42, mafia_target=4)
    state.turn = 2
    _apply_night_resolution(state, events, agents=agents, seed=42, mafia_target=4)
    for actor_id, agent in agents.items():
        history = agent.calls[1]["history"]
        results = [event for event in history if event.kind == "investigation_result"]
        expected = 1 if actor_id in {2, 5} else 0
        assert len(results) == expected
        assert all(event.turn == 1 and event.private_to == state.players[actor_id - 1].name for event in results)


def test_invalid_night_response_is_private_and_does_not_trigger_random_protection() -> None:
    state = _state()
    agents = {key: ActionAgent("nonsense") for key in _agents()}
    events: list[GameEvent] = []
    _apply_night_resolution(state, events, agents=agents, seed=42, mafia_target=4)
    assert not state.players[3].alive
    assert len([event for event in events if event.kind == "night_action_invalid"]) == 4
    assert not any(event.kind == "investigation_result" for event in events)
    assert all(event.private_to for event in events if event.kind == "night_action_invalid")
