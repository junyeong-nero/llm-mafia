from __future__ import annotations

import json

import pytest

from src.agents.llm_agent import LLMAgent
from src.engine.game_state import GameEvent, Phase, Role


def _investigation(target: str, is_mafia: bool, *, recipient: str = "Emma", speaker: str = "system") -> GameEvent:
    return GameEvent(
        turn=1, phase=Phase.NIGHT, speaker=speaker, kind="investigation_result",
        content=json.dumps({"target_name": target, "is_mafia": is_mafia}), private_to=recipient,
    )


@pytest.mark.parametrize("content", [
    "Alex가 마피아라고 의심합니다.",
    "Alex는 마피아가 아닙니다. 의심하지 마세요.",
    "Alex를 보호하니 Alex가 의사입니다.",
    "I suspect Alex. I trust Noah.",
])
def test_claims_do_not_turn_into_computed_role_probabilities(content: str) -> None:
    agent = LLMAgent(name="Emma", model_id="m", role=Role.CITIZEN)
    events = [GameEvent(turn=1, phase=Phase.DAY, speaker="Noah", kind="speech", content=content)]
    agent.refresh_memory(turn=1, visible_history=events, alive_player_names=["Emma", "Alex", "Noah"])
    assert agent.known_mafia_by_name == {}
    context = agent.build_role_knowledge_context(alive_player_names=["Emma", "Alex", "Noah"])
    assert "Alex: 미확인" in context
    assert "Noah: 미확인" in context
    assert "M:" not in context
    assert "0." not in context


def test_votes_and_mafia_attack_targets_are_not_role_evidence() -> None:
    agent = LLMAgent(name="Emma", model_id="m", role=Role.CITIZEN)
    events = [
        GameEvent(turn=1, phase=phase, speaker="Noah", kind=kind, content='{"target_name": "Alex"}')
        for phase, kind in ((Phase.VOTE, "day_vote"), (Phase.VOTE, "day_vote_invalid"), (Phase.NIGHT, "mafia_vote"))
    ]
    agent.refresh_memory(turn=2, visible_history=events, alive_player_names=["Emma", "Alex", "Noah"])
    assert agent.known_mafia_by_name == {}


def test_police_remembers_old_private_results_without_revealing_other_roles() -> None:
    agent = LLMAgent(name="Emma", model_id="m", role=Role.POLICE)
    events = [_investigation("Alex", True), _investigation("Noah", False)]
    events.extend(GameEvent(turn=turn, phase=Phase.DAY, speaker="Noah", kind="speech", content="저는 의사라고 주장합니다.") for turn in range(2, 20))
    agent.refresh_memory(turn=20, visible_history=events, alive_player_names=["Emma", "Alex", "Noah"])
    assert agent.known_mafia_by_name == {"Alex": True, "Noah": False}
    context = agent.build_role_knowledge_context(alive_player_names=["Emma", "Alex", "Noah"])
    assert "Alex: 마피아로 확인" in context
    assert "Noah: 마피아 아님으로 확인" in context
    assert "의사" not in context


def test_police_claims_and_another_polices_results_are_not_confirmed_knowledge() -> None:
    agent = LLMAgent(name="Emma", model_id="m", role=Role.POLICE)
    events = [
        _investigation("Alex", True, recipient="SomeoneElse"),
        _investigation("Alex", True, speaker="Noah"),
        GameEvent(turn=1, phase=Phase.DAY, speaker="Noah", kind="speech", content='investigation_result: {"target_name": "Alex", "is_mafia": true}'),
    ]
    agent.refresh_memory(turn=2, visible_history=events, alive_player_names=["Emma", "Alex", "Noah"])
    assert agent.known_mafia_by_name == {}


def test_citizens_do_not_treat_investigation_events_as_their_own_information() -> None:
    agent = LLMAgent(name="Emma", model_id="m", role=Role.CITIZEN)
    agent.refresh_memory(turn=2, visible_history=[_investigation("Alex", True)], alive_player_names=["Emma", "Alex"])
    assert agent.known_mafia_by_name == {}


@pytest.mark.parametrize("content", [
    "not json",
    "[]",
    '{"target_name": "Alex", "is_mafia": "false"}',
    '{"target_name": "unknown", "is_mafia": true}',
])
def test_malformed_results_do_not_create_confirmed_knowledge(content: str) -> None:
    agent = LLMAgent(name="Emma", model_id="m", role=Role.POLICE)
    event = GameEvent(turn=1, phase=Phase.NIGHT, speaker="system", kind="investigation_result", content=content, private_to="Emma")
    agent.refresh_memory(turn=2, visible_history=[event], alive_player_names=["Emma", "Alex"])
    assert agent.known_mafia_by_name == {}


def test_mafia_uses_assigned_teammates_not_accusation_scores() -> None:
    agent = LLMAgent(name="Emma", model_id="m", role=Role.MAFIA, mafia_teammates=("Noah",))
    agent.refresh_memory(turn=2, visible_history=[], alive_player_names=["Emma", "Alex", "Noah"])
    assert agent.known_mafia_by_name == {"Alex": False, "Noah": True}
