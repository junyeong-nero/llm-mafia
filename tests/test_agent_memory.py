from __future__ import annotations

from src.agents.llm_agent import LLMAgent, _history_to_context
from src.engine.game_state import GameEvent, Phase, Player, Role
from src.runner.single_match import _visible_history_for_player


def test_refresh_memory_tracks_only_own_public_dialogue_not_private_plans() -> None:
    agent = LLMAgent(name="Liam", model_id="m", role=Role.CITIZEN)
    history = [
        GameEvent(turn=1, phase=Phase.DAY, speaker="Liam", kind="strategy", content="I suspect Ava.", private_to="Liam"),
        GameEvent(turn=1, phase=Phase.DAY, speaker="Emma", kind="speech", content="Ava is suspicious."),
        GameEvent(turn=1, phase=Phase.DAY, speaker="Liam", kind="speech", content="Ava changed stories."),
    ]
    agent.refresh_memory(turn=2, visible_history=history, alive_player_names=["Liam", "Emma", "Ava"])
    assert agent.last_memory_turn == 2
    assert agent.own_dialogue_history == ["Ava changed stories."]
    assert agent.known_mafia_by_name == {}


def test_own_dialogue_summary_is_bounded_without_dropping_original_history() -> None:
    agent = LLMAgent(name="Liam", model_id="m", role=Role.CITIZEN)
    history = [GameEvent(turn=1, phase=Phase.DAY, speaker="Liam", kind="speech", content=f"Statement {index}") for index in range(25)]
    agent.refresh_memory(turn=3, visible_history=history, alive_player_names=["Liam", "Emma", "Ava"])
    assert len(agent.own_dialogue_history) == 20
    assert agent.own_dialogue_history[0] == "Statement 5"
    assert "Statement 0" in _history_to_context(history)


def test_empty_memory_and_target_contexts_are_explicit() -> None:
    agent = LLMAgent(name="Liam", model_id="m", role=Role.CITIZEN)
    assert agent.build_own_dialogue_context() == "- (no prior public statements)"
    assert agent.build_role_knowledge_context(alive_player_names=["Liam"]) == "- (no alive targets to evaluate)"


def test_history_context_preserves_complete_original_records() -> None:
    long_text = "A" * 220
    history = [
        GameEvent(turn=index + 1, phase=Phase.DAY, speaker="Emma", kind="speech", content=f"day-{index} {long_text}")
        for index in range(10)
    ]
    history.extend(
        GameEvent(turn=index + 1, phase=Phase.NIGHT, speaker="Ethan", kind="mafia_chat", content=f"night-{index} {long_text}")
        for index in range(5)
    )
    context = _history_to_context(history)
    assert "day-0" in context
    assert "day-9" in context
    assert "night-0" in context
    assert "night-4" in context
    assert long_text in context
    assert "미검증 주장" in context
    assert "Role-inference tags" not in context


def test_non_mafia_context_does_not_include_private_night_events_and_targets() -> None:
    citizen = Player(id=1, name="Emma", model_name="m", model_id="m", role=Role.CITIZEN)
    events = [
        GameEvent(turn=1, phase=Phase.NIGHT, speaker="Ethan", kind="mafia_chat", content="private mafia chat"),
        GameEvent(turn=1, phase=Phase.NIGHT, speaker="Ethan", kind="mafia_vote", content='{"target_name": "Noah"}'),
        GameEvent(turn=1, phase=Phase.NIGHT, speaker="system", kind="mafia_consensus", content="Mafia consensus target: Noah."),
        GameEvent(turn=1, phase=Phase.NIGHT, speaker="system", kind="night_result", content="Night result: Alex was eliminated. mafia_target=1 doctor_target=2 police_target=3"),
        GameEvent(turn=1, phase=Phase.DAY, speaker="Liam", kind="speech", content="Noah is suspicious."),
    ]
    context = _history_to_context(_visible_history_for_player(events, citizen))
    assert "private mafia chat" not in context
    assert "Mafia consensus target" not in context
    assert "mafia_target=" not in context
    assert "Noah is suspicious." in context
