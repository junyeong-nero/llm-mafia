from __future__ import annotations

from src.agents.llm_agent import LLMAgent, _history_to_context
from src.engine.game_state import GameEvent, Phase, Player, Role
from src.runner.single_match import _visible_history_for_player


def test_refresh_memory_tracks_own_dialogue_and_role_beliefs() -> None:
    agent = LLMAgent(name="Liam", model_id="m", role=Role.CITIZEN)
    alive_player_names = ["Liam", "Emma", "Ava"]
    visible_history = [
        GameEvent(turn=1, phase=Phase.DAY, speaker="Liam", kind="strategy", content="I suspect Ava."),
        GameEvent(turn=1, phase=Phase.DAY, speaker="Emma", kind="speech", content="Ava is suspicious."),
        GameEvent(turn=1, phase=Phase.DAY, speaker="Liam", kind="speech", content="Ava changed stories."),
    ]

    agent.refresh_memory(turn=2, visible_history=visible_history, alive_player_names=alive_player_names)

    assert agent.last_memory_turn == 2
    assert agent.own_dialogue_history == ["I suspect Ava.", "Ava changed stories."]
    assert "Ava" in agent.role_beliefs_by_name
    assert agent.role_beliefs_by_name["Ava"].mafia > agent.role_beliefs_by_name["Ava"].doctor


def test_refresh_memory_trims_history_and_evidence_lengths() -> None:
    agent = LLMAgent(name="Liam", model_id="m", role=Role.CITIZEN)
    alive_player_names = ["Liam", "Emma", "Ava"]

    visible_history: list[GameEvent] = []
    for index in range(25):
        visible_history.append(
            GameEvent(
                turn=1,
                phase=Phase.DAY,
                speaker="Liam",
                kind="speech",
                content=f"Statement {index}: I suspect Ava.",
            )
        )
        visible_history.append(
            GameEvent(
                turn=1,
                phase=Phase.DAY,
                speaker="Emma",
                kind="speech",
                content=f"Signal {index}: Ava is suspicious.",
            )
        )

    agent.refresh_memory(turn=3, visible_history=visible_history, alive_player_names=alive_player_names)

    assert len(agent.own_dialogue_history) == 20
    ava_belief = agent.role_beliefs_by_name["Ava"]
    assert len(ava_belief.evidence) <= 2


def test_build_belief_context_uses_compact_tag_format() -> None:
    agent = LLMAgent(name="Liam", model_id="m", role=Role.CITIZEN)
    alive_player_names = ["Liam", "Emma", "Ava"]
    visible_history = [
        GameEvent(turn=1, phase=Phase.DAY, speaker="Emma", kind="speech", content="Ava is suspicious."),
        GameEvent(
            turn=1,
            phase=Phase.VOTE,
            speaker="Emma",
            kind="day_vote",
            content='{"target_id": 3, "target_name": "Ava"}',
        ),
    ]

    agent.refresh_memory(turn=2, visible_history=visible_history, alive_player_names=alive_player_names)
    belief_context = agent.build_belief_context(alive_player_names=alive_player_names)

    assert "- Ava | M:" in belief_context
    assert "| E:" in belief_context
    assert "mafia=" not in belief_context


def test_night_inference_adds_night_kill_pattern_tag_for_mafia() -> None:
    agent = LLMAgent(name="Ethan", model_id="m", role=Role.MAFIA)
    alive_player_names = ["Ethan", "Noah", "Emma"]
    visible_history = [
        GameEvent(
            turn=2,
            phase=Phase.NIGHT,
            speaker="Ethan",
            kind="mafia_vote",
            content='{"target_id": 2, "target_name": "Noah"}',
        )
    ]

    agent.refresh_memory(
        turn=2,
        visible_history=visible_history,
        alive_player_names=alive_player_names,
        inference_mode="night",
    )

    noah_belief = agent.role_beliefs_by_name["Noah"]
    assert noah_belief.mafia > 0.34
    assert "night_kill_pattern" in noah_belief.evidence


def test_history_context_is_budgeted_and_truncated() -> None:
    long_text = "A" * 220
    history: list[GameEvent] = []
    for index in range(10):
        history.append(
            GameEvent(
                turn=index + 1,
                phase=Phase.DAY,
                speaker="Emma",
                kind="speech",
                content=f"day-{index} {long_text}",
            )
        )
    for index in range(5):
        history.append(
            GameEvent(
                turn=index + 1,
                phase=Phase.NIGHT,
                speaker="Ethan",
                kind="mafia_chat",
                content=f"night-{index} {long_text}",
            )
        )

    context = _history_to_context(history)

    assert "day-0" not in context
    assert "day-9" in context
    assert "night-0" not in context
    assert "night-4" in context
    assert "..." in context


def test_non_mafia_memory_does_not_include_private_night_events() -> None:
    citizen = Player(id=1, name="Emma", model_name="m", model_id="m", role=Role.CITIZEN, alive=True)
    agent = LLMAgent(name="Emma", model_id="m", role=Role.CITIZEN)
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
        GameEvent(turn=1, phase=Phase.DAY, speaker="Emma", kind="strategy", content="I trust Liam."),
        GameEvent(turn=1, phase=Phase.DAY, speaker="Liam", kind="speech", content="Noah is suspicious."),
    ]

    visible_history = _visible_history_for_player(events, citizen)
    alive_player_names = ["Emma", "Liam", "Noah"]
    agent.refresh_memory(turn=2, visible_history=visible_history, alive_player_names=alive_player_names)
    belief_context = agent.build_belief_context(alive_player_names=alive_player_names)

    assert "private mafia chat" not in belief_context
    assert "Mafia consensus target" not in belief_context
    assert "mafia_target=" not in belief_context
