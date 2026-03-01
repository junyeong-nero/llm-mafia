from __future__ import annotations

from src.agents.llm_agent import LLMAgent
from src.engine.game_state import GameEvent, Phase, Role


def _build_visible_history() -> list[GameEvent]:
    return [
        GameEvent(
            turn=1,
            phase=Phase.DAY,
            speaker="Emma",
            kind="speech",
            content="I suspect Noah.",
        ),
        GameEvent(
            turn=1,
            phase=Phase.NIGHT,
            speaker="Ethan",
            kind="mafia_vote",
            content='{"target_id": 2, "target_name": "Noah"}',
        ),
    ]


def test_inference_mode_day_ignores_night_signals() -> None:
    agent = LLMAgent(name="Ethan", model_id="m", role=Role.MAFIA)
    alive_player_names = ["Ethan", "Noah", "Emma"]

    agent.refresh_memory(
        turn=2,
        visible_history=_build_visible_history(),
        alive_player_names=alive_player_names,
        inference_mode="day",
    )

    noah_belief = agent.role_beliefs_by_name["Noah"]
    assert "sus" in noah_belief.evidence
    assert "night_kill_pattern" not in noah_belief.evidence


def test_inference_mode_combined_uses_day_and_night_signals() -> None:
    agent = LLMAgent(name="Ethan", model_id="m", role=Role.MAFIA)
    alive_player_names = ["Ethan", "Noah", "Emma"]

    agent.refresh_memory(
        turn=2,
        visible_history=_build_visible_history(),
        alive_player_names=alive_player_names,
        inference_mode="combined",
    )

    noah_belief = agent.role_beliefs_by_name["Noah"]
    assert "sus" in noah_belief.evidence
    assert "night_kill_pattern" in noah_belief.evidence


def test_repeated_day_signal_is_deduped_by_speaker_and_tag() -> None:
    agent = LLMAgent(name="Liam", model_id="m", role=Role.CITIZEN)
    alive_player_names = ["Liam", "Noah", "Emma"]
    visible_history = [
        GameEvent(
            turn=1,
            phase=Phase.DAY,
            speaker="Emma",
            kind="speech",
            content="I suspect Noah.",
        ),
        GameEvent(
            turn=1,
            phase=Phase.DAY,
            speaker="Emma",
            kind="speech",
            content="I still suspect Noah.",
        ),
    ]

    agent.refresh_memory(turn=2, visible_history=visible_history, alive_player_names=alive_player_names)

    noah_belief = agent.role_beliefs_by_name["Noah"]
    assert noah_belief.evidence.count("sus") == 1


def test_repeated_day_signal_does_not_stack_score_after_dedupe() -> None:
    alive_player_names = ["Liam", "Noah", "Emma"]
    single_signal_agent = LLMAgent(name="Liam", model_id="m", role=Role.CITIZEN)
    repeated_signal_agent = LLMAgent(name="Liam", model_id="m", role=Role.CITIZEN)

    single_signal_history = [
        GameEvent(
            turn=1,
            phase=Phase.DAY,
            speaker="Emma",
            kind="speech",
            content="I suspect Noah.",
        )
    ]
    repeated_signal_history = [
        GameEvent(
            turn=1,
            phase=Phase.DAY,
            speaker="Emma",
            kind="speech",
            content="I suspect Noah.",
        ),
        GameEvent(
            turn=1,
            phase=Phase.DAY,
            speaker="Emma",
            kind="speech",
            content="I still suspect Noah.",
        ),
    ]

    single_signal_agent.refresh_memory(
        turn=2,
        visible_history=single_signal_history,
        alive_player_names=alive_player_names,
        inference_mode="day",
    )
    repeated_signal_agent.refresh_memory(
        turn=2,
        visible_history=repeated_signal_history,
        alive_player_names=alive_player_names,
        inference_mode="day",
    )

    single_belief = single_signal_agent.role_beliefs_by_name["Noah"]
    repeated_belief = repeated_signal_agent.role_beliefs_by_name["Noah"]
    assert repeated_belief.mafia == single_belief.mafia
