from __future__ import annotations

from src.engine.game_state import GameEvent, Phase, Player, Role
from src.runner.single_match import _build_day_speech_prompt, _build_self_speech_context


def _build_player() -> Player:
    return Player(id=1, name="Alex", model_name="m", model_id="m", role=Role.CITIZEN, alive=True)


def test_build_day_speech_prompt_reflects_speech_number_and_prior_context() -> None:
    player = _build_player()
    events = [
        GameEvent(turn=1, phase=Phase.DAY, speaker="Alex", kind="strategy", content="I suspect Emma."),
    ]

    prompt_first = _build_day_speech_prompt(
        player=player,
        speech_number=1,
        max_speeches_per_player=2,
        night_result="Night result: no one was eliminated.",
        strategy="I suspect Emma.",
        self_speech_context=_build_self_speech_context(events, player),
        naming_instruction="Use only these player names when referring to others: Alex, Emma.",
    )

    events.append(
        GameEvent(turn=1, phase=Phase.DAY, speaker="Alex", kind="speech", content="Emma contradicted herself."),
    )

    prompt_second = _build_day_speech_prompt(
        player=player,
        speech_number=2,
        max_speeches_per_player=2,
        night_result="Night result: no one was eliminated.",
        strategy="I suspect Emma.",
        self_speech_context=_build_self_speech_context(events, player),
        naming_instruction="Use only these player names when referring to others: Alex, Emma.",
    )

    assert "speech #1 out of 2" in prompt_first
    assert "speech #2 out of 2" in prompt_second
    assert "Emma contradicted herself." in prompt_second
    assert prompt_first != prompt_second
