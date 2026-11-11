from __future__ import annotations

import pytest

from src.agents.prompt.personas import role_persona
from src.engine.game_state import GameEvent, GameState, Phase, Player, Role
from src.runner.single_match import _build_day_speech_prompt, _record_setup


@pytest.mark.parametrize("role", list(Role))
def test_personas_allow_uncertainty_and_distinguish_claims_from_game_facts(role: Role) -> None:
    prompt = role_persona(role)
    assert "첫날 밤 이전에는 토론이나 투표가 없었습니다" in prompt
    assert "근거가 부족하다" in prompt
    assert "플레이어의 주장을 구분" in prompt
    assert "확정되지는 않습니다" in prompt
    assert "소속 진영의 승리" in prompt
    assert "마피아의 공개적인 전략적 기만은 허용" in prompt
    assert "include one observed inconsistency" not in prompt
    assert "include one concrete observation" not in prompt


def test_public_speech_does_not_require_fabricating_new_clues_or_reading_private_strategy() -> None:
    player = Player(id=1, name="Alex", model_name="m", model_id="m", role=Role.POLICE)
    prompt = _build_day_speech_prompt(
        player=player, speech_number=1, max_speeches_per_player=1,
        night_result="No one died", strategy="private investigation", self_speech_context="none", naming_instruction="names",
    )
    assert "do not fabricate a clue or force an accusation" in prompt
    assert "it is fine to say there is no new evidence" in prompt
    assert "decide what to reveal" in prompt


def test_setup_explains_starting_phase_and_public_role_counts_without_assignments() -> None:
    state = GameState(turn=1, phase=Phase.SETUP, players=[
        Player(id=1, name="Alex", model_name="m", model_id="m", role=Role.MAFIA),
        Player(id=2, name="Emma", model_name="m", model_id="m", role=Role.CITIZEN),
    ])
    events: list[GameEvent] = []
    _record_setup(state, events, progress_callback=None)
    assert "mafia=1" in events[0].content
    assert "citizen=1" in events[0].content
    assert "no discussion or vote has occurred yet" in events[0].content
    assert "Alex" not in events[0].content
    assert "Emma" not in events[0].content
