from __future__ import annotations

import json
from typing import Any

from src.agents.llm_agent import LLMAgent
from src.engine.game_state import GameEvent, GameState, Phase, Player, Role
from src.runner.single_match import _collect_day_vote_ballots


class VoteClient:
    def __init__(self, target: str) -> None:
        self.target = target
        self.messages: list[dict[str, str]] = []

    def chat_completion(self, **kwargs: Any) -> dict[str, str]:
        self.messages = kwargs["messages"]
        return {"model": "test", "text": f"VOTE: {self.target}"}


def test_voters_get_full_public_history_and_only_their_own_secrets_without_current_ballots() -> None:
    players = [
        Player(id=1, name="Alex", model_name="test", model_id="test", role=Role.CITIZEN),
        Player(id=2, name="Emma", model_name="test", model_id="test", role=Role.POLICE),
        Player(id=3, name="Noah", model_name="test", model_id="test", role=Role.DOCTOR),
        Player(id=4, name="Liam", model_name="test", model_id="test", role=Role.MAFIA),
    ]
    state = GameState(turn=3, phase=Phase.VOTE, players=players)
    old_speech = "첫날 공개 발언 " + "중요한 근거 " * 100
    events = [
        GameEvent(turn=1, phase=Phase.DAY, speaker="Alex", kind="speech", content=old_speech),
        GameEvent(turn=1, phase=Phase.VOTE, speaker="Noah", kind="day_vote", content='{"target_name": "Liam"}'),
        GameEvent(turn=1, phase=Phase.NIGHT, speaker="system", kind="investigation_result",
                  content='{"target_name": "Liam", "is_mafia": true}', private_to="Emma"),
        GameEvent(turn=2, phase=Phase.NIGHT, speaker="Noah", kind="night_action", content="doctor secret", private_to="Noah"),
        GameEvent(turn=2, phase=Phase.NIGHT, speaker="Liam", kind="mafia_chat", content="mafia secret"),
        GameEvent(turn=3, phase=Phase.DAY, speaker="Emma", kind="strategy", content="cop private plan", private_to="Emma"),
        GameEvent(turn=3, phase=Phase.DAY, speaker="Emma", kind="speak_request_reason", content="cop private reason", private_to="Emma"),
    ]
    events.extend(GameEvent(turn=2, phase=Phase.DAY, speaker="Alex", kind="speech", content=f"후속 발언 {i}") for i in range(12))
    clients = {p.id: VoteClient("Liam" if p.name != "Liam" else "Alex") for p in players}
    agents = {p.id: LLMAgent(name=p.name, model_id="test", role=p.role, client=clients[p.id]) for p in players}

    assert _collect_day_vote_ballots(state, agents, events) == {1: 4, 2: 4, 3: 4, 4: 1}
    for player in players:
        context = clients[player.id].messages[1]["content"]
        assert old_speech in context
        records = [json.loads(line) for line in context.splitlines()[1:]]
        votes = [event for event in records if event["kind"] == "day_vote"]
        assert len(votes) == 1
        assert votes[0]["turn"] == 1
        assert ("cop private plan" in context) == (player.name == "Emma")
        assert ("cop private reason" in context) == (player.name == "Emma")
        assert any(event["kind"] == "investigation_result" for event in records) == (player.name == "Emma")
        assert ("doctor secret" in context) == (player.name == "Noah")
        assert ("mafia secret" in context) == (player.name == "Liam")
        vote_prompt = clients[player.id].messages[-1]["content"]
        if player.name == "Emma":
            assert "Liam: 마피아로 확인" in vote_prompt
        elif player.name != "Liam":
            assert "Liam: 미확인" in vote_prompt
