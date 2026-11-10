from __future__ import annotations

from typing import Any

import pytest

from src.agents.llm_agent import LLMAgent, _mentions_name
from src.engine.game_state import GameEvent, Phase, Role
from src.providers.chat_client import ProviderError


class RecordingClient:
    def __init__(self, model: str | None = "test/model", text: str = "Emma는 수상합니다.") -> None:
        self.model = model
        self.text = text
        self.calls: list[dict[str, object]] = []

    def chat_completion(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(kwargs)
        return {"text": self.text, "model": self.model}


@pytest.mark.parametrize("role", list(Role))
@pytest.mark.parametrize("phase", ["night", "day", "vote"])
def test_all_roles_and_phases_receive_korean_policy(role: Role, phase: str) -> None:
    client = RecordingClient()
    agent = LLMAgent(name="Alex", model_id="test/model:free", role=role, client=client)
    assert agent.speak(phase=phase, turn=1, prompt="Action") == "Emma는 수상합니다."
    request = client.calls[0]
    policy = request["messages"][0]["content"]
    assert "반드시 한국어" in policy
    for token in ("CHAT:", "VOTE_JSON:", "REQUEST:", "PASS:", "VOTE:", "target"):
        assert token in policy
    assert request["model"] == "test/model:free"
    assert "fallback_models" not in request


def test_non_korean_output_is_retried_once_with_the_same_model() -> None:
    class LanguageClient(RecordingClient):
        def chat_completion(self, **kwargs: Any) -> dict[str, object]:
            response = super().chat_completion(**kwargs)
            response["text"] = "I suspect Emma." if len(self.calls) == 1 else "Emma를 의심합니다."
            return response

    client = LanguageClient()
    agent = LLMAgent(name="Alex", model_id="test/model", role=Role.CITIZEN, client=client)
    assert agent.speak(phase="day", turn=1, prompt="Strategy") == "Emma를 의심합니다."
    assert [call["model"] for call in client.calls] == ["test/model", "test/model"]
    assert "같은 판단을 한국어로" in client.calls[-1]["messages"][-1]["content"]


def test_persistent_non_korean_output_stops_the_match() -> None:
    client = RecordingClient(text="I suspect Emma.")
    agent = LLMAgent(name="Alex", model_id="test/model", role=Role.CITIZEN, client=client)
    with pytest.raises(ProviderError, match="non-Korean response after language retry"):
        agent.speak(phase="day", turn=1, prompt="Strategy")
    assert len(client.calls) == 2


def test_vote_protocol_does_not_require_hangul() -> None:
    client = RecordingClient(text="VOTE: Emma")
    agent = LLMAgent(name="Alex", model_id="test/model", role=Role.CITIZEN, client=client)
    assert agent.speak(phase="vote", turn=1, prompt="Vote") == "VOTE: Emma"
    assert len(client.calls) == 1


def test_agent_propagates_api_failure_instead_of_fabricating_response() -> None:
    class FailingClient:
        def chat_completion(self, **kwargs: Any) -> dict[str, object]:
            raise ProviderError("quota exhausted")

    agent = LLMAgent(name="Alex", model_id="m", role=Role.CITIZEN, client=FailingClient())
    with pytest.raises(ProviderError, match="quota exhausted"):
        agent.speak(phase="day", turn=1, prompt="발언")


@pytest.mark.parametrize("model,text,error", [
    ("another/model", "안녕", "requested model"),
    (None, "안녕", "requested model"),
    ("test/model", " ", "empty response"),
])
def test_agent_rejects_switched_model_or_empty_response(model: str | None, text: str, error: str) -> None:
    agent = LLMAgent(name="Alex", model_id="test/model", role=Role.CITIZEN, client=RecordingClient(model, text))
    with pytest.raises(ProviderError, match=error):
        agent.speak(phase="day", turn=1, prompt="발언")


@pytest.mark.parametrize("content, tag, score", [
    ("Emma는 거짓말을 하므로 의심합니다.", "sus", "mafia"),
    ("Emma를 보호하고 치료하겠습니다.", "protect", "doctor"),
    ("Emma의 말이 일관되어 신뢰합니다.", "trust", "citizen"),
])
def test_korean_role_inference_handles_name_particles(content: str, tag: str, score: str) -> None:
    agent = LLMAgent(name="Alex", model_id="m", role=Role.CITIZEN)
    agent.refresh_memory(
        turn=1,
        visible_history=[GameEvent(turn=1, phase=Phase.DAY, speaker="Noah", kind="speech", content=content)],
        alive_player_names=["Alex", "Emma", "Noah"],
    )
    belief = agent.role_beliefs_by_name["Emma"]
    assert tag in belief.evidence
    assert getattr(belief, score) > 0.34
    assert not _mentions_name("Emma2는", "Emma")
    assert not _mentions_name("xEmma는", "Emma")
