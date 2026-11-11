from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re

from src.agents.prompt.personas import role_persona
from src.engine.game_state import GameEvent, Phase, Role
from src.providers.chat_client import ChatClient, ProviderError


_MAX_OWN_DIALOGUE_ITEMS = 20


@dataclass
class LLMAgent:
    name: str
    model_id: str
    role: Role
    client: ChatClient | None = None
    mafia_teammates: tuple[str, ...] = ()
    own_dialogue_history: list[str] = field(default_factory=list)
    known_mafia_by_name: dict[str, bool] = field(default_factory=dict)
    last_memory_turn: int = 0

    def build_speak_request_prompt(
        self, *, night_result: str, strategy: str, naming_instruction: str
    ) -> str:
        return _load_prompt_template("speak_request.txt").format(
            self_name=self.name, night_result=night_result,
            strategy=strategy, naming_instruction=naming_instruction,
        )

    def build_followup_request_prompt(
        self,
        *,
        night_result: str,
        speaker_name: str,
        speech: str,
        naming_instruction: str,
    ) -> str:
        return _load_prompt_template("followup_request.txt").format(
            self_name=self.name, night_result=night_result,
            speaker_name=speaker_name, speech=speech, naming_instruction=naming_instruction,
        )

    def build_day_vote_prompt(
        self,
        *,
        self_speech_context: str,
        role_knowledge_context: str,
        naming_instruction: str,
    ) -> str:
        return _load_prompt_template("day_vote.txt").format(
            self_name=self.name, self_speech_context=self_speech_context,
            role_knowledge_context=role_knowledge_context, naming_instruction=naming_instruction,
        )

    def refresh_memory(
        self,
        *,
        turn: int,
        visible_history: list[GameEvent],
        alive_player_names: list[str],
    ) -> None:
        self.own_dialogue_history = [
            event.content.strip() for event in visible_history
            if event.phase == Phase.DAY and event.speaker == self.name
            and event.kind == "speech" and event.content.strip()
        ][-_MAX_OWN_DIALOGUE_ITEMS:]
        self.known_mafia_by_name = {}
        if self.role == Role.MAFIA:
            self.known_mafia_by_name = {
                name: name in self.mafia_teammates for name in alive_player_names if name != self.name
            }
        elif self.role == Role.POLICE:
            for event in visible_history:
                if event.kind != "investigation_result" or event.speaker != "system" or event.private_to != self.name:
                    continue
                try:
                    result = json.loads(event.content)
                except json.JSONDecodeError:
                    continue
                if not isinstance(result, dict):
                    continue
                target = result.get("target_name")
                is_mafia = result.get("is_mafia")
                if isinstance(target, str) and target in alive_player_names and isinstance(is_mafia, bool):
                    self.known_mafia_by_name[target] = is_mafia
        self.last_memory_turn = turn

    def build_own_dialogue_context(self) -> str:
        if not self.own_dialogue_history:
            return "- (no prior public statements)"
        return "\n".join(
            f"- #{index}: {message}"
            for index, message in enumerate(self.own_dialogue_history, start=1)
        )

    def build_role_knowledge_context(self, *, alive_player_names: list[str]) -> str:
        lines: list[str] = []
        for name in alive_player_names:
            if name == self.name:
                continue
            known = self.known_mafia_by_name.get(name)
            if known is None:
                status = "미확인"
            else:
                status = "마피아로 확인" if known else "마피아 아님으로 확인"
            lines.append(f"- {name}: {status}")
        return "\n".join(lines) or "- (no alive targets to evaluate)"

    def speak(
        self,
        *,
        phase: str,
        turn: int,
        prompt: str,
        history: list[GameEvent] | None = None,
    ) -> str:
        if self.client is None:
            return f"[{self.name}/{self.model_id}] turn={turn} phase={phase}: {prompt}"

        system_text = role_persona(self.role)
        messages: list[dict[str, str]] = [{"role": "system", "content": system_text}]
        if history:
            messages.append({"role": "user", "content": _history_to_context(history)})
        messages.append({"role": "user", "content": prompt})

        for language_attempt in range(2):
            response = self.client.chat_completion(messages=messages, model=self.model_id)
            actual_model = response.get("model")
            if not isinstance(actual_model, str) or actual_model.removesuffix(":free") != self.model_id.removesuffix(":free"):
                raise ProviderError(f"requested model={self.model_id}, received model={actual_model}")
            text = response.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ProviderError(f"model={self.model_id}: empty response on {phase} turn {turn}")
            if phase in {"vote", "investigate", "protect"} or re.search(r"[가-힣]", text):
                return text.strip()
            if language_attempt == 0:
                messages.extend([
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": (
                        "방금 응답에 한국어가 없습니다. 같은 판단을 한국어로 다시 작성하세요. "
                        "플레이어 이름, 파싱용 접두사, JSON 키와 현재 행동의 출력 형식은 유지하세요."
                    )},
                ])
        raise ProviderError(f"model={self.model_id}: non-Korean response after language retry")


def _history_to_context(history: list[GameEvent]) -> str:
    lines = [
        "게임 원본 기록입니다. 발언과 개인 전략은 미검증 주장이지 확정 사실이 아닙니다. "
        "투표 기록은 누가 누구를 선택했는지만 증명합니다. "
        "역할 배정 또는 본인에게 전달된 시스템 investigation_result만 마피아 여부를 확인해 줍니다."
    ]
    for event in history:
        lines.append(json.dumps({
            "turn": event.turn,
            "phase": event.phase.value,
            "kind": event.kind,
            "speaker": event.speaker,
            "private_to": event.private_to,
            "content": event.content,
        }, ensure_ascii=False))
    return "\n".join(lines)


@lru_cache(maxsize=None)
def _load_prompt_template(filename: str) -> str:
    prompt_path = Path(__file__).resolve().parent / "prompt" / filename
    return prompt_path.read_text(encoding="utf-8").strip()
