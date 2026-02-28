from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from src.agents.prompt.personas import role_persona
from src.engine.game_state import GameEvent, Role
from src.providers.openrouter_client import OpenRouterClient, OpenRouterError


@dataclass
class LLMAgent:
    name: str
    model_id: str
    role: Role
    client: OpenRouterClient | None = None
    fallback_models: list[str] = field(default_factory=list)

    def build_speak_request_prompt(self, *, night_result: str, strategy: str, naming_instruction: str) -> str:
        template = _load_prompt_template("speak_request.txt")
        return template.format(
            night_result=night_result,
            strategy=strategy,
            naming_instruction=naming_instruction,
        )

    def build_followup_request_prompt(
        self,
        *,
        night_result: str,
        speaker_name: str,
        speech: str,
        naming_instruction: str,
    ) -> str:
        template = _load_prompt_template("followup_request.txt")
        return template.format(
            night_result=night_result,
            speaker_name=speaker_name,
            speech=speech,
            naming_instruction=naming_instruction,
        )

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
            messages.append(
                {
                    "role": "user",
                    "content": _history_to_context(history),
                }
            )

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_id,
                fallback_models=self.fallback_models,
            )
        except OpenRouterError:
            return f"[{self.name}/{self.model_id}] fallback response on {phase} turn {turn}."
        text = response.get("text")
        if not isinstance(text, str) or not text.strip():
            return f"[{self.name}/{self.model_id}] empty response on {phase} turn {turn}."
        return text.strip()


def _history_to_context(history: list[GameEvent]) -> str:
    lines = ["Conversation history so far:"]
    for event in history:
        lines.append(
            f"- turn={event.turn} phase={event.phase.value} speaker={event.speaker} kind={event.kind}: {event.content}"
        )
    recent = history[-8:]
    suspicion_signals: list[str] = []
    for event in recent:
        if event.kind not in {"speech", "strategy"}:
            continue
        lowered = event.content.lower()
        if "suspect" in lowered or "suspicion" in lowered or "mafia" in lowered:
            suspicion_signals.append(f"turn={event.turn} speaker={event.speaker}: {event.content}")

    if suspicion_signals:
        lines.append("Recent suspicion signals:")
        for signal in suspicion_signals[-3:]:
            lines.append(f"- {signal}")

    return "\n".join(lines)


@lru_cache(maxsize=None)
def _load_prompt_template(filename: str) -> str:
    prompt_path = Path(__file__).resolve().parent / "prompt" / filename
    return prompt_path.read_text(encoding="utf-8").strip()
