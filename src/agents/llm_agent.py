from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re
from typing import Literal

from src.agents.prompt.personas import role_persona
from src.engine.game_state import GameEvent, Phase, Role
from src.providers.openrouter_client import OpenRouterClient, OpenRouterError


_MAX_OWN_DIALOGUE_ITEMS = 20
_MAX_EVIDENCE_ITEMS = 2
_MAX_ROLE_SIGNAL_EVENTS = 60
_MAX_HISTORY_DAY_EVENTS = 6
_MAX_HISTORY_NIGHT_EVENTS = 3
_MAX_HISTORY_CONTENT_CHARS = 140
_MAX_HISTORY_ROLE_SIGNALS = 4

_SUSPICION_KEYWORDS = {
    "suspect",
    "suspicion",
    "mafia",
    "lying",
    "lie",
    "contradict",
    "inconsistent",
    "deceptive",
    "vote out",
    "eliminate",
}

_PROTECTION_KEYWORDS = {
    "protect",
    "save",
    "shield",
    "guard",
    "doctor",
    "heal",
}

_TRUST_KEYWORDS = {
    "trust",
    "innocent",
    "citizen",
    "credible",
    "consistent",
    "reliable",
}

_ROLE_SIGNAL_KEYWORDS = _SUSPICION_KEYWORDS | _PROTECTION_KEYWORDS | _TRUST_KEYWORDS
_EVIDENCE_TAGS = {
    "vote",
    "invalid_vote",
    "sus",
    "protect",
    "trust",
    "night_kill_pattern",
    "night_claim_mismatch",
    "signal",
}

InferenceMode = Literal["day", "night", "combined"]


@dataclass(frozen=True)
class RoleBelief:
    mafia: float
    doctor: float
    citizen: float
    evidence: tuple[str, ...] = ()


@dataclass
class LLMAgent:
    name: str
    model_id: str
    role: Role
    client: OpenRouterClient | None = None
    fallback_models: list[str] = field(default_factory=list)
    own_dialogue_history: list[str] = field(default_factory=list)
    role_beliefs_by_name: dict[str, RoleBelief] = field(default_factory=dict)
    last_memory_turn: int = 0

    def build_speak_request_prompt(
        self, *, night_result: str, strategy: str, naming_instruction: str
    ) -> str:
        template = _load_prompt_template("speak_request.txt")
        return template.format(
            self_name=self.name,
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
            self_name=self.name,
            night_result=night_result,
            speaker_name=speaker_name,
            speech=speech,
            naming_instruction=naming_instruction,
        )

    def build_day_vote_prompt(
        self,
        *,
        self_speech_context: str,
        belief_context: str,
        naming_instruction: str,
    ) -> str:
        template = _load_prompt_template("day_vote.txt")
        return template.format(
            self_name=self.name,
            self_speech_context=self_speech_context,
            belief_context=belief_context,
            naming_instruction=naming_instruction,
        )

    def refresh_memory(
        self,
        *,
        turn: int,
        visible_history: list[GameEvent],
        alive_player_names: list[str],
        inference_mode: str = "day",
    ) -> None:
        normalized_mode = _normalize_inference_mode(inference_mode)
        self.own_dialogue_history = _extract_own_dialogue(self.name, visible_history)
        self.role_beliefs_by_name = _infer_role_beliefs(
            observer_name=self.name,
            visible_history=visible_history,
            alive_player_names=alive_player_names,
            inference_mode=normalized_mode,
        )
        self.last_memory_turn = turn
        self.trim_memory()

    def trim_memory(
        self,
        *,
        max_own_dialogue_items: int = _MAX_OWN_DIALOGUE_ITEMS,
        max_evidence_items: int = _MAX_EVIDENCE_ITEMS,
    ) -> None:
        if len(self.own_dialogue_history) > max_own_dialogue_items:
            self.own_dialogue_history = self.own_dialogue_history[
                -max_own_dialogue_items:
            ]

        trimmed_beliefs: dict[str, RoleBelief] = {}
        for target_name, belief in self.role_beliefs_by_name.items():
            trimmed_beliefs[target_name] = RoleBelief(
                mafia=belief.mafia,
                doctor=belief.doctor,
                citizen=belief.citizen,
                evidence=belief.evidence[-max_evidence_items:],
            )
        self.role_beliefs_by_name = trimmed_beliefs

    def build_own_dialogue_context(self) -> str:
        if not self.own_dialogue_history:
            return "- (no prior public statements)"
        lines = [
            f"- #{index}: {message}"
            for index, message in enumerate(self.own_dialogue_history, start=1)
        ]
        return "\n".join(lines)

    def build_belief_context(self, *, alive_player_names: list[str]) -> str:
        targets = [name for name in alive_player_names if name != self.name]
        if not targets:
            return "- (no alive targets to evaluate)"

        ordered_targets = sorted(
            targets,
            key=lambda target_name: self.role_beliefs_by_name.get(
                target_name, _default_role_belief()
            ).mafia,
            reverse=True,
        )
        lines: list[str] = []
        for target_name in ordered_targets:
            belief = self.role_beliefs_by_name.get(target_name, _default_role_belief())
            lines.append(_compact_belief_line(target_name, belief))
        return "\n".join(lines)

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
            return (
                f"[{self.name}/{self.model_id}] empty response on {phase} turn {turn}."
            )
        return text.strip()


def _history_to_context(history: list[GameEvent]) -> str:
    lines = ["Recent conversation context (budgeted):"]

    day_indices = [
        index
        for index, event in enumerate(history)
        if event.phase in {Phase.DAY, Phase.VOTE}
    ]
    night_indices = [
        index for index, event in enumerate(history) if event.phase == Phase.NIGHT
    ]
    selected_indices = set(
        day_indices[-_MAX_HISTORY_DAY_EVENTS:]
        + night_indices[-_MAX_HISTORY_NIGHT_EVENTS:]
    )

    for index, event in enumerate(history):
        if index not in selected_indices:
            continue
        compact_content = _truncate_event_content(event.content)
        lines.append(
            f"- t{event.turn} {event.phase.value}/{event.kind} {event.speaker}: {compact_content}"
        )
    recent = [event for index, event in enumerate(history) if index in selected_indices]
    role_signals: list[str] = []
    for event in recent:
        if event.kind not in {
            "speech",
            "strategy",
            "speak_request_reason",
            "day_vote",
            "day_vote_invalid",
        }:
            continue
        tags = _role_signal_tags_for_event(event)
        if tags:
            role_signals.append(f"t{event.turn} {event.speaker}:{'+'.join(tags)}")

    if role_signals:
        lines.append("Role-inference tags:")
        for signal in role_signals[-_MAX_HISTORY_ROLE_SIGNALS:]:
            lines.append(f"- {signal}")

    return "\n".join(lines)


def _extract_own_dialogue(
    observer_name: str, visible_history: list[GameEvent]
) -> list[str]:
    dialogue: list[str] = []
    for event in visible_history:
        if event.phase != Phase.DAY:
            continue
        if event.speaker != observer_name:
            continue
        if event.kind not in {"strategy", "speech"}:
            continue
        content = event.content.strip()
        if content:
            dialogue.append(content)
    return dialogue[-_MAX_OWN_DIALOGUE_ITEMS:]


def _infer_role_beliefs(
    *,
    observer_name: str,
    visible_history: list[GameEvent],
    alive_player_names: list[str],
    inference_mode: InferenceMode = "day",
) -> dict[str, RoleBelief]:
    targets = [name for name in alive_player_names if name != observer_name]
    if not targets:
        return {}

    score_by_target: dict[str, dict[str, float]] = {
        target: {"mafia": 0.34, "doctor": 0.33, "citizen": 0.33} for target in targets
    }
    evidence_by_target: dict[str, list[str]] = {target: [] for target in targets}
    evidence_seen_by_target: dict[str, set[tuple[str, str]]] = {
        target: set() for target in targets
    }

    if inference_mode in {"day", "combined"}:
        _apply_day_inference(
            visible_history=visible_history,
            targets=targets,
            score_by_target=score_by_target,
            evidence_by_target=evidence_by_target,
            evidence_seen_by_target=evidence_seen_by_target,
        )

    if inference_mode in {"night", "combined"}:
        _apply_night_inference(
            visible_history=visible_history,
            targets=targets,
            score_by_target=score_by_target,
            evidence_by_target=evidence_by_target,
            evidence_seen_by_target=evidence_seen_by_target,
        )

    beliefs: dict[str, RoleBelief] = {}
    for target_name in targets:
        normalized_scores = _normalize_scores(score_by_target[target_name])
        beliefs[target_name] = RoleBelief(
            mafia=normalized_scores["mafia"],
            doctor=normalized_scores["doctor"],
            citizen=normalized_scores["citizen"],
            evidence=tuple(evidence_by_target[target_name][-_MAX_EVIDENCE_ITEMS:]),
        )
    return beliefs


def _extract_day_vote_target_name(content: str) -> str | None:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    target_name = payload.get("target_name")
    if not isinstance(target_name, str):
        return None
    stripped = target_name.strip()
    if not stripped:
        return None
    return stripped


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    mafia = max(scores.get("mafia", 0.0), 0.0)
    doctor = max(scores.get("doctor", 0.0), 0.0)
    citizen = max(scores.get("citizen", 0.0), 0.0)
    total = mafia + doctor + citizen
    if total <= 0:
        baseline = _default_role_belief()
        return {
            "mafia": baseline.mafia,
            "doctor": baseline.doctor,
            "citizen": baseline.citizen,
        }
    return {
        "mafia": round(mafia / total, 2),
        "doctor": round(doctor / total, 2),
        "citizen": round(citizen / total, 2),
    }


def _append_evidence(evidence_items: list[str], evidence_text: str) -> None:
    text = evidence_text.strip()
    if not text:
        return
    if evidence_items and evidence_items[-1] == text:
        return
    evidence_items.append(text)


def _append_tagged_evidence(
    *,
    evidence_by_target: dict[str, list[str]],
    evidence_seen_by_target: dict[str, set[tuple[str, str]]],
    target_name: str,
    speaker: str,
    tag: str,
) -> bool:
    normalized_tag = _evidence_to_tag(tag)
    dedupe_key = (speaker.strip().lower(), normalized_tag)
    seen = evidence_seen_by_target[target_name]
    if dedupe_key in seen:
        return False
    seen.add(dedupe_key)
    _append_evidence(evidence_by_target[target_name], normalized_tag)
    return True


def _apply_day_inference(
    *,
    visible_history: list[GameEvent],
    targets: list[str],
    score_by_target: dict[str, dict[str, float]],
    evidence_by_target: dict[str, list[str]],
    evidence_seen_by_target: dict[str, set[tuple[str, str]]],
) -> None:
    day_events = [
        event for event in visible_history if event.phase in {Phase.DAY, Phase.VOTE}
    ]
    for event in day_events[-_MAX_ROLE_SIGNAL_EVENTS:]:
        content = event.content.strip()
        if not content:
            continue
        lowered = content.lower()

        if event.kind == "day_vote":
            target_name = _extract_day_vote_target_name(content)
            if target_name in score_by_target:
                should_apply = _append_tagged_evidence(
                    evidence_by_target=evidence_by_target,
                    evidence_seen_by_target=evidence_seen_by_target,
                    target_name=target_name,
                    speaker=event.speaker,
                    tag="vote",
                )
                if should_apply:
                    score_by_target[target_name]["mafia"] += 0.14
            continue

        if event.kind == "day_vote_invalid":
            if event.speaker in score_by_target:
                should_apply = _append_tagged_evidence(
                    evidence_by_target=evidence_by_target,
                    evidence_seen_by_target=evidence_seen_by_target,
                    target_name=event.speaker,
                    speaker=event.speaker,
                    tag="invalid_vote",
                )
                if should_apply:
                    score_by_target[event.speaker]["mafia"] += 0.06
            continue

        if event.kind not in {"strategy", "speech", "speak_request_reason"}:
            continue

        for target_name in targets:
            if not _mentions_name(content, target_name):
                continue

            if _contains_keyword(lowered, _SUSPICION_KEYWORDS):
                should_apply = _append_tagged_evidence(
                    evidence_by_target=evidence_by_target,
                    evidence_seen_by_target=evidence_seen_by_target,
                    target_name=target_name,
                    speaker=event.speaker,
                    tag="sus",
                )
                if should_apply:
                    score_by_target[target_name]["mafia"] += 0.20

            if _contains_keyword(lowered, _PROTECTION_KEYWORDS):
                should_apply = _append_tagged_evidence(
                    evidence_by_target=evidence_by_target,
                    evidence_seen_by_target=evidence_seen_by_target,
                    target_name=target_name,
                    speaker=event.speaker,
                    tag="protect",
                )
                if should_apply:
                    score_by_target[target_name]["doctor"] += 0.14

            if _contains_keyword(lowered, _TRUST_KEYWORDS):
                should_apply = _append_tagged_evidence(
                    evidence_by_target=evidence_by_target,
                    evidence_seen_by_target=evidence_seen_by_target,
                    target_name=target_name,
                    speaker=event.speaker,
                    tag="trust",
                )
                if should_apply:
                    score_by_target[target_name]["citizen"] += 0.15


def _apply_night_inference(
    *,
    visible_history: list[GameEvent],
    targets: list[str],
    score_by_target: dict[str, dict[str, float]],
    evidence_by_target: dict[str, list[str]],
    evidence_seen_by_target: dict[str, set[tuple[str, str]]],
) -> None:
    night_events = [event for event in visible_history if event.phase == Phase.NIGHT]
    for event in night_events[-_MAX_ROLE_SIGNAL_EVENTS:]:
        content = event.content.strip()
        if not content:
            continue

        if event.kind == "mafia_vote":
            target_name = _extract_day_vote_target_name(content)
            if target_name in score_by_target:
                should_apply = _append_tagged_evidence(
                    evidence_by_target=evidence_by_target,
                    evidence_seen_by_target=evidence_seen_by_target,
                    target_name=target_name,
                    speaker=event.speaker,
                    tag="night_kill_pattern",
                )
                if should_apply:
                    score_by_target[target_name]["mafia"] += 0.18
            continue

        if event.kind == "mafia_vote_invalid" and event.speaker in score_by_target:
            should_apply = _append_tagged_evidence(
                evidence_by_target=evidence_by_target,
                evidence_seen_by_target=evidence_seen_by_target,
                target_name=event.speaker,
                speaker=event.speaker,
                tag="night_claim_mismatch",
            )
            if should_apply:
                score_by_target[event.speaker]["mafia"] += 0.05
            continue

        if event.kind == "mafia_consensus" and event.content.startswith(
            "Mafia consensus target:"
        ):
            _, _, suffix = event.content.partition(":")
            target_name = suffix.strip().removesuffix(".")
            if target_name in score_by_target:
                should_apply = _append_tagged_evidence(
                    evidence_by_target=evidence_by_target,
                    evidence_seen_by_target=evidence_seen_by_target,
                    target_name=target_name,
                    speaker=event.speaker,
                    tag="night_kill_pattern",
                )
                if should_apply:
                    score_by_target[target_name]["mafia"] += 0.10
            continue

        if event.kind != "mafia_chat":
            continue

        lowered = content.lower()
        for target_name in targets:
            if not _mentions_name(content, target_name):
                continue

            if _contains_keyword(lowered, _SUSPICION_KEYWORDS):
                should_apply = _append_tagged_evidence(
                    evidence_by_target=evidence_by_target,
                    evidence_seen_by_target=evidence_seen_by_target,
                    target_name=target_name,
                    speaker=event.speaker,
                    tag="night_kill_pattern",
                )
                if should_apply:
                    score_by_target[target_name]["mafia"] += 0.10


def _evidence_to_tag(evidence_text: str) -> str:
    lowered = evidence_text.strip().lower()
    if not lowered:
        return "signal"
    if lowered in _EVIDENCE_TAGS:
        return lowered
    if "invalid vote" in lowered:
        return "invalid_vote"
    if "voted" in lowered:
        return "vote"
    if "suspicious" in lowered or "suspect" in lowered or "mafia" in lowered:
        return "sus"
    if "protect" in lowered or "doctor" in lowered or "heal" in lowered:
        return "protect"
    if "trust" in lowered or "citizen" in lowered or "reliable" in lowered:
        return "trust"
    if "night" in lowered and "vote" in lowered:
        return "night_kill_pattern"
    return "signal"


def _normalize_inference_mode(inference_mode: str) -> InferenceMode:
    normalized = inference_mode.strip().lower()
    if normalized == "day":
        return "day"
    if normalized == "night":
        return "night"
    if normalized == "combined":
        return "combined"
    return "day"


def _compact_belief_line(target_name: str, belief: RoleBelief) -> str:
    tags: list[str] = []
    for evidence_text in belief.evidence:
        tag = _evidence_to_tag(evidence_text)
        if tag in tags:
            continue
        tags.append(tag)
        if len(tags) >= _MAX_EVIDENCE_ITEMS:
            break
    evidence_repr = "+".join(tags) if tags else "none"
    return f"- {target_name} | M:{belief.mafia:.2f} D:{belief.doctor:.2f} C:{belief.citizen:.2f} | E:{evidence_repr}"


def _truncate_event_content(content: str) -> str:
    compact = " ".join(content.strip().split())
    if len(compact) <= _MAX_HISTORY_CONTENT_CHARS:
        return compact
    return f"{compact[:_MAX_HISTORY_CONTENT_CHARS - 3]}..."


def _role_signal_tags_for_event(event: GameEvent) -> list[str]:
    tags: list[str] = []
    content = event.content.lower()
    if _contains_keyword(content, _SUSPICION_KEYWORDS):
        tags.append("sus")
    if _contains_keyword(content, _PROTECTION_KEYWORDS):
        tags.append("protect")
    if _contains_keyword(content, _TRUST_KEYWORDS):
        tags.append("trust")
    if event.kind == "day_vote":
        tags.append("vote")
    if event.kind == "day_vote_invalid":
        tags.append("invalid_vote")
    return tags


def _contains_keyword(content: str, keywords: set[str]) -> bool:
    return any(keyword in content for keyword in keywords)


def _mentions_name(content: str, target_name: str) -> bool:
    return (
        re.search(rf"\b{re.escape(target_name)}\b", content, flags=re.IGNORECASE)
        is not None
    )


def _default_role_belief() -> RoleBelief:
    return RoleBelief(mafia=0.34, doctor=0.33, citizen=0.33, evidence=())


@lru_cache(maxsize=None)
def _load_prompt_template(filename: str) -> str:
    prompt_path = Path(__file__).resolve().parent / "prompt" / filename
    return prompt_path.read_text(encoding="utf-8").strip()
