"""Pure view-model logic for the web dashboard.

This module converts raw runner progress payloads into view-ready events and
keeps the dashboard state (turn/phase, player status, speech queue, live chat,
vote summaries, provider retries). It must stay free of I/O and framework
imports so it is easy to unit test.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.runner.single_match import MatchResult

# Avatar values are icon names rendered by ``static/icons.js``.
AGENT_AVATARS = ["bot", "brain", "cpu", "sparkles", "orbit", "flask-conical", "satellite", "atom"]
SYSTEM_AVATAR = "terminal"
_SYSTEM_KIND_AVATARS = {
    "setup": "settings",
    "night_result": "moon",
    "vote_result": "gavel",
    "game_end": "flag",
}

_HIDDEN_EVENT_KINDS = frozenset(
    {
        "day_vote",
        "mafia_vote",
        "day_vote_invalid",
        "mafia_vote_invalid",
        "speak_request",
        "speak_request_reason",
    }
)

_HIDDEN_MESSAGE_PREFIXES = (
    "now speaking:",
    "follow-up requests processed after ",
)

_SQUELCHED_PROGRESS_KINDS = frozenset({"speech_queue", "provider_retry"})

_CHAT_EVENT_KINDS = frozenset(
    {
        "setup",
        "night_result",
        "vote_result",
        "game_end",
        "speech",
        "mafia_chat",
        "day_vote",
        "mafia_vote",
    }
)


def should_render_chat_event(kind: str, message: str) -> bool:
    normalized_kind = kind.strip().lower()
    if normalized_kind in _HIDDEN_EVENT_KINDS:
        return False

    if "speech_queue" in normalized_kind:
        return False

    normalized_message = message.strip().lower()
    return not normalized_message.startswith(_HIDDEN_MESSAGE_PREFIXES)


def extract_vote_target_name(message: str) -> str | None:
    """Extract ``target_name`` from a JSON-encoded vote message."""
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    target_name = payload.get("target_name")
    if isinstance(target_name, str) and target_name.strip():
        return target_name.strip()
    return None


def extract_mafia_chat_message(message: str) -> str:
    """Extract the human-readable chat line from a mafia chat payload."""
    chat_line: str | None = None
    lines: list[str] = []
    for raw_line in message.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.upper().startswith("CHAT:"):
            candidate = line.split(":", maxsplit=1)[1].strip()
            if candidate:
                chat_line = candidate
            continue
        if line.upper().startswith("VOTE_JSON:"):
            continue
        lines.append(line)

    if chat_line:
        return chat_line
    if lines:
        return "\n".join(lines)
    return message.strip()


def format_chat_entry(*, kind: str, speaker: str, message: str) -> tuple[str, str]:
    """Format a chat entry into ``(speaker, message)`` for rendering."""
    normalized_kind = kind.strip().lower()
    if normalized_kind == "mafia_chat":
        return speaker, extract_mafia_chat_message(message)

    if normalized_kind in {"day_vote", "mafia_vote"}:
        vote_label = "Day vote" if normalized_kind == "day_vote" else "Night vote"
        target_name = extract_vote_target_name(message)
        if target_name and speaker.strip():
            return "system", f"{vote_label}: {speaker} -> {target_name}"
        return "system", message

    return speaker, message


def chat_role_and_avatar(speaker: str, kind: str = "") -> tuple[str, str]:
    """Return ``(role, avatar_icon)`` where role is ``"system"`` or ``"agent"``."""
    if speaker == "system":
        return "system", _SYSTEM_KIND_AVATARS.get(kind.strip().lower(), SYSTEM_AVATAR)

    hashed = int(hashlib.sha1(speaker.encode("utf-8")).hexdigest(), 16)
    return "agent", AGENT_AVATARS[hashed % len(AGENT_AVATARS)]


def normalize_vote_result_message(*, vote_type: str, message: str) -> str:
    normalized = message.strip()
    if vote_type == "night" and " mafia_target=" in normalized:
        normalized = normalized.split(" mafia_target=", maxsplit=1)[0].strip()
    return normalized


def progress_text_for_event(*, kind: str, speaker: str, message: str) -> str:
    normalized_kind = kind.strip().lower()
    normalized_speaker = speaker.strip()
    normalized_message = message.strip()

    if normalized_kind in {"speech", "mafia_chat"}:
        if normalized_speaker:
            return f"Streaming live chat from {normalized_speaker}"
        return "Streaming live chat"

    if normalized_kind == "day_vote":
        if normalized_speaker:
            return f"{normalized_speaker} submitted a day vote"
        return "Submitting day vote"

    if normalized_kind == "mafia_vote":
        if normalized_speaker:
            return f"{normalized_speaker} submitted a night vote"
        return "Submitting night vote"

    if normalized_kind == "provider_retry":
        return "Retrying provider call"

    if normalized_kind in {"night_result", "vote_result", "game_end"} and normalized_message:
        return normalized_message

    if normalized_kind == "agent_thinking":
        if normalized_speaker:
            return f"{normalized_speaker} is thinking"
        return "Agent is thinking"

    if normalized_kind == "agent_spoke":
        if normalized_speaker:
            return f"{normalized_speaker} finished speaking"
        return "Agent finished speaking"

    if normalized_kind == "phase":
        return "Advancing game phase"

    if normalized_kind == "setup":
        return "Preparing match"

    if normalized_kind:
        return f"Processing {normalized_kind.replace('_', ' ')}"
    return "Match in progress"


def clean_player_status(items: object) -> list[dict[str, Any]]:
    """Normalize player status payloads into well-formed dicts."""
    if not isinstance(items, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in items:
        if not (
            isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and isinstance(item.get("model_name"), str)
            and isinstance(item.get("role"), str)
            and isinstance(item.get("alive"), bool)
        ):
            continue
        normalized.append(
            {
                "name": item["name"],
                "model_name": item["model_name"],
                "role": item["role"],
                "alive": item["alive"],
            }
        )
    return normalized


def player_status_from_players(players: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": player.name,
            "model_name": player.model_name,
            "role": player.role.value,
            "alive": player.alive,
        }
        for player in players
    ]


def cycle_text(turn: int, phase: str) -> str:
    if phase == "day":
        return f"Day {turn}"
    if phase == "night":
        return f"Night {turn}"
    if phase == "vote":
        return f"Day {turn} (Voting)"
    if phase == "setup":
        return "Preparing game"
    return "Game over"


def _vote_rows_from_live_events(
    events: list[dict[str, Any]],
    *,
    vote_kind: str,
    vote_phase: str,
    vote_type: str,
    turn_token: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in events:
        kind = str(item.get("kind", "")).strip().lower()
        phase = str(item.get("phase", "")).strip().lower()
        turn = str(item.get("turn", "")).strip()
        if kind != vote_kind or phase != vote_phase or turn != turn_token:
            continue

        target_name = _target_name_from_vote_event(item)
        if target_name is None:
            continue

        voter_name = str(item.get("speaker", "")).strip() or "unknown"
        rows.append(
            {
                "type": vote_type,
                "voter": voter_name,
                "target": target_name,
            }
        )
    return rows


def _target_name_from_vote_event(item: dict[str, Any]) -> str | None:
    raw_target_name = item.get("target_name")
    if isinstance(raw_target_name, str) and raw_target_name.strip():
        return raw_target_name.strip()

    message = str(item.get("message", ""))
    target_name = extract_vote_target_name(message)
    if target_name is not None:
        return target_name

    sentence_match = re.search(r"voted\s+to\s+eliminate\s+(.+?)(?:\.|$)", message, flags=re.IGNORECASE)
    if sentence_match is None:
        return None

    candidate = sentence_match.group(1).strip()
    if not candidate:
        return None
    return candidate


def vote_summary_rows_for_chat_event(
    live_vote_events: list[dict[str, Any]],
    *,
    kind: str,
    turn: object,
) -> list[dict[str, str]] | None:
    """Aggregate vote rows shown with a ``vote_result``/``night_result`` chat event."""
    normalized_kind = kind.strip().lower()
    turn_token = str(turn).strip()
    if normalized_kind == "vote_result":
        rows = _vote_rows_from_live_events(
            live_vote_events,
            vote_kind="day_vote",
            vote_phase="vote",
            vote_type="day",
            turn_token=turn_token,
        )
        return rows or None
    if normalized_kind == "night_result":
        rows = _vote_rows_from_live_events(
            live_vote_events,
            vote_kind="mafia_vote",
            vote_phase="night",
            vote_type="night",
            turn_token=turn_token,
        )
        return rows or None
    return None


@dataclass
class SpeechQueueState:
    queue: list[str] = field(default_factory=list)
    active_speaker: str | None = None
    total_slots: int = 0
    queue_turn: int | None = None

    @property
    def active_count(self) -> int:
        return 1 if self.active_speaker else 0

    @property
    def completed_slots(self) -> int:
        return max(self.total_slots - len(self.queue) - self.active_count, 0)

    def snapshot(self) -> dict[str, Any]:
        return {
            "queue": list(self.queue),
            "active_speaker": self.active_speaker,
            "total_slots": self.total_slots,
            "completed_slots": self.completed_slots,
        }


@dataclass
class DashboardState:
    """Aggregates runner progress into dashboard view state and SSE events."""

    turn: int = 1
    phase: str = "setup"
    players_status: list[dict[str, Any]] = field(default_factory=list)
    live_chat_events: list[dict[str, Any]] = field(default_factory=list)
    live_vote_events: list[dict[str, Any]] = field(default_factory=list)
    retries: list[str] = field(default_factory=list)
    speech_queue: SpeechQueueState = field(default_factory=SpeechQueueState)
    progress_step: int = 0

    def reset(self) -> None:
        self.turn = 1
        self.phase = "setup"
        self.players_status = []
        self.live_chat_events = []
        self.live_vote_events = []
        self.retries = []
        self.speech_queue = SpeechQueueState()
        self.progress_step = 0

    def to_view(self, result: MatchResult) -> None:
        """Sync state fields from a finished match result."""
        if not result.events:
            return
        last = result.events[-1]
        self.turn = last.turn
        self.phase = last.phase.value
        self.players_status = player_status_from_players(result.state.players)
        self.speech_queue = SpeechQueueState()

    def result_snapshot(self, result: MatchResult) -> dict[str, Any]:
        return {
            "current": cycle_text(self.turn, self.phase),
            "players_status": list(self.players_status),
            "output_dir": str(result.output_dir),
            "events_path": str(result.events_path),
            "summary_path": str(result.summary_path),
        }

    def state_event(self) -> dict[str, Any]:
        return {
            "type": "state",
            "current": cycle_text(self.turn, self.phase),
            "turn": self.turn,
            "phase": self.phase,
            "players_status": list(self.players_status),
            "speech_queue": self.speech_queue.snapshot(),
        }

    def _chat_event(self, kind: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        speaker = str(payload.get("speaker", "")).strip() or "system"
        message = str(payload.get("message", "")) or "Match in progress"
        turn = payload.get("turn", self.turn)
        phase = str(payload.get("phase", self.phase))

        formatted_speaker, formatted_message = format_chat_entry(
            kind=kind,
            speaker=speaker,
            message=message,
        )
        role, avatar = chat_role_and_avatar(formatted_speaker, kind)
        row_turn = turn if isinstance(turn, int) else str(turn)
        vote_summary_rows = vote_summary_rows_for_chat_event(
            self.live_vote_events,
            kind=kind,
            turn=row_turn,
        )

        event: dict[str, Any] = {
            "type": "chat",
            "kind": kind or "progress",
            "speaker": formatted_speaker,
            "message": formatted_message,
            "turn": str(row_turn),
            "phase": phase,
            "role": role,
            "avatar": avatar,
        }

        normalized_kind = kind.strip().lower()
        if vote_summary_rows:
            event["vote_summary_rows"] = vote_summary_rows
            if normalized_kind in {"vote_result", "night_result"}:
                vote_type = "day" if normalized_kind == "vote_result" else "night"
                result_text = normalize_vote_result_message(
                    vote_type=vote_type,
                    message=formatted_message,
                )
                if result_text:
                    event["vote_result_text"] = result_text

        self.live_chat_events.append(event)
        return event

    def _apply_baseline(self, payload: dict[str, Any]) -> None:
        turn = payload.get("turn")
        if isinstance(turn, int):
            self.turn = turn
        phase = payload.get("phase")
        if isinstance(phase, str) and phase:
            self.phase = phase

        players_status = clean_player_status(payload.get("players_status"))
        if players_status:
            self.players_status = players_status

        raw_speech_queue = payload.get("speech_queue")
        if isinstance(raw_speech_queue, list):
            self.speech_queue.queue = [item for item in raw_speech_queue if isinstance(item, str)]

    def _track_vote_event(self, kind: str, payload: dict[str, Any]) -> None:
        speaker = str(payload.get("speaker", "")).strip()
        vote_event: dict[str, Any] = {
            "speaker": speaker or "system",
            "turn": str(payload.get("turn", self.turn)),
            "phase": str(payload.get("phase", self.phase)),
            "kind": kind,
            "message": str(payload.get("message", "")),
        }
        target_name = payload.get("target_name")
        if isinstance(target_name, str) and target_name.strip():
            vote_event["target_name"] = target_name.strip()
        self.live_vote_events.append(vote_event)

    def _update_speech_queue(self, kind: str, payload: dict[str, Any]) -> None:
        if kind != "speech_queue":
            return

        queue_turn = payload.get("turn")
        if isinstance(queue_turn, int) and self.speech_queue.queue_turn != queue_turn:
            self.speech_queue.queue_turn = queue_turn
            self.speech_queue.total_slots = 0

        speaker = str(payload.get("speaker", "")).strip()
        if speaker:
            self.speech_queue.active_speaker = speaker
        elif not self.speech_queue.queue:
            self.speech_queue.active_speaker = None

        total_slots = len(self.speech_queue.queue) + self.speech_queue.active_count
        self.speech_queue.total_slots = max(self.speech_queue.total_slots, total_slots)

    def handle_progress(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Apply one raw progress payload and return SSE events to stream."""
        kind = str(payload.get("kind", ""))
        message = str(payload.get("message", ""))
        speaker = str(payload.get("speaker", "")).strip()

        self._apply_baseline(payload)
        self._update_speech_queue(kind, payload)

        if kind in {"day_vote", "mafia_vote"}:
            self._track_vote_event(kind, payload)

        events: list[dict[str, Any]] = []

        if kind == "provider_retry":
            attempt = payload.get("attempt")
            max_attempts = payload.get("max_attempts")
            detail = payload.get("detail")
            retry_line = f"Retry {attempt}/{max_attempts}: {detail}"
            self.retries.append(retry_line)
            events.append({"type": "retry", "message": retry_line, "log": self.retries[-5:]})
            return events

        if kind not in _SQUELCHED_PROGRESS_KINDS:
            self.progress_step = min(self.progress_step + 1, 95)
            progress_text = progress_text_for_event(
                kind=kind,
                speaker=speaker,
                message=message,
            )
            events.append(
                {
                    "type": "progress",
                    "text": progress_text,
                    "percent": self.progress_step,
                }
            )

        if kind in _CHAT_EVENT_KINDS and should_render_chat_event(kind, message):
            chat_event = self._chat_event(kind, payload)
            if chat_event is not None:
                events.append(chat_event)

        events.append(self.state_event())
        return events
