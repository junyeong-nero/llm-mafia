from __future__ import annotations

import html
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import AppConfig, load_config
from src.engine.game_state import GameEvent
from src.runner.match_runner import MatchResult, run_match


def _load_app_config() -> AppConfig:
    config_path = Path(st.session_state.get("config_path", "config.yaml"))
    return load_config(config_path)


def _initialize_state() -> None:
    if "config_path" not in st.session_state:
        st.session_state.config_path = "config.yaml"
    if "match_result" not in st.session_state:
        st.session_state.match_result = None
    if "turn" not in st.session_state:
        st.session_state.turn = 1
    if "phase" not in st.session_state:
        st.session_state.phase = "setup"
    if "retry_updates" not in st.session_state:
        st.session_state.retry_updates = []
    if "live_chat_events" not in st.session_state:
        st.session_state.live_chat_events = []
    if "player_status" not in st.session_state:
        st.session_state.player_status = []
    if "speech_queue" not in st.session_state:
        st.session_state.speech_queue = []
    if "active_speaker" not in st.session_state:
        st.session_state.active_speaker = None
    if "speech_queue_total" not in st.session_state:
        st.session_state.speech_queue_total = 0
    if "speech_queue_turn" not in st.session_state:
        st.session_state.speech_queue_turn = None
    if "live_vote_events" not in st.session_state:
        st.session_state.live_vote_events = []


def _set_current_view_state(result: MatchResult) -> None:
    if not result.events:
        return
    last = result.events[-1]
    st.session_state.turn = last.turn
    st.session_state.phase = last.phase.value
    st.session_state.player_status = _player_status_from_result(result)
    st.session_state.speech_queue = []
    st.session_state.active_speaker = None
    st.session_state.speech_queue_total = 0
    st.session_state.speech_queue_turn = None


def _player_status_from_result(result: MatchResult) -> list[dict[str, object]]:
    return [
        {
            "name": player.name,
            "model_name": player.model_name,
            "role": player.role.value,
            "alive": player.alive,
        }
        for player in result.state.players
    ]


def _render_cycle_indicator() -> None:
    phase = st.session_state.phase
    turn = st.session_state.turn

    if phase == "day":
        text = f"Day {turn}"
    elif phase == "night":
        text = f"Night {turn}"
    elif phase == "vote":
        text = f"Day {turn} (Voting)"
    elif phase == "setup":
        text = "Preparing game"
    else:
        text = "Game over"

    st.subheader("Game Progress")
    st.metric("Current", text)


def _render_feed(result: MatchResult | None) -> None:
    st.subheader("Live chat stream")
    if result is None and not st.session_state.live_chat_events:
        return

    if result is None:
        _render_live_chat(st.container(), st.session_state.live_chat_events, "Live chat stream")
        return

    with st.container():
        for event in result.events:
            if not _should_render_chat_event(str(event.kind), str(event.content)):
                continue
            vote_summary_rows = _vote_summary_rows_for_result_events(
                result.events,
                kind=str(event.kind),
                turn=event.turn,
            )
            _render_chat_event(
                kind=str(event.kind),
                speaker=str(event.speaker),
                message=str(event.content),
                turn=event.turn,
                phase=str(event.phase.value),
                vote_summary_rows=vote_summary_rows,
            )


def _render_player_status(result: MatchResult | None) -> None:
    st.subheader("Player Status")
    if result is not None:
        players = _player_status_from_result(result)
    else:
        players = st.session_state.player_status

    if not players:
        st.caption("Run a match to see alive/dead status.")
        return

    table_rows: list[dict[str, str]] = []
    for idx, player in enumerate(players, start=1):
        is_alive = bool(player.get("alive", True))
        status = "🟢 live" if is_alive else "🔴 dead"
        name = str(player.get("name", f"Player{idx}"))
        model_name = str(player.get("model_name", "unknown"))
        role = str(player.get("role", "unknown"))
        table_rows.append(
            {
                "name": name,
                "model": model_name,
                "role": role,
                "status": status,
            }
        )

    st.dataframe(table_rows, width="stretch", hide_index=True)


def _render_speech_queue() -> None:
    st.subheader("Speech Queue")
    queue = st.session_state.speech_queue
    active_speaker = st.session_state.active_speaker
    active_count = 1 if active_speaker else 0
    total_slots = max(int(st.session_state.speech_queue_total), len(queue) + active_count)
    completed_slots = max(total_slots - len(queue) - active_count, 0)

    if total_slots > 0:
        completion_ratio = completed_slots / total_slots
        st.progress(completion_ratio, text=f"Completed {completed_slots}/{total_slots}")

    cards: list[str] = []
    if active_speaker:
        cards.append(
            "<div class='speech-queue-card speech-queue-card--active'>"
            "<span class='speech-queue-index'>NOW</span>"
            f"<span class='speech-queue-name'>{html.escape(active_speaker)}</span>"
            "</div>"
        )

    for index, speaker in enumerate(queue, start=1):
        cards.append(
            "<div class='speech-queue-card speech-queue-card--pending'>"
            f"<span class='speech-queue-index'>{index}</span>"
            f"<span class='speech-queue-name'>{html.escape(speaker)}</span>"
            "</div>"
        )

    if cards:
        st.markdown(f"<div class='speech-queue-stack'>{''.join(cards)}</div>", unsafe_allow_html=True)
        return

    if total_slots > 0:
        st.caption("All queued speeches finished.")
        return

    st.caption("No pending speakers.")


def _render_live_chat(
    container: Any,
    events: list[dict[str, str]],
    title: str,
    section_title: str | None = None,
) -> None:
    with container:
        if section_title:
            st.subheader(section_title)
        st.caption(title)
        with st.container():
            for item in events[-30:]:
                kind = str(item.get("kind", "progress"))
                message = str(item.get("message", ""))
                if not _should_render_chat_event(kind, message):
                    continue
                vote_summary_rows = _vote_summary_rows_for_live_events(
                    st.session_state.live_vote_events,
                    kind=kind,
                    turn=item.get("turn", "?"),
                )
                _render_chat_event(
                    kind=kind,
                    speaker=str(item.get("speaker", "system")),
                    message=message,
                    turn=item.get("turn", "?"),
                    phase=str(item.get("phase", "?")),
                    vote_summary_rows=vote_summary_rows,
                )


def _render_chat_event(
    *,
    kind: str,
    speaker: str,
    message: str,
    turn: object,
    phase: str,
    vote_summary_rows: list[dict[str, str]] | None = None,
) -> None:
    formatted_speaker, formatted_message = _format_chat_entry(
        kind=kind,
        speaker=speaker,
        message=message,
    )
    role, avatar = _chat_role_and_avatar(formatted_speaker)
    with st.chat_message(role, avatar=avatar):
        st.markdown(f"**{formatted_speaker}**")
        st.caption(f"Turn {turn} | {phase} | {kind}")
        if vote_summary_rows:
            st.table(vote_summary_rows)
            normalized_kind = kind.strip().lower()
            if normalized_kind in {"vote_result", "night_result"}:
                vote_type = "day" if normalized_kind == "vote_result" else "night"
                result_text = _normalize_vote_result_message(vote_type=vote_type, message=formatted_message)
                if result_text:
                    st.markdown(f"Result: {result_text}")
            return
        st.markdown(formatted_message)


def _vote_summary_rows_for_live_events(
    events: list[dict[str, str]],
    *,
    kind: str,
    turn: object,
) -> list[dict[str, str]] | None:
    normalized_kind = kind.strip().lower()
    turn_token = str(turn).strip()
    if normalized_kind == "vote_result":
        rows = _collect_vote_rows_from_live_events(
            events,
            vote_kind="day_vote",
            vote_phase="vote",
            vote_type="day",
            turn_token=turn_token,
        )
        return _append_vote_result_row(rows)
    if normalized_kind == "night_result":
        rows = _collect_vote_rows_from_live_events(
            events,
            vote_kind="mafia_vote",
            vote_phase="night",
            vote_type="night",
            turn_token=turn_token,
        )
        return _append_vote_result_row(rows)
    return None


def _collect_vote_rows_from_live_events(
    events: list[dict[str, str]],
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

        target_name = _extract_target_name_from_live_vote_event(item)
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


def _vote_summary_rows_for_result_events(
    events: list[GameEvent],
    *,
    kind: str,
    turn: int,
) -> list[dict[str, str]] | None:
    normalized_kind = kind.strip().lower()
    if normalized_kind == "vote_result":
        rows = _collect_vote_rows_from_result_events(
            events,
            vote_kind="day_vote",
            vote_phase="vote",
            vote_type="day",
            turn=turn,
        )
        return _append_vote_result_row(rows)
    if normalized_kind == "night_result":
        rows = _collect_vote_rows_from_result_events(
            events,
            vote_kind="mafia_vote",
            vote_phase="night",
            vote_type="night",
            turn=turn,
        )
        return _append_vote_result_row(rows)
    return None


def _collect_vote_rows_from_result_events(
    events: list[GameEvent],
    *,
    vote_kind: str,
    vote_phase: str,
    vote_type: str,
    turn: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for event in events:
        event_kind = str(event.kind).strip().lower()
        event_phase = str(event.phase.value).strip().lower()
        if event_kind != vote_kind or event_phase != vote_phase or event.turn != turn:
            continue

        target_name = _extract_vote_target_name(str(event.content))
        if target_name is None:
            continue

        voter_name = str(event.speaker).strip() or "unknown"
        rows.append(
            {
                "type": vote_type,
                "voter": voter_name,
                "target": target_name,
            }
        )

    return rows


def _extract_target_name_from_live_vote_event(item: dict[str, str]) -> str | None:
    raw_target_name = item.get("target_name")
    if isinstance(raw_target_name, str) and raw_target_name.strip():
        return raw_target_name.strip()

    message = str(item.get("message", ""))
    target_name = _extract_vote_target_name(message)
    if target_name is not None:
        return target_name

    sentence_match = re.search(r"voted\s+to\s+eliminate\s+(.+?)(?:\.|$)", message, flags=re.IGNORECASE)
    if sentence_match is None:
        return None

    candidate = sentence_match.group(1).strip()
    if not candidate:
        return None
    return candidate


def _append_vote_result_row(
    rows: list[dict[str, str]],
) -> list[dict[str, str]] | None:
    if not rows:
        return None
    return rows


def _normalize_vote_result_message(*, vote_type: str, message: str) -> str:
    normalized = message.strip()
    if vote_type == "night" and " mafia_target=" in normalized:
        normalized = normalized.split(" mafia_target=", maxsplit=1)[0].strip()
    return normalized


def _format_chat_entry(*, kind: str, speaker: str, message: str) -> tuple[str, str]:
    normalized_kind = kind.strip().lower()
    if normalized_kind == "mafia_chat":
        return speaker, _extract_mafia_chat_message(message)

    if normalized_kind in {"day_vote", "mafia_vote"}:
        vote_label = "Day vote" if normalized_kind == "day_vote" else "Night vote"
        target_name = _extract_vote_target_name(message)
        if target_name and speaker.strip():
            return "system", f"{vote_label}: {speaker} -> {target_name}"
        return "system", message

    return speaker, message


def _extract_vote_target_name(message: str) -> str | None:
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


def _extract_mafia_chat_message(message: str) -> str:
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


def _should_render_chat_event(kind: str, message: str) -> bool:
    normalized_kind = kind.strip().lower()
    hidden_kinds = {
        "day_vote",
        "mafia_vote",
        "day_vote_invalid",
        "mafia_vote_invalid",
        "speak_request",
        "speak_request_reason",
    }
    if normalized_kind in hidden_kinds:
        return False

    if "speech_queue" in normalized_kind:
        return False

    normalized_message = message.strip().lower()
    hidden_prefixes = (
        "now speaking:",
        "follow-up requests processed after ",
    )
    return not normalized_message.startswith(hidden_prefixes)


def _chat_role_and_avatar(speaker: str) -> tuple[str, str]:
    if speaker == "system":
        return "user", "🛠️"

    avatars = ["🤖", "🧠", "🦊", "🦉", "🛰️", "🧪", "🛡️", "🐼"]
    hashed = int(hashlib.sha1(speaker.encode("utf-8")).hexdigest(), 16)
    return "assistant", avatars[hashed % len(avatars)]


def _progress_text_for_event(*, kind: str, speaker: str, message: str) -> str:
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

    if normalized_kind == "speech_queue":
        return "Updating speech queue"

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


def _inject_sidebar_styles() -> None:
    st.markdown(
        """
        <style>
        section.main > div.block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f7f9fc 0%, #eef2f8 100%);
            border-right: 1px solid #d7dde8;
        }

        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.65rem;
        }

        section[data-testid="stSidebar"] .stSubheader {
            font-size: 1.02rem;
            font-weight: 650;
            letter-spacing: -0.01em;
            color: #1f2a44;
            margin-top: 0.3rem;
            margin-bottom: 0.2rem;
        }

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
            font-size: 0.93rem;
            line-height: 1.45;
            color: #2d3853;
        }

        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            font-size: 0.83rem;
            line-height: 1.35;
            color: #5f6b85;
        }

        section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
            font-size: 0.86rem;
            font-weight: 600;
            color: #4a5873;
        }

        section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
            font-size: 1.06rem;
            font-weight: 700;
            color: #1f2a44;
        }

        section[data-testid="stSidebar"] .stTextInput input {
            font-size: 0.92rem;
            border-radius: 0.5rem;
        }

        section[data-testid="stSidebar"] .stButton > button {
            font-size: 0.91rem;
            font-weight: 600;
            min-height: 2.45rem;
            border-radius: 0.55rem;
            border: 1px solid #c6cfdf;
            background: #f3f6fc;
            color: #1f2a44;
        }

        section[data-testid="stSidebar"] .stButton > button:hover {
            border-color: #9facc5;
            background: #e8edf7;
        }

        section[data-testid="stSidebar"] .speech-queue-stack {
            display: grid;
            gap: 0.42rem;
            margin-top: 0.1rem;
        }

        section[data-testid="stSidebar"] .speech-queue-card {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            padding: 0.44rem 0.58rem;
            border-radius: 0.56rem;
            border: 1px solid #cdd6e5;
            background: #f7f9fd;
        }

        section[data-testid="stSidebar"] .speech-queue-card--active {
            border-color: #3b6fc7;
            background: #e8f0ff;
        }

        section[data-testid="stSidebar"] .speech-queue-card--pending {
            border-color: #cfd7e8;
            background: #f7f9fd;
        }

        section[data-testid="stSidebar"] .speech-queue-index {
            min-width: 2.6rem;
            padding: 0.12rem 0.36rem;
            border-radius: 0.4rem;
            background: #dbe5f6;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            text-align: center;
            color: #335089;
        }

        section[data-testid="stSidebar"] .speech-queue-card--active .speech-queue-index {
            background: #3b6fc7;
            color: #ffffff;
        }

        section[data-testid="stSidebar"] .speech-queue-name {
            font-size: 0.88rem;
            font-weight: 600;
            color: #1f2a44;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_controls(
    result: MatchResult | None,
    progress_placeholder: Any,
    live_chat_placeholder: Any,
    cycle_placeholder: Any,
    player_status_placeholder: Any,
    speech_queue_placeholder: Any,
) -> None:
    st.subheader("Match Setup")
    st.text_input("Config Path", key="config_path")
    run_button = st.button("Run Full Match", width="stretch")
    refresh_button = st.button("Refresh View", width="stretch")

    retry_placeholder = st.empty()

    if run_button:
        st.session_state.retry_updates = []
        st.session_state.live_chat_events = []
        st.session_state.speech_queue = []
        st.session_state.live_vote_events = []
        st.session_state.active_speaker = None
        st.session_state.speech_queue_total = 0
        st.session_state.speech_queue_turn = None
        progress_bar = progress_placeholder.progress(0, text="Preparing match")
        chat_event_kinds = {
            "setup",
            "night_result",
            "vote_result",
            "game_end",
            "speech",
            "mafia_chat",
            "day_vote",
            "mafia_vote",
        }

        progress_step = {"value": 0}

        def on_progress(payload: dict[str, object]) -> None:
            kind = str(payload.get("kind", ""))
            message = str(payload.get("message", ""))
            speaker = str(payload.get("speaker", "")).strip()

            turn = payload.get("turn")
            if isinstance(turn, int):
                st.session_state.turn = turn
            phase = payload.get("phase")
            if isinstance(phase, str) and phase:
                st.session_state.phase = phase

            player_status = payload.get("players_status")
            if isinstance(player_status, list):
                st.session_state.player_status = [
                    item
                    for item in player_status
                    if isinstance(item, dict)
                    and isinstance(item.get("name"), str)
                    and isinstance(item.get("model_name"), str)
                    and isinstance(item.get("role"), str)
                    and isinstance(item.get("alive"), bool)
                ]

            speech_queue = payload.get("speech_queue")
            if isinstance(speech_queue, list):
                st.session_state.speech_queue = [
                    item
                    for item in speech_queue
                    if isinstance(item, str)
                ]

            if kind in {"day_vote", "mafia_vote"}:
                vote_event = {
                    "speaker": speaker or "system",
                    "turn": str(payload.get("turn", st.session_state.turn)),
                    "phase": str(payload.get("phase", st.session_state.phase)),
                    "kind": kind,
                    "message": message,
                }
                target_name = payload.get("target_name")
                if isinstance(target_name, str) and target_name.strip():
                    vote_event["target_name"] = target_name.strip()
                st.session_state.live_vote_events.append(vote_event)

            if kind == "speech_queue":
                queue_turn = payload.get("turn")
                if isinstance(queue_turn, int) and st.session_state.speech_queue_turn != queue_turn:
                    st.session_state.speech_queue_turn = queue_turn
                    st.session_state.speech_queue_total = 0

                if speaker:
                    st.session_state.active_speaker = speaker
                elif isinstance(speech_queue, list) and not st.session_state.speech_queue:
                    st.session_state.active_speaker = None

                active_count = 1 if st.session_state.active_speaker else 0
                total_slots = len(st.session_state.speech_queue) + active_count
                st.session_state.speech_queue_total = max(st.session_state.speech_queue_total, total_slots)

            with cycle_placeholder.container():
                _render_cycle_indicator()
            with player_status_placeholder.container():
                _render_player_status(None)
            with speech_queue_placeholder.container():
                _render_speech_queue()

            if kind == "provider_retry":
                attempt = payload.get("attempt")
                max_attempts = payload.get("max_attempts")
                detail = payload.get("detail")
                retry_line = f"Retry {attempt}/{max_attempts}: {detail}"
                st.session_state.retry_updates.append(retry_line)
                retry_text = "\n".join(f"- {line}" for line in st.session_state.retry_updates[-5:])
                retry_placeholder.markdown(f"**Provider Retries**\n{retry_text}")
                return

            progress_step["value"] = min(progress_step["value"] + 1, 95)
            progress_text = _progress_text_for_event(kind=kind, speaker=speaker, message=message)
            progress_bar.progress(progress_step["value"], text=progress_text)

            if kind in chat_event_kinds and _should_render_chat_event(kind, message):
                st.session_state.live_chat_events.append(
                    {
                        "speaker": speaker or "system",
                        "turn": str(payload.get("turn", st.session_state.turn)),
                        "phase": str(payload.get("phase", st.session_state.phase)),
                        "kind": kind or "progress",
                        "message": message or "Match in progress",
                    }
                )
                _render_live_chat(
                    live_chat_placeholder.container(),
                    st.session_state.live_chat_events,
                    "Live chat stream",
                )

        with st.status("Running match", state="running") as status:
            try:
                config = _load_app_config()
                result = run_match(config, progress_callback=on_progress)
                st.session_state.match_result = result
                _set_current_view_state(result)
                live_chat_placeholder.empty()
                with cycle_placeholder.container():
                    _render_cycle_indicator()
                with player_status_placeholder.container():
                    _render_player_status(result)
                with speech_queue_placeholder.container():
                    _render_speech_queue()
                progress_bar.progress(100, text="Match complete")
                status.update(label="Match complete", state="complete")
                st.caption(f"Logs: {result.output_dir}")
                st.caption(f"Events: {result.events_path}")
                st.caption(f"Summary: {result.summary_path}")
            except Exception as exc:
                status.update(label=f"Error: {exc}", state="error")
                progress_bar.progress(100, text="Match ended with error")
                retry_placeholder.markdown("**Provider Retries**\n- Stopped due to an error.")

    if refresh_button:
        result = st.session_state.match_result
        if result is not None:
            _set_current_view_state(result)
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="llm-mafia", layout="wide")
    _inject_sidebar_styles()
    _initialize_state()
    progress_placeholder = st.empty()
    live_chat_placeholder = st.empty()
    with st.sidebar:
        st.title("LLM Mafia Dashboard")
        cycle_placeholder = st.empty()
        with cycle_placeholder.container():
            _render_cycle_indicator()

        player_status_placeholder = st.empty()
        with player_status_placeholder.container():
            _render_player_status(st.session_state.match_result)

        speech_queue_placeholder = st.empty()
        with speech_queue_placeholder.container():
            _render_speech_queue()

        _render_controls(
            st.session_state.match_result,
            progress_placeholder,
            live_chat_placeholder,
            cycle_placeholder,
            player_status_placeholder,
            speech_queue_placeholder,
        )
    _render_feed(st.session_state.match_result)


if __name__ == "__main__":
    main()
