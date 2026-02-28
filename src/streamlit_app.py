from __future__ import annotations

import html
import hashlib
from pathlib import Path
from typing import Any

import streamlit as st

from src.runner.single_match import MatchResult, run_single_match
from src.config import AppConfig, load_config


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
        text = f"{turn}번째 낮"
    elif phase == "night":
        text = f"{turn}번째 밤"
    elif phase == "vote":
        text = f"{turn}번째 낮 (투표)"
    elif phase == "setup":
        text = "게임 준비 중"
    else:
        text = "게임 종료"

    st.subheader("진행 상태")
    st.metric("현재", text)


def _render_feed(result: MatchResult | None) -> None:
    st.subheader("LLM Chat Replay")
    if result is None and not st.session_state.live_chat_events:
        return

    if result is None:
        _render_live_chat(st.container(height=620), st.session_state.live_chat_events, "Live replay")
        return

    with st.container(height=620):
        for event in result.events:
            if not _should_render_chat_event(str(event.kind), str(event.content)):
                continue
            role, avatar = _chat_role_and_avatar(event.speaker)
            with st.chat_message(role, avatar=avatar):
                st.markdown(f"**{event.speaker}**")
                st.caption(f"Turn {event.turn} | {event.phase.value} | {event.kind}")
                st.markdown(event.content)


def _render_player_status(result: MatchResult | None) -> None:
    st.subheader("Player Status")
    if result is not None:
        players = _player_status_from_result(result)
    else:
        players = st.session_state.player_status

    if not players:
        st.caption("Run a match to see alive/dead status.")
        return

    columns = st.columns(4)
    for idx, player in enumerate(players):
        is_alive = bool(player.get("alive", True))
        icon = "🔵" if is_alive else "🔴"
        label = "Alive" if is_alive else "Dead"
        name = str(player.get("name", f"Player{idx + 1}"))
        role = str(player.get("role", "unknown"))
        with columns[idx % 4]:
            st.markdown(f"**{name}**")
            st.caption(f"{icon} {label} | role: {role}")


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
        with st.container(height=480):
            for item in events[-30:]:
                kind = str(item.get("kind", "progress"))
                message = str(item.get("message", ""))
                if not _should_render_chat_event(kind, message):
                    continue
                speaker = item.get("speaker", "system")
                role, avatar = _chat_role_and_avatar(speaker)
                with st.chat_message(role, avatar=avatar):
                    st.markdown(f"**{speaker}**")
                    turn = item.get("turn", "?")
                    phase = item.get("phase", "?")
                    st.caption(f"Turn {turn} | {phase} | {kind}")
                    st.markdown(message)


def _should_render_chat_event(kind: str, message: str) -> bool:
    normalized_kind = kind.strip().lower()
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


def _inject_sidebar_styles() -> None:
    st.markdown(
        """
        <style>
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
    live_chat_placeholder: Any,
    cycle_placeholder: Any,
    player_status_placeholder: Any,
    speech_queue_placeholder: Any,
) -> None:
    st.subheader("Match Setup")
    st.text_input("Config Path", key="config_path")
    run_button = st.button("Run Full Match", use_container_width=True)
    refresh_button = st.button("Refresh View", use_container_width=True)

    progress_placeholder = st.empty()
    retry_placeholder = st.empty()

    if run_button:
        st.session_state.retry_updates = []
        st.session_state.live_chat_events = []
        st.session_state.speech_queue = []
        st.session_state.active_speaker = None
        st.session_state.speech_queue_total = 0
        st.session_state.speech_queue_turn = None
        progress_bar = progress_placeholder.progress(0, text="Preparing match")
        chat_event_kinds = {"setup", "night_result", "vote_result", "game_end", "speech", "mafia_chat"}

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
            progress_bar.progress(progress_step["value"], text=message or "Match in progress")

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
                    section_title="LLM Chat Replay",
                )

        with st.status("Running match", state="running") as status:
            try:
                config = _load_app_config()
                result = run_single_match(config, progress_callback=on_progress)
                st.session_state.match_result = result
                _set_current_view_state(result)
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
    st.title("LLM Mafia Dashboard")
    _initialize_state()
    live_chat_placeholder = st.empty()
    with st.sidebar:
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
            live_chat_placeholder,
            cycle_placeholder,
            player_status_placeholder,
            speech_queue_placeholder,
        )
    _render_feed(st.session_state.match_result)


if __name__ == "__main__":
    main()
