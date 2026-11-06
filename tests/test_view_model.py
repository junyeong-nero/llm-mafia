from __future__ import annotations

from src.web.view_model import (
    AGENT_AVATARS,
    DashboardState,
    chat_role_and_avatar,
    cycle_text,
    extract_mafia_chat_message,
    extract_vote_target_name,
    format_chat_entry,
    normalize_vote_result_message,
    progress_text_for_event,
    should_render_chat_event,
    vote_summary_rows_for_chat_event,
)


def test_should_render_chat_event_hides_internal_kinds() -> None:
    assert should_render_chat_event("speech", "I trust Alice.")
    assert not should_render_chat_event("day_vote", '{"target_name": "Alice"}')
    assert not should_render_chat_event("speak_request", "requested")
    assert not should_render_chat_event("speech_queue", "queued")
    assert not should_render_chat_event("agent_thinking", "NOW speaking: Alice")
    assert not should_render_chat_event("speech", "follow-up requests processed after all speeches")


def test_extract_vote_target_name_parses_json_payload() -> None:
    assert extract_vote_target_name('{"target_name": "Alice"}') == "Alice"
    assert extract_vote_target_name('{"target_name": "  "}') is None
    assert extract_vote_target_name("not json") is None
    assert extract_vote_target_name('{"other": "Alice"}') is None


def test_extract_mafia_chat_message_prefers_chat_line() -> None:
    raw = "CHAT: vote for Bob\nVOTE_JSON: {\"target_name\": \"Bob\"}\n"
    assert extract_mafia_chat_message(raw) == "vote for Bob"

    fallback = "plain thought line\nsecond line"
    assert extract_mafia_chat_message(fallback) == "plain thought line\nsecond line"

    assert extract_mafia_chat_message('VOTE_JSON: {"target_name": "Bob"}') == (
        'VOTE_JSON: {"target_name": "Bob"}'
    )


def test_format_chat_entry_formats_votes_and_mafia_chat() -> None:
    speaker, message = format_chat_entry(
        kind="mafia_chat",
        speaker="Wolf",
        message="CHAT: it is clear",
    )
    assert speaker == "Wolf"
    assert message == "it is clear"

    formatted = format_chat_entry(kind="day_vote", speaker="Alice", message='{"target_name": "Bob"}')
    assert formatted == ("system", "Day vote: Alice -> Bob")
    assert format_chat_entry(kind="day_vote", speaker="Alice", message="no json") == ("system", "no json")
    assert format_chat_entry(kind="speech", speaker="Alice", message="hello") == ("Alice", "hello")


def test_chat_role_and_avatar_is_deterministic() -> None:
    assert chat_role_and_avatar("system") == ("system", "terminal")
    assert chat_role_and_avatar("system", "night_result") == ("system", "moon")
    assert chat_role_and_avatar("system", "vote_result") == ("system", "gavel")

    role, avatar = chat_role_and_avatar("Alice")
    assert role == "agent"
    assert chat_role_and_avatar("Alice") == (role, avatar)
    assert avatar in {"bot", "brain", "cpu", "sparkles", "orbit", "flask-conical", "satellite", "atom"}


def test_chat_avatars_contain_no_emoji() -> None:
    assert all(name.isascii() for name in AGENT_AVATARS)


def test_progress_text_for_event_uses_kind_and_speaker() -> None:
    assert progress_text_for_event(kind="speech", speaker="Alice", message="hi") == (
        "Streaming live chat from Alice"
    )
    assert progress_text_for_event(kind="day_vote", speaker="Alice", message="") == (
        "Alice submitted a day vote"
    )
    assert progress_text_for_event(kind="night_result", speaker="", message="Alice died") == "Alice died"
    assert progress_text_for_event(kind="", speaker="", message="") == "Match in progress"


def test_normalize_vote_result_message_strips_night_target() -> None:
    assert normalize_vote_result_message(
        vote_type="night",
        message="Night result: Alice removed mafia_target=Bob",
    ) == "Night result: Alice removed"
    kept = "Day result: Alice removed"
    assert normalize_vote_result_message(vote_type="day", message=kept) == kept


def test_vote_summary_rows_aggregate_live_vote_events_by_turn() -> None:
    live_vote_events = [
        {"speaker": "Alice", "turn": "1", "phase": "vote", "kind": "day_vote", "target_name": "Bob"},
        {"speaker": "Carol", "turn": "1", "phase": "vote", "kind": "day_vote", "target_name": "Bob"},
        {"speaker": "Alice", "turn": "1", "phase": "night", "kind": "mafia_vote", "target_name": "Drew"},
        {"speaker": "Drew", "turn": "2", "phase": "vote", "kind": "day_vote", "target_name": "Nobody"},
    ]

    day_rows = vote_summary_rows_for_chat_event(live_vote_events, kind="vote_result", turn="1")
    assert day_rows == [
        {"type": "day", "voter": "Alice", "target": "Bob"},
        {"type": "day", "voter": "Carol", "target": "Bob"},
    ]

    night_rows = vote_summary_rows_for_chat_event(live_vote_events, kind="night_result", turn="1")
    assert night_rows == [{"type": "night", "voter": "Alice", "target": "Drew"}]

    assert vote_summary_rows_for_chat_event(live_vote_events, kind="vote_result", turn="3") is None
    assert vote_summary_rows_for_chat_event(live_vote_events, kind="speech", turn="1") is None


def test_vote_summary_rows_fall_back_to_message_parsing() -> None:
    live_vote_events = [
        {
            "speaker": "Alice",
            "turn": "2",
            "phase": "vote",
            "kind": "day_vote",
            "message": "voted to eliminate Bob well before.",
        }
    ]
    rows = vote_summary_rows_for_chat_event(live_vote_events, kind="vote_result", turn="2")
    assert rows == [{"type": "day", "voter": "Alice", "target": "Bob well before"}]


def test_dashboard_state_state_event_describes_cycle() -> None:
    state = DashboardState()
    assert state.state_event()["current"] == "Preparing game"
    assert state.state_event()["speech_queue"]["queue"] == []

    assert cycle_text(turn=3, phase="day") == "Day 3"
    assert cycle_text(turn=3, phase="vote") == "Day 3 (Voting)"
    assert cycle_text(turn=3, phase="night") == "Night 3"
    assert cycle_text(turn=3, phase="end") == "Game over"


def test_dashboard_state_route_votes_into_chat_summaries() -> None:
    state = DashboardState()

    payload_vote_one = {"kind": "day_vote", "turn": 3, "phase": "vote", "speaker": "Alice", "message": '{"target_name": "Bob"}'}
    handle_events = state.handle_progress(dict(payload_vote_one))

    kinds = [event["type"] for event in handle_events]
    assert "chat" not in kinds
    assert any(event["type"] == "state" for event in handle_events)

    result_payload = {
        "kind": "vote_result",
        "turn": 3,
        "phase": "vote",
        "speaker": "system",
        "message": "Day result: 2/2 voted for Bob.",
    }
    chat_events = [event for event in state.handle_progress(dict(result_payload)) if event["type"] == "chat"]

    assert len(chat_events) == 1
    chat_event = chat_events[0]
    assert chat_event["speaker"] == "system"
    assert chat_event["vote_summary_rows"] == [
        {"type": "day", "voter": "Alice", "target": "Bob"},
    ]
    assert chat_event["vote_result_text"] == "Day result: 2/2 voted for Bob."


def test_dashboard_state_provider_retry_does_not_advance_progress() -> None:
    state = DashboardState()
    before_step = state.progress_step

    events = state.handle_progress(
        {"kind": "provider_retry", "attempt": 2, "max_attempts": 5, "detail": "upstream 5xx"}
    )

    assert events[0]["type"] == "retry"
    assert events[0]["message"].startswith("Retry 2/5")
    assert state.progress_step == before_step
    assert state.retries == ["Retry 2/5: upstream 5xx"]


def test_dashboard_state_speech_queue_tracks_active_and_totals() -> None:
    state = DashboardState()

    state.handle_progress(
        {"kind": "speech_queue", "turn": 2, "phase": "day", "message": "", "speaker": "Alice", "speech_queue": ["Carol", "Drew"]}
    )
    snapshot = state.speech_queue.snapshot()
    assert snapshot["active_speaker"] == "Alice"
    assert snapshot["queue"] == ["Carol", "Drew"]
    assert snapshot["total_slots"] == 3

    state.handle_progress(
        {"kind": "speech_queue", "turn": 2, "phase": "day", "message": "", "speaker": "", "speech_queue": []}
    )
    finished = state.speech_queue.snapshot()
    assert finished["active_speaker"] is None
    assert finished["total_slots"] == 3
    assert finished["completed_slots"] == 3


def test_dashboard_state_progress_capped_at_95() -> None:
    state = DashboardState()
    for _ in range(200):
        state.handle_progress({"kind": "speech", "turn": 2, "phase": "day", "speaker": "Alice", "message": "hi"})
    assert state.progress_step == 95


def test_dashboard_state_players_status_filters_malformed_entries() -> None:
    state = DashboardState()
    state.handle_progress(
        {
            "kind": "setup",
            "turn": 1,
            "phase": "setup",
            "players_status": [
                {"name": "Alice", "model_name": "m1", "role": "mafia", "alive": True},
                {"name": 42, "model_name": "m1", "role": "mafia", "alive": True},
                {"name": "Bob", "model_name": "m1", "role": "citizen", "alive": True, "extra": True},
            ],
        }
    )
    assert state.players_status == [
        {
            "name": "Alice",
            "model_name": "m1",
            "role": "mafia",
            "alive": True,
        },
        {
            "name": "Bob",
            "model_name": "m1",
            "role": "citizen",
            "alive": True,
        },
    ]
