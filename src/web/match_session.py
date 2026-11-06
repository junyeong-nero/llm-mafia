"""Background match session for the web dashboard.

Runs ``run_match`` in a worker thread, funnels progress payloads through the
view model, and exposes them as an event stream plus a synchronizable
snapshot so reconnecting browsers can resume.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Literal, cast

from src.config import AppConfig, load_config
from src.runner.match_runner import run_match
from src.web.view_model import DashboardState

if TYPE_CHECKING:
    from src.runner.single_match import MatchResult

RunnerType = Literal["legacy", "graph"]
SessionStatus = Literal["idle", "running", "complete", "error"]


class MatchAlreadyRunningError(RuntimeError):
    """Raised when a match start is requested while another match runs."""


@dataclass(frozen=True)
class MatchRequest:
    config_path: str
    seed: int | None
    max_rounds: int
    runner: RunnerType


class EventHub:
    """Thread-safe publish/subscribe queue for SSE streaming."""

    def __init__(self) -> None:
        self._events: list[dict[str, object]] = []
        self._condition = threading.Condition()
        self._closed = False

    def publish(self, event: dict[str, object]) -> None:
        with self._condition:
            self._events.append(event)
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def subscribe(self) -> Iterator[dict[str, object]]:
        """Yield buffered and future events until the hub is closed and drained."""
        index = 0
        while True:
            with self._condition:
                while index >= len(self._events) and not self._closed:
                    self._condition.wait()
                if index >= len(self._events) and self._closed:
                    return
                event = self._events[index]
            index += 1
            yield event


class MatchSession:
    """Runs one match at a time and tracks dashboard state for it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._dashboard = DashboardState()
        self._hub = EventHub()
        self._thread: threading.Thread | None = None
        self._status: SessionStatus = "idle"
        self._result: MatchResult | None = None
        self._error_message: str | None = None
        self._request: MatchRequest | None = None

    @property
    def status(self) -> SessionStatus:
        with self._lock:
            return self._status

    def start(self, request: MatchRequest) -> dict[str, object]:
        """Validate config, then run the match in a worker thread."""
        config = load_config(Path(request.config_path))

        with self._lock:
            if self._status == "running":
                raise MatchAlreadyRunningError(
                    "A match is already running; wait for it to finish."
                )
            self._dashboard.reset()
            self._hub = EventHub()
            self._status = "running"
            self._result = None
            self._error_message = None
            self._request = request
            self._hub.publish(
                {"type": "progress", "text": "Preparing match", "percent": 0}
            )
            self._hub.publish(self._dashboard.state_event())
            self._thread = threading.Thread(
                target=self._run,
                args=(config, request),
                daemon=True,
                name="llm-mafia-match",
            )
            self._thread.start()
        return self.snapshot()

    def _run(self, config: AppConfig, request: MatchRequest) -> None:
        try:
            result = run_match(
                config,
                seed=request.seed,
                max_rounds=request.max_rounds,
                progress_callback=self._on_progress,
                runner=cast(RunnerType, request.runner),
            )
        except Exception as exc:  # noqa: BLE001 - streamed to browser
            with self._lock:
                self._status = "error"
                self._error_message = str(exc)
            self._hub.publish(
                {
                    "type": "match_error",
                    "message": str(exc),
                    "retries": list(self._dashboard.retries),
                }
            )
        else:
            with self._lock:
                self._dashboard.to_view(result)
                self._result = result
                self._status = "complete"
                self._hub.publish(
                    {
                        "type": "match_complete",
                        **self._result_snapshot_locked(),
                    }
                )
        finally:
            self._hub.close()

    def _on_progress(self, payload: dict[str, object]) -> None:
        for event in self._dashboard.handle_progress(dict(payload)):
            self._hub.publish(event)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, object]:
        snapshot: dict[str, object] = {
            "status": self._status,
            "error_message": self._error_message,
            "view": self._dashboard.state_event(),
        }
        if self._request is not None:
            snapshot["request"] = {
                "config_path": self._request.config_path,
                "seed": self._request.seed,
                "max_rounds": self._request.max_rounds,
                "runner": self._request.runner,
            }
        if self._result is not None:
            snapshot["result"] = self._result_snapshot_locked()
        return snapshot

    def _result_snapshot_locked(self) -> dict[str, object]:
        assert self._result is not None
        fields = self._dashboard.result_snapshot(self._result)
        return {
            "current": fields["current"],
            "players_status": fields["players_status"],
            "output_dir": fields["output_dir"],
            "events_path": fields["events_path"],
            "summary_path": fields["summary_path"],
        }

    def stream_events(self) -> Iterator[dict[str, object]]:
        """Subscribe to the current match's event stream."""
        return self._hub.subscribe()

    def wait(self) -> None:
        """Block until the worker thread finishes (mainly for tests)."""
        thread = self._thread
        if thread is not None:
            thread.join()
