"""FastAPI application serving the llm-mafia web dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.web.match_session import (
    MatchAlreadyRunningError,
    MatchRequest,
    MatchSession,
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"

RUNNER_OPTIONS = Literal["legacy", "graph"]


class MatchStartRequest(BaseModel):
    config_path: str
    seed: int | None = None
    max_rounds: int = Field(default=10, ge=1)
    runner: RUNNER_OPTIONS = "graph"


def _sse_line(event: dict[str, object]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _sse_stream(session: MatchSession) -> Iterator[str]:
    yield _sse_line(session.snapshot())
    for event in session.stream_events():
        yield _sse_line(event)


def create_app(session: MatchSession | None = None) -> FastAPI:
    app_session = session or MatchSession()

    app = FastAPI(title="llm-mafia", docs_url=None, redoc_url=None)

    @app.post("/api/match/start")
    def start_match(body: MatchStartRequest) -> dict[str, object]:
        request = MatchRequest(
            config_path=body.config_path,
            seed=body.seed,
            max_rounds=body.max_rounds,
            runner=body.runner,
        )
        try:
            return app_session.start(request)
        except MatchAlreadyRunningError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Configuration error: {exc}",
            ) from exc

    @app.get("/api/match/state")
    def match_state() -> dict[str, object]:
        return app_session.snapshot()

    @app.get("/api/match/stream")
    def match_stream() -> StreamingResponse:
        return StreamingResponse(
            _sse_stream(app_session),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    return app


app = create_app()
