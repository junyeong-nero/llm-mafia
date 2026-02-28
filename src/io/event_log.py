from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
import uuid
from typing import Mapping

from src.engine.game_state import GameEvent, GameState


def build_output_dir(base_dir: Path | None = None) -> Path:
    root = base_dir if base_dir is not None else Path("logs")
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    for _ in range(5):
        suffix = uuid.uuid4().hex[:6]
        target = root / f"{stamp}-{suffix}"
        try:
            target.mkdir(parents=True, exist_ok=False)
            return target
        except FileExistsError:
            continue
    raise RuntimeError("failed to create unique output directory")


def write_events_jsonl(events: list[GameEvent], output_dir: Path) -> Path:
    path = output_dir / "events.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            payload = {
                "turn": event.turn,
                "phase": event.phase.value,
                "speaker": event.speaker,
                "kind": event.kind,
                "content": event.content,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def write_summary_json(summary: Mapping[str, object], output_dir: Path) -> Path:
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def state_to_dict(state: GameState) -> dict[str, object]:
    return {
        "turn": state.turn,
        "phase": state.phase.value,
        "winner": state.winner,
        "players": [asdict(player) for player in state.players],
    }
