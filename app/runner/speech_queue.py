from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SpeechQueue:
    items: list[int] = field(default_factory=list)

    def enqueue(self, player_id: int) -> None:
        self.items.append(player_id)

    def dequeue(self) -> int | None:
        if not self.items:
            return None
        return self.items.pop(0)

    def is_empty(self) -> bool:
        return len(self.items) == 0

    def __len__(self) -> int:
        return len(self.items)
