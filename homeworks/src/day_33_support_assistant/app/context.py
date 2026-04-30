from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UserContext:
    user_id: str
    name: str
    plan: str
    locale: str


@dataclass(frozen=True)
class TicketContext:
    ticket_id: str
    status: str
    topic: str
    last_error: str


class LocalJsonContextProvider:
    """Simple local provider. Replace with MCP implementation later."""

    def __init__(self, data_dir: Path) -> None:
        self._users = self._read_json(data_dir / "users.json")
        self._tickets = self._read_json(data_dir / "tickets.json")

    @staticmethod
    def _read_json(path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def get_user(self, user_id: str) -> UserContext | None:
        for row in self._users:
            if row["user_id"] == user_id:
                return UserContext(**row)
        return None

    def get_ticket(self, ticket_id: str) -> TicketContext | None:
        for row in self._tickets:
            if row["ticket_id"] == ticket_id:
                return TicketContext(**row)
        return None
