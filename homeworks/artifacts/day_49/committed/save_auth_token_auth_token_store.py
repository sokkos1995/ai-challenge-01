"""Secure: token only from argument / env, never hardcoded."""
from __future__ import annotations

import os
from pathlib import Path

TOKEN_PATH = Path(os.environ.get("AUTH_TOKEN_PATH", ".auth_token_local"))


def save_token(token: str | None = None) -> Path:
    value = (token or os.environ.get("AUTH_TOKEN") or "").strip()
    if not value:
        raise ValueError("token must come from argument or AUTH_TOKEN env")
    TOKEN_PATH.write_text(value, encoding="utf-8")
    return TOKEN_PATH


def load_token() -> str:
    if not TOKEN_PATH.is_file():
        raise FileNotFoundError(str(TOKEN_PATH))
    return TOKEN_PATH.read_text(encoding="utf-8").strip()
