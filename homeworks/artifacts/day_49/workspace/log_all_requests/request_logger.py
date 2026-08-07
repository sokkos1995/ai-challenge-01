"""Secure: redact Authorization and token-like headers before logging."""
from __future__ import annotations

import logging
from typing import Mapping

logger = logging.getLogger("requests")

_REDACT_KEYS = {"authorization", "x-api-key", "api-key", "cookie"}


def _redact(headers: Mapping[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in headers.items():
        if k.lower() in _REDACT_KEYS or "token" in k.lower():
            out[k] = "[REDACTED]"
        else:
            out[k] = v
    return out


def log_request(method: str, url: str, headers: Mapping[str, str] | None = None) -> str:
    safe = _redact(dict(headers or {}))
    line = f"{method} {url} headers={safe}"
    logger.info(line)
    return line
