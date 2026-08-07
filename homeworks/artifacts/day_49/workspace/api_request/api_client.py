"""Secure: HTTPS + key from env header."""
from __future__ import annotations

import os
import urllib.request


def fetch(path: str = "", *, base: str | None = None) -> bytes:
    root = (base or os.environ.get("API_BASE_URL") or "https://api.example.com/v1/data").rstrip("/")
    if root.startswith("http://"):
        raise ValueError("only https:// base URLs are allowed")
    key = (os.environ.get("API_KEY") or "").strip()
    if not key:
        raise ValueError("API_KEY env is required")
    url = f"{root}{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
        return resp.read()
