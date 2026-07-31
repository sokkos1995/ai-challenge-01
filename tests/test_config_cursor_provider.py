from __future__ import annotations

import os

import pytest

from app.config import (
    CURSOR_API_URL,
    CURSOR_DEFAULT_MODEL,
    cursor_cwd_from_env,
    get_provider_config,
)


def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "LLM_PROVIDER",
        "LLM_API_KEY",
        "LLM_API_URL",
        "LLM_MODEL",
        "LLM_FALLBACK_MODELS",
        "OPENROUTER_API_KEY",
        "GROQ_API_KEY",
        "CURSOR_API_KEY",
        "CURSOR_CWD",
    ):
        monkeypatch.delenv(key, raising=False)


def test_auto_selects_cursor_when_only_cursor_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("CURSOR_API_KEY", "crsr_test")

    provider, api_url, api_key, models = get_provider_config()

    assert provider == "cursor"
    assert api_url == CURSOR_API_URL
    assert api_key == "crsr_test"
    assert models[0] == CURSOR_DEFAULT_MODEL
    assert "composer-2" in models
    assert "auto" in models


def test_explicit_cursor_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "cursor")
    monkeypatch.setenv("CURSOR_API_KEY", "crsr_explicit")
    monkeypatch.setenv("LLM_MODEL", "composer-2")

    provider, api_url, api_key, models = get_provider_config()

    assert provider == "cursor"
    assert api_url == CURSOR_API_URL
    assert api_key == "crsr_explicit"
    assert models[0] == "composer-2"


def test_cursor_accepts_llm_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "cursor")
    monkeypatch.setenv("LLM_API_KEY", "crsr_via_llm")

    provider, _, api_key, _ = get_provider_config()

    assert provider == "cursor"
    assert api_key == "crsr_via_llm"


def test_cursor_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "cursor")

    with pytest.raises(RuntimeError, match="CURSOR_API_KEY"):
        get_provider_config()


def test_cursor_cwd_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("CURSOR_CWD", str(tmp_path))
    assert cursor_cwd_from_env() == str(tmp_path)

    monkeypatch.delenv("CURSOR_CWD", raising=False)
    assert cursor_cwd_from_env() == os.getcwd()


def test_get_routing_models_cursor_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import (
        CURSOR_ROUTING_CHEAP_MODEL,
        CURSOR_ROUTING_STRONG_MODEL,
        get_routing_models,
    )

    monkeypatch.delenv("LLM_CHEAP_MODEL", raising=False)
    monkeypatch.delenv("LLM_STRONG_MODEL", raising=False)
    cheap, strong = get_routing_models("cursor")
    assert cheap == CURSOR_ROUTING_CHEAP_MODEL
    assert strong == CURSOR_ROUTING_STRONG_MODEL

    monkeypatch.setenv("LLM_CHEAP_MODEL", "composer-2")
    monkeypatch.setenv("LLM_STRONG_MODEL", "composer-2.5")
    cheap2, strong2 = get_routing_models("cursor")
    assert cheap2 == "composer-2"
    assert strong2 == "composer-2.5"
