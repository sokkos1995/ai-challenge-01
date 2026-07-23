from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.provider_client import post_cursor_completion


def _install_fake_cursor_sdk(monkeypatch: pytest.MonkeyPatch, *, prompt_return: object) -> MagicMock:
    agent = MagicMock()
    agent.prompt.return_value = prompt_return

    def agent_options(**kwargs):
        return SimpleNamespace(**kwargs)

    def local_options(**kwargs):
        return SimpleNamespace(**kwargs)

    fake_sdk = ModuleType("cursor_sdk")
    fake_sdk.Agent = agent  # type: ignore[attr-defined]
    fake_sdk.AgentOptions = agent_options  # type: ignore[attr-defined]
    fake_sdk.LocalAgentOptions = local_options  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cursor_sdk", fake_sdk)
    return agent


def test_post_cursor_completion_maps_openai_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_usage = SimpleNamespace(
        input_tokens=11,
        output_tokens=7,
        total_tokens=18,
        reasoning_tokens=2,
    )
    fake_result = SimpleNamespace(
        id="run_1",
        status="finished",
        result="Hello from Cursor",
        model=SimpleNamespace(id="composer-2.5"),
        usage=fake_usage,
    )
    agent = _install_fake_cursor_sdk(monkeypatch, prompt_return=fake_result)

    data = post_cursor_completion(
        api_key="crsr_test",
        model="composer-2.5",
        messages=[
            {"role": "system", "content": "Be brief"},
            {"role": "user", "content": "Hi"},
        ],
        cwd="/tmp/project",
    )

    assert data["object"] == "chat.completion"
    assert data["model"] == "composer-2.5"
    assert data["choices"][0]["message"]["content"] == "Hello from Cursor"
    assert data["usage"]["prompt_tokens"] == 11
    assert data["usage"]["completion_tokens"] == 7
    assert data["usage"]["total_tokens"] == 18
    assert data["usage"]["completion_tokens_details"]["reasoning_tokens"] == 2

    agent.prompt.assert_called_once()
    prompt_arg = agent.prompt.call_args.args[0]
    assert "system: Be brief" in prompt_arg
    assert "user: Hi" in prompt_arg


def test_post_cursor_completion_requires_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cursor_sdk", None)

    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cursor_sdk" or name.startswith("cursor_sdk."):
            raise ImportError("No module named cursor_sdk")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="pip install cursor-sdk"):
        post_cursor_completion(
            api_key="crsr_test",
            model="composer-2.5",
            messages=[{"role": "user", "content": "Hi"}],
            cwd=".",
        )


def test_post_cursor_completion_rejects_non_finished(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_result = SimpleNamespace(
        id="run_err",
        status="error",
        result=None,
        model=None,
        usage=None,
    )
    _install_fake_cursor_sdk(monkeypatch, prompt_return=fake_result)

    with pytest.raises(RuntimeError, match="did not finish"):
        post_cursor_completion(
            api_key="crsr_test",
            model="composer-2.5",
            messages=[{"role": "user", "content": "Hi"}],
            cwd=".",
        )
