"""In-process LLM Input/Output Guard for ProviderService (day_48 logic in app/)."""
from __future__ import annotations

import copy
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from app.config import bool_from_env, guard_mode_from_env
from app.services.llm_input_guard import (
    Finding,
    GuardMode,
    GuardResult,
    check_input,
    detect_secrets,
)

GuardStage = Literal["input", "output"]

_SUSPICIOUS_URL = re.compile(
    r"(?i)\bhttps?://("
    r"(?:\d{1,3}\.){3}\d{1,3}"
    r"|[a-z0-9.-]+\.onion"
    r"|evil\.example"
    r"|malware\."
    r"|pastebin\.com/[A-Za-z0-9]+"
    r")\S*"
)
_SHELL_CMD = re.compile(
    r"(?i)("
    r"\brm\s+-rf\b|"
    r"\bcurl\b[^\n]{0,80}\|\s*(?:ba)?sh\b|"
    r"\bwget\b[^\n]{0,80}\|\s*(?:ba)?sh\b|"
    r"\bpowershell\s+-enc\b|"
    r"\bchmod\s+777\b"
    r")"
)


@dataclass(frozen=True)
class GuardEvent:
    stage: GuardStage
    blocked: bool
    mode: str
    findings: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class _AppOutputResult:
    ok: bool
    findings: list[Finding]
    reasons: list[str]
    safe_text: str
    warnings: list[str]


def _check_output_app(text: str, *, mode: GuardMode) -> _AppOutputResult:
    """Output guard for CLI: secrets / suspicious URL / shell — not broad 'you are a' leak regex."""
    findings: list[Finding] = []
    reasons: list[str] = []
    secret_hits = detect_secrets(text)
    if secret_hits:
        findings.extend(secret_hits)
        reasons.append("secret_in_output")
    for m in _SUSPICIOUS_URL.finditer(text):
        findings.append(
            Finding(kind="suspicious_url", match=m.group(0)[:120], start=m.start(), end=m.end())
        )
        reasons.append("suspicious_url")
    for m in _SHELL_CMD.finditer(text):
        findings.append(
            Finding(kind="shell_command", match=m.group(0)[:120], start=m.start(), end=m.end())
        )
        reasons.append("shell_command")
    reasons = sorted(set(reasons))
    if not findings:
        return _AppOutputResult(ok=True, findings=[], reasons=[], safe_text=text, warnings=[])

    warning = f"Output Guard: {', '.join(reasons)}"
    if mode == "redact":
        secret_result = check_input(text, mode="redact")
        safe = secret_result.cleaned_text
        safe = _SUSPICIOUS_URL.sub("[REDACTED_URL]", safe)
        safe = _SHELL_CMD.sub("[REDACTED_COMMAND]", safe)
        return _AppOutputResult(
            ok=True,
            findings=findings,
            reasons=reasons,
            safe_text=safe,
            warnings=[warning, "Output redacted before delivery"],
        )
    return _AppOutputResult(
        ok=False,
        findings=findings,
        reasons=reasons,
        safe_text="",
        warnings=[warning, "Model response blocked — not delivered to client"],
    )


class LlmGuardService:
    """Apply day_48-style guards around every provider completion."""

    def __init__(
        self,
        *,
        input_enabled: bool = True,
        output_enabled: bool = True,
        input_mode: GuardMode = "redact",
        output_mode: GuardMode = "redact",
    ) -> None:
        self.input_enabled = input_enabled
        self.output_enabled = output_enabled
        self.input_mode: GuardMode = input_mode if input_mode in ("block", "redact") else "redact"
        self.output_mode: GuardMode = output_mode if output_mode in ("block", "redact") else "redact"
        self.last_events: list[GuardEvent] = []

    @classmethod
    def from_env(cls) -> "LlmGuardService":
        return cls(
            input_enabled=bool_from_env("LLM_INPUT_GUARD", default=True),
            output_enabled=bool_from_env("LLM_OUTPUT_GUARD", default=True),
            input_mode=cast(GuardMode, guard_mode_from_env("LLM_INPUT_GUARD_MODE", default="redact")),
            output_mode=cast(GuardMode, guard_mode_from_env("LLM_OUTPUT_GUARD_MODE", default="redact")),
        )

    @classmethod
    def disabled(cls) -> "LlmGuardService":
        return cls(input_enabled=False, output_enabled=False)

    def prepare_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        self.last_events = []
        if not self.input_enabled:
            return [dict(m) for m in messages]

        prepared: list[dict[str, str]] = []
        for message in messages:
            content = str(message.get("content") or "")
            result = check_input(content, mode=self.input_mode)
            event = self._event_from_input(result)
            if result.findings:
                self.last_events.append(event)
                self._warn(event)
            if not result.ok:
                kinds = ", ".join(result.finding_kinds) or "secret"
                raise RuntimeError(
                    f"LLM Input Guard blocked the request ({kinds}). "
                    "Remove secrets from the prompt or set LLM_INPUT_GUARD_MODE=redact / "
                    "LLM_INPUT_GUARD=0 (not recommended)."
                )
            prepared.append({**message, "content": result.cleaned_text})
        return prepared

    def filter_response_payload(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.output_enabled:
            return data
        content = _assistant_content(data)
        if not content:
            return data
        result = _check_output_app(content, mode=self.output_mode)
        if result.findings:
            event = GuardEvent(
                stage="output",
                blocked=not result.ok,
                mode=self.output_mode,
                findings=sorted({f.kind for f in result.findings}),
                warnings=list(result.warnings),
            )
            self.last_events.append(event)
            self._warn(event)
        if not result.ok:
            reasons = ", ".join(result.reasons) or "policy"
            raise RuntimeError(
                f"LLM Output Guard blocked the model response ({reasons}). "
                "Set LLM_OUTPUT_GUARD_MODE=redact or LLM_OUTPUT_GUARD=0 to override."
            )
        if result.safe_text != content:
            return _with_assistant_content(data, result.safe_text)
        return data

    def _event_from_input(self, result: GuardResult) -> GuardEvent:
        return GuardEvent(
            stage="input",
            blocked=not result.ok,
            mode=result.mode,
            findings=list(result.finding_kinds),
            warnings=list(result.warnings),
        )

    def _warn(self, event: GuardEvent) -> None:
        detail = ", ".join(event.findings) or "hit"
        print(
            f"Warning: LLM {event.stage} guard ({event.mode}): {detail}",
            file=sys.stderr,
        )


def _assistant_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _with_assistant_content(data: dict[str, Any], text: str) -> dict[str, Any]:
    cloned = copy.deepcopy(data)
    choices = cloned.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].setdefault("message", {})
        if isinstance(message, dict):
            message["content"] = text
    return cloned
