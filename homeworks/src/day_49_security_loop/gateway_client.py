"""HTTP / in-process client for day_48 LLM gateway."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Optional

GuardMode = Literal["block", "redact"]


@dataclass
class GatewayEvent:
    stage: str  # generate | security_review | ...
    blocked: bool
    blocked_stage: str | None
    findings: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    audit_id: str = ""
    status: str = "clean"  # clean | redacted | blocked
    answer_preview: str = ""
    live: bool = False
    model: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GatewayChatResult:
    answer: str
    event: GatewayEvent
    raw: dict[str, Any] = field(default_factory=dict)


CompleterFn = Callable[[list[dict[str, str]], Optional[str]], str]


class GatewayClient:
    """Call day_48 gateway over HTTP, or an in-process completer (offline)."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        mode: GuardMode | None = None,
        timeout_sec: float = 120.0,
        in_process: CompleterFn | None = None,
        use_input_guard: bool = True,
    ) -> None:
        self.base_url = (base_url or os.getenv("GATEWAY_URL", "http://127.0.0.1:8848")).rstrip("/")
        raw_mode = (mode or os.getenv("GATEWAY_INPUT_MODE", "redact")).strip().lower()
        self.mode: GuardMode = "block" if raw_mode == "block" else "redact"
        self.timeout_sec = timeout_sec
        self.in_process = in_process
        self.use_input_guard = use_input_guard
        self.events: list[GatewayEvent] = []

    def chat(
        self,
        *,
        prompt: str,
        stage: str,
        system: str | None = None,
        model: str | None = None,
    ) -> GatewayChatResult:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if self.in_process is not None:
            return self._chat_in_process(messages=messages, stage=stage, model=model)

        return self._chat_http(messages=messages, prompt=prompt, stage=stage, model=model)

    def _classify(self, *, blocked: bool, findings: list[str], warnings: list[str]) -> str:
        if blocked:
            return "blocked"
        if findings or any("redact" in w.lower() or "REDACTED" in w for w in warnings):
            return "redacted" if findings or warnings else "clean"
        if findings:
            return "redacted"
        return "clean"

    def _chat_http(
        self,
        *,
        messages: list[dict[str, str]],
        prompt: str,
        stage: str,
        model: str | None,
    ) -> GatewayChatResult:
        body: dict[str, Any] = {
            "prompt": prompt,
            "messages": messages,
            "mode": self.mode,
            "output_mode": "redact",
        }
        if model:
            body["model"] = model
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
                status_code = getattr(resp, "status", 200)
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            try:
                raw = json.loads(payload)
            except json.JSONDecodeError:
                event = GatewayEvent(
                    stage=stage,
                    blocked=True,
                    blocked_stage="http",
                    findings=[],
                    warnings=[f"HTTP {exc.code}"],
                    status="blocked",
                    error=payload[:300],
                )
                self.events.append(event)
                return GatewayChatResult(answer="", event=event, raw={})
            status_code = exc.code
        except urllib.error.URLError as exc:
            event = GatewayEvent(
                stage=stage,
                blocked=True,
                blocked_stage="transport",
                findings=[],
                warnings=[str(exc.reason)],
                status="blocked",
                error=str(exc),
            )
            self.events.append(event)
            return GatewayChatResult(answer="", event=event, raw={})

        blocked = bool(raw.get("blocked")) or status_code >= 400
        findings = list(raw.get("findings") or [])
        warnings = list(raw.get("warnings") or [])
        status = self._classify(blocked=blocked, findings=findings, warnings=warnings)
        if findings and not blocked and self.mode == "redact":
            status = "redacted"
        event = GatewayEvent(
            stage=stage,
            blocked=blocked,
            blocked_stage=raw.get("blocked_stage"),
            findings=findings,
            warnings=warnings,
            audit_id=str(raw.get("audit_id") or ""),
            status=status if not blocked else "blocked",
            answer_preview=str(raw.get("answer") or "")[:200],
            live=bool(raw.get("live")),
            model=str(raw.get("model") or ""),
        )
        self.events.append(event)
        return GatewayChatResult(answer=str(raw.get("answer") or ""), event=event, raw=raw)

    def _chat_in_process(
        self,
        *,
        messages: list[dict[str, str]],
        stage: str,
        model: str | None,
    ) -> GatewayChatResult:
        user_text = "\n".join(
            m["content"] for m in messages if m.get("role") == "user"
        )
        findings: list[str] = []
        warnings: list[str] = []
        cleaned = user_text
        blocked = False
        blocked_stage: str | None = None

        if self.use_input_guard:
            from day_48_llm_gateway.input_guard import check_input

            result = check_input(user_text, mode=self.mode)
            findings = list(result.finding_kinds)
            warnings = list(result.warnings)
            cleaned = result.cleaned_text
            if not result.ok:
                blocked = True
                blocked_stage = "input"
                status = "blocked"
                event = GatewayEvent(
                    stage=stage,
                    blocked=True,
                    blocked_stage=blocked_stage,
                    findings=findings,
                    warnings=warnings,
                    status=status,
                )
                self.events.append(event)
                return GatewayChatResult(answer="", event=event, raw={"blocked": True})

        assert self.in_process is not None
        # rebuild messages with cleaned user content
        out_messages: list[dict[str, str]] = []
        for m in messages:
            if m.get("role") == "user":
                out_messages.append({"role": "user", "content": cleaned})
            else:
                out_messages.append(m)
        answer = self.in_process(out_messages, model)
        status = "redacted" if findings else "clean"
        event = GatewayEvent(
            stage=stage,
            blocked=False,
            blocked_stage=None,
            findings=findings,
            warnings=warnings,
            status=status,
            answer_preview=answer[:200],
            model=model or "offline",
        )
        self.events.append(event)
        return GatewayChatResult(
            answer=answer,
            event=event,
            raw={"answer": answer, "findings": findings, "blocked": False},
        )
