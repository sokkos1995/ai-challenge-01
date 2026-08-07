"""FastAPI LLM gateway with input/output guards, rate limit, cost, audit."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .audit import AuditLogger, AuditRecord, preview_hash, safe_preview
from .cost import estimate_cost_usd
from .input_guard import GuardMode, check_input
from .output_guard import check_output
from .proxy import get_completer, set_completer
from .rate_limit import RateLimiter

ART = Path(__file__).resolve().parents[2] / "artifacts" / "day_48"
DEFAULT_AUDIT_PATH = ART / "audit.jsonl"

app = FastAPI(title="Day 48 LLM Gateway", version="0.1.0")

_rate_limiter = RateLimiter(
    limit=int(os.getenv("GATEWAY_RATE_LIMIT_PER_MIN", "30")),
    window_sec=60.0,
)
_audit = AuditLogger(Path(os.getenv("GATEWAY_AUDIT_PATH", str(DEFAULT_AUDIT_PATH))))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_mode() -> GuardMode:
    raw = os.getenv("GATEWAY_INPUT_MODE", "block").strip().lower()
    return "redact" if raw == "redact" else "block"


def configure_for_tests(
    *,
    audit_path: Path | None = None,
    rate_limit: int = 30,
    completer: Any = None,
) -> None:
    """Reset mutable gateway state (used by pytest)."""
    global _rate_limiter, _audit
    _rate_limiter = RateLimiter(limit=rate_limit, window_sec=60.0)
    if audit_path is not None:
        _audit = AuditLogger(audit_path)
    set_completer(completer)


class ChatRequest(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[list[dict[str, str]]] = None
    mode: GuardMode = Field(default_factory=_default_mode)
    model: Optional[str] = None
    output_mode: Literal["block", "redact"] = "block"


class ChatResponse(BaseModel):
    answer: str
    blocked: bool
    blocked_stage: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    cost_usd: float = 0.0
    audit_id: str = ""
    model: str = ""
    live: bool = False


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _resolve_user_text(body: ChatRequest) -> str:
    if body.prompt and body.prompt.strip():
        return body.prompt.strip()
    if body.messages:
        parts = [
            (m.get("content") or "").strip()
            for m in body.messages
            if (m.get("role") or "user") == "user"
        ]
        joined = "\n".join(p for p in parts if p)
        if joined:
            return joined
    raise HTTPException(status_code=400, detail="Provide non-empty prompt or messages")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "day48-llm-gateway"}


@app.post("/v1/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request) -> ChatResponse | JSONResponse:
    ip = _client_ip(request)
    mode: GuardMode = body.mode if body.mode in ("block", "redact") else "block"
    audit_id = _audit.new_id()

    if not _rate_limiter.allow(ip):
        record = AuditRecord(
            audit_id=audit_id,
            ts=_now_iso(),
            ip=ip,
            mode=mode,
            blocked_stage="rate_limit",
            warnings=["Rate limit exceeded"],
            prompt_hash="",
            prompt_preview="",
        )
        _audit.write(record)
        return JSONResponse(
            status_code=429,
            content=ChatResponse(
                answer="",
                blocked=True,
                blocked_stage="rate_limit",
                warnings=["Rate limit: too many requests from this IP"],
                findings=["rate_limit"],
                audit_id=audit_id,
            ).model_dump(),
        )

    user_text = _resolve_user_text(body)
    input_result = check_input(user_text, mode=mode)

    if not input_result.ok:
        findings = input_result.finding_kinds
        # Never log raw secrets — preview via redact pass
        redacted_preview = check_input(user_text, mode="redact").cleaned_text
        record = AuditRecord(
            audit_id=audit_id,
            ts=_now_iso(),
            ip=ip,
            mode=mode,
            blocked_stage="input",
            input_findings=findings,
            warnings=list(input_result.warnings),
            prompt_hash=preview_hash(user_text),
            prompt_preview=safe_preview(redacted_preview or "[blocked]"),
        )
        _audit.write(record)
        return JSONResponse(
            status_code=403,
            content=ChatResponse(
                answer="",
                blocked=True,
                blocked_stage="input",
                warnings=list(input_result.warnings),
                findings=findings,
                audit_id=audit_id,
            ).model_dump(),
        )

    # Build messages for upstream
    cleaned = input_result.cleaned_text
    if body.messages:
        messages: list[dict[str, str]] = []
        for m in body.messages:
            role = m.get("role") or "user"
            content = m.get("content") or ""
            if role == "user":
                # Replace all user contents with redacted aggregate once
                continue
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": cleaned})
    else:
        messages = [{"role": "user", "content": cleaned}]

    completer = get_completer()
    try:
        proxy_result = completer(messages, model=body.model)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upstream LLM error: {exc}") from exc

    out = check_output(proxy_result.answer, mode=body.output_mode)
    cost = estimate_cost_usd(
        proxy_result.usage.prompt_tokens,
        proxy_result.usage.completion_tokens,
    )
    usage = {
        "prompt_tokens": proxy_result.usage.prompt_tokens,
        "completion_tokens": proxy_result.usage.completion_tokens,
        "total_tokens": proxy_result.usage.total_tokens,
    }

    if not out.ok:
        out_kinds = sorted({f.kind for f in out.findings})
        record = AuditRecord(
            audit_id=audit_id,
            ts=_now_iso(),
            ip=ip,
            mode=mode,
            blocked_stage="output",
            input_findings=input_result.finding_kinds,
            output_findings=out_kinds,
            warnings=list(input_result.warnings) + list(out.warnings),
            prompt_hash=preview_hash(cleaned),
            answer_hash=preview_hash(proxy_result.answer),
            prompt_preview=safe_preview(cleaned),
            model=proxy_result.model,
            prompt_tokens=proxy_result.usage.prompt_tokens,
            completion_tokens=proxy_result.usage.completion_tokens,
            cost_usd=cost,
        )
        _audit.write(record)
        return JSONResponse(
            status_code=403,
            content=ChatResponse(
                answer="",
                blocked=True,
                blocked_stage="output",
                warnings=list(input_result.warnings) + list(out.warnings),
                findings=out_kinds,
                usage=usage,
                cost_usd=cost,
                audit_id=audit_id,
                model=proxy_result.model,
                live=proxy_result.live,
            ).model_dump(),
        )

    answer = out.safe_text
    all_findings = input_result.finding_kinds + sorted({f.kind for f in out.findings})
    warnings = list(input_result.warnings) + list(out.warnings)

    record = AuditRecord(
        audit_id=audit_id,
        ts=_now_iso(),
        ip=ip,
        mode=mode,
        blocked_stage=None,
        input_findings=input_result.finding_kinds,
        output_findings=sorted({f.kind for f in out.findings}),
        warnings=warnings,
        prompt_hash=preview_hash(cleaned),
        answer_hash=preview_hash(answer),
        prompt_preview=safe_preview(cleaned),
        model=proxy_result.model,
        prompt_tokens=proxy_result.usage.prompt_tokens,
        completion_tokens=proxy_result.usage.completion_tokens,
        cost_usd=cost,
    )
    _audit.write(record)

    return ChatResponse(
        answer=answer,
        blocked=False,
        blocked_stage=None,
        warnings=warnings,
        findings=all_findings,
        usage=usage,
        cost_usd=cost,
        audit_id=audit_id,
        model=proxy_result.model,
        live=proxy_result.live,
    )


def main() -> None:
    import uvicorn

    port = int(os.getenv("GATEWAY_PORT", "8848"))
    uvicorn.run(
        "homeworks.src.day_48_llm_gateway.gateway_app:app",
        host="127.0.0.1",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
