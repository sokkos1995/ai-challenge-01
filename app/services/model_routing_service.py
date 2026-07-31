from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

from app.models import AgentRequestOptions
from app.response_parser import parse_agent_response

DEFAULT_MIN_ANSWER_CHARS = 24
DEFAULT_MIN_CONFIDENCE = 0.55

_UNCERTAIN_MARKERS = (
    "не уверен",
    "не уверена",
    "не уверены",
    "не знаю",
    "затрудняюсь",
    "возможно",
    "кажется",
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "unsure",
    "uncertain",
    "not confident",
    "cannot determine",
    "can't determine",
    "no idea",
)

_ROUTING_SYSTEM = (
    "You answer user questions. Reply with a single JSON object only, no markdown:\n"
    '{"answer":"<final answer text>","confidence":0.0,"status":"OK"|"UNSURE"|"FAIL"}\n'
    "confidence is your self-rated certainty in [0,1]. "
    "Use UNSURE or FAIL when the question is ambiguous, underspecified, or you lack facts."
)

_STRONG_SYSTEM = (
    "You are a stronger fallback model. Give a clear, direct answer. "
    "If still uncertain, say so explicitly and list what is missing."
)


class SupportsComplete(Protocol):
    provider: str

    def complete(
        self,
        messages: list[dict[str, str]],
        options: AgentRequestOptions,
        *,
        model_candidates: list[str] | None = None,
    ) -> tuple[dict, str, float]:
        ...


@dataclass
class RoutingMetrics:
    llm_calls: int = 0
    cheap_latency_sec: float = 0.0
    strong_latency_sec: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    escalate_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "llm_calls": self.llm_calls,
            "cheap_latency_sec": round(self.cheap_latency_sec, 4),
            "strong_latency_sec": round(self.strong_latency_sec, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "escalate_reasons": list(self.escalate_reasons),
        }


@dataclass(frozen=True)
class RoutingResult:
    answer: str
    model_used: str
    tier: str
    escalated: bool
    cheap_model: str
    strong_model: str
    cheap_answer: str
    confidence: Optional[float]
    status: Optional[str]
    metrics: RoutingMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "model_used": self.model_used,
            "tier": self.tier,
            "escalated": self.escalated,
            "cheap_model": self.cheap_model,
            "strong_model": self.strong_model,
            "cheap_answer": self.cheap_answer,
            "confidence": self.confidence,
            "status": self.status,
            "metrics": self.metrics.to_dict(),
        }


def looks_uncertain(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _UNCERTAIN_MARKERS)


def extract_json_object(text: str) -> Optional[dict]:
    clean = text.strip()
    if not clean:
        return None

    candidates = [clean]
    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidates.append(clean[start : end + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def evaluate_escalation(
    *,
    answer: str,
    confidence: Optional[float],
    status: Optional[str],
    finish_reason: Optional[str],
    min_answer_chars: int,
    min_confidence: float,
) -> list[str]:
    """Return escalate reasons; empty list means stay on cheap model."""
    reasons: list[str] = []
    cleaned = answer.strip()
    status_up = status.strip().upper() if status else None

    if not cleaned:
        reasons.append("empty_answer")
    elif len(cleaned) < min_answer_chars:
        # Short but high-confidence OK answers (numbers, yes/no) stay on cheap.
        confident_short = (
            confidence is not None
            and confidence >= max(min_confidence, 0.85)
            and status_up == "OK"
        )
        if not confident_short:
            reasons.append("short_answer")

    if status_up in {"UNSURE", "FAIL"}:
        reasons.append(f"status_{status_up.lower()}")

    if confidence is not None and confidence < min_confidence:
        reasons.append("low_confidence")

    if looks_uncertain(cleaned):
        reasons.append("uncertain_markers")

    if finish_reason and str(finish_reason).lower() in {"length", "max_tokens"}:
        reasons.append("truncated")

    # Deduplicate while preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            ordered.append(reason)
    return ordered


def _usage_tokens(data: dict) -> tuple[int, int]:
    usage = data.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return prompt, completion


def _parse_routed_payload(raw_text: str) -> tuple[str, Optional[float], Optional[str]]:
    payload = extract_json_object(raw_text)
    if not payload:
        return raw_text.strip(), None, None

    answer = str(payload.get("answer") or "").strip()
    if not answer:
        # Model returned JSON but forgot answer — keep full text as fallback.
        answer = raw_text.strip()

    confidence: Optional[float] = None
    conf_raw = payload.get("confidence")
    if isinstance(conf_raw, (int, float)):
        confidence = float(conf_raw)
    elif isinstance(conf_raw, str):
        try:
            confidence = float(conf_raw.strip())
        except ValueError:
            confidence = None

    status_raw = payload.get("status")
    status = str(status_raw).strip().upper() if status_raw is not None else None
    if status == "":
        status = None
    return answer, confidence, status


_MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = _MARKDOWN_FENCE_RE.sub("", cleaned).strip()
    return cleaned


class ModelRoutingService:
    """
    Quality-based routing: try cheap/fast model first, escalate to strong if uncertain.

    Heuristics (any one triggers escalate):
    - answer length below threshold / empty
    - self-reported confidence below min
    - status UNSURE/FAIL or uncertainty markers in text
    - truncated finish_reason
    """

    def __init__(
        self,
        provider: SupportsComplete,
        cheap_model: str,
        strong_model: str,
        *,
        min_answer_chars: int = DEFAULT_MIN_ANSWER_CHARS,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        if not cheap_model.strip():
            raise ValueError("cheap_model must not be empty")
        if not strong_model.strip():
            raise ValueError("strong_model must not be empty")
        self._provider = provider
        self.cheap_model = cheap_model.strip()
        self.strong_model = strong_model.strip()
        self.min_answer_chars = max(1, min_answer_chars)
        self.min_confidence = min(1.0, max(0.0, min_confidence))

    def route(
        self,
        user_text: str,
        *,
        options: Optional[AgentRequestOptions] = None,
        system_prompt: Optional[str] = None,
    ) -> RoutingResult:
        cleaned_user = user_text.strip()
        if not cleaned_user:
            raise ValueError("user_text must not be empty")

        opts = options or AgentRequestOptions(
            temperature=0.2,
            top_p=None,
            top_k=None,
            response_format=None,
            max_output_tokens=512,
            stop_sequences=[],
            finish_instruction=None,
        )
        metrics = RoutingMetrics()

        cheap_messages = [
            {"role": "system", "content": system_prompt or _ROUTING_SYSTEM},
            {"role": "user", "content": cleaned_user},
        ]
        cheap_data, cheap_tried, cheap_latency = self._provider.complete(
            cheap_messages,
            opts,
            model_candidates=[self.cheap_model],
        )
        metrics.llm_calls += 1
        metrics.cheap_latency_sec = cheap_latency
        p_tok, c_tok = _usage_tokens(cheap_data)
        metrics.prompt_tokens += p_tok
        metrics.completion_tokens += c_tok

        cheap_parsed = parse_agent_response(
            cheap_data, cheap_tried, cheap_latency, self._provider.provider
        )
        raw_cheap = _strip_fences(cheap_parsed.answer)
        answer, confidence, status = _parse_routed_payload(raw_cheap)
        finish_reason = None
        choices = cheap_data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            finish_reason = choices[0].get("finish_reason")

        reasons = evaluate_escalation(
            answer=answer,
            confidence=confidence,
            status=status,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            min_answer_chars=self.min_answer_chars,
            min_confidence=self.min_confidence,
        )
        metrics.escalate_reasons = list(reasons)

        if not reasons:
            return RoutingResult(
                answer=answer,
                model_used=cheap_tried,
                tier="cheap",
                escalated=False,
                cheap_model=self.cheap_model,
                strong_model=self.strong_model,
                cheap_answer=answer,
                confidence=confidence,
                status=status,
                metrics=metrics,
            )

        strong_messages = [
            {"role": "system", "content": _STRONG_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Original question:\n{cleaned_user}\n\n"
                    f"A smaller model answered with low certainty "
                    f"(reasons: {', '.join(reasons)}):\n{answer}\n\n"
                    "Provide a better final answer."
                ),
            },
        ]
        strong_data, strong_tried, strong_latency = self._provider.complete(
            strong_messages,
            opts,
            model_candidates=[self.strong_model],
        )
        metrics.llm_calls += 1
        metrics.strong_latency_sec = strong_latency
        sp, sc = _usage_tokens(strong_data)
        metrics.prompt_tokens += sp
        metrics.completion_tokens += sc

        strong_parsed = parse_agent_response(
            strong_data, strong_tried, strong_latency, self._provider.provider
        )
        strong_answer = _strip_fences(strong_parsed.answer).strip()
        # Strong path may still return JSON; prefer nested answer if present.
        strong_text, _, _ = _parse_routed_payload(strong_answer)
        if not strong_text.strip():
            strong_text = strong_answer

        return RoutingResult(
            answer=strong_text,
            model_used=strong_tried,
            tier="strong",
            escalated=True,
            cheap_model=self.cheap_model,
            strong_model=self.strong_model,
            cheap_answer=answer,
            confidence=confidence,
            status=status,
            metrics=metrics,
        )
