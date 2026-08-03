from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

from app.models import AgentRequestOptions
from app.response_parser import parse_agent_response

ALLOWED_LABELS: frozenset[str] = frozenset(
    {
        "faq",
        "code_help",
        "action",
        "rag",
        "chitchat",
        "other",
    }
)
ALLOWED_STATUSES: frozenset[str] = frozenset({"OK", "UNSURE", "FAIL"})

DEFAULT_MIN_CONFIDENCE = 0.60

_MICRO_SYSTEM = (
    "You are a micro classifier for an LLM CLI agent.\n"
    "Classify the user query into exactly one label.\n"
    "Labels:\n"
    "- faq: short factual / howto question\n"
    "- code_help: programming, debug, refactor, code review\n"
    "- action: request to create/complete/list a task or run a tool\n"
    "- rag: question about this project's docs/codebase/homework\n"
    "- chitchat: greeting, thanks, small talk\n"
    "- other: none of the above\n"
    "Reply with a single JSON object only, no markdown:\n"
    '{"label":"faq|code_help|action|rag|chitchat|other",'
    '"confidence":0.0,"status":"OK"|"UNSURE"|"FAIL"}\n'
    "confidence is self-rated certainty in [0,1]. "
    "Use UNSURE when the query is ambiguous, mixed, or noisy."
)

_FALLBACK_SYSTEM = (
    "You are a strong classifier for an LLM CLI agent.\n"
    "A smaller model was unsure or returned an invalid payload.\n"
    "Classify the user query into exactly one label: "
    "faq | code_help | action | rag | chitchat | other.\n"
    "Reply with a single JSON object only, no markdown:\n"
    '{"label":"...","confidence":0.0,"status":"OK"|"UNSURE"|"FAIL"}'
)

_MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


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
class MicroFirstMetrics:
    llm_calls: int = 0
    micro_calls: int = 0
    fallback_calls: int = 0
    micro_latency_sec: float = 0.0
    fallback_latency_sec: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    fallback_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "llm_calls": self.llm_calls,
            "micro_calls": self.micro_calls,
            "fallback_calls": self.fallback_calls,
            "micro_latency_sec": round(self.micro_latency_sec, 4),
            "fallback_latency_sec": round(self.fallback_latency_sec, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "fallback_reasons": list(self.fallback_reasons),
        }


@dataclass(frozen=True)
class ClassificationPayload:
    label: str
    confidence: float
    status: str


@dataclass(frozen=True)
class ParseResult:
    ok: bool
    payload: Optional[ClassificationPayload] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class MicroFirstResult:
    label: str
    confidence: float
    status: str
    tier: str
    escalated: bool
    model_used: str
    micro_model: str
    fallback_model: str
    micro_label: Optional[str]
    micro_confidence: Optional[float]
    micro_status: Optional[str]
    metrics: MicroFirstMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "status": self.status,
            "tier": self.tier,
            "escalated": self.escalated,
            "model_used": self.model_used,
            "micro_model": self.micro_model,
            "fallback_model": self.fallback_model,
            "micro_label": self.micro_label,
            "micro_confidence": self.micro_confidence,
            "micro_status": self.micro_status,
            "metrics": self.metrics.to_dict(),
        }


def strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = _MARKDOWN_FENCE_RE.sub("", cleaned).strip()
    return cleaned


def extract_json_object(text: str) -> Optional[dict]:
    clean = strip_fences(text)
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


def parse_classification_payload(raw_text: str) -> ParseResult:
    data = extract_json_object(raw_text)
    if data is None:
        return ParseResult(ok=False, reason="invalid_format")

    label_raw = data.get("label")
    status_raw = data.get("status")
    confidence_raw = data.get("confidence")

    if not isinstance(label_raw, str) or not label_raw.strip():
        return ParseResult(ok=False, reason="missing_label")
    label = label_raw.strip().lower()
    if label not in ALLOWED_LABELS:
        return ParseResult(ok=False, reason=f"invalid_label:{label}")

    if not isinstance(status_raw, str) or not status_raw.strip():
        return ParseResult(ok=False, reason="missing_status")
    status = status_raw.strip().upper()
    if status not in ALLOWED_STATUSES:
        return ParseResult(ok=False, reason=f"invalid_status:{status}")

    confidence: Optional[float] = None
    if isinstance(confidence_raw, (int, float)):
        confidence = float(confidence_raw)
    elif isinstance(confidence_raw, str):
        try:
            confidence = float(confidence_raw.strip())
        except ValueError:
            confidence = None
    if confidence is None:
        return ParseResult(ok=False, reason="missing_confidence")
    if confidence < 0.0 or confidence > 1.0:
        return ParseResult(ok=False, reason="confidence_out_of_range")

    return ParseResult(
        ok=True,
        payload=ClassificationPayload(label=label, confidence=confidence, status=status),
    )


def evaluate_fallback(
    parsed: ParseResult,
    *,
    min_confidence: float,
) -> list[str]:
    """Return fallback reasons; empty list means accept micro-model result."""
    if not parsed.ok or parsed.payload is None:
        return [parsed.reason or "invalid_format"]

    reasons: list[str] = []
    payload = parsed.payload
    if payload.status in {"UNSURE", "FAIL"}:
        reasons.append(f"status_{payload.status.lower()}")
    if payload.confidence < min_confidence:
        reasons.append("low_confidence")
    return reasons


def _usage_tokens(data: dict) -> tuple[int, int]:
    usage = data.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return prompt, completion


class MicroModelFirstService:
    """
    Two-tier inference for query classification.

    Level 1 — micro model (cheap/small) returns structured label + confidence/status.
    Level 2 — large LLM only if micro is UNSURE / low confidence / invalid format.
    """

    def __init__(
        self,
        provider: SupportsComplete,
        micro_model: str,
        fallback_model: str,
        *,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        if not micro_model.strip():
            raise ValueError("micro_model must not be empty")
        if not fallback_model.strip():
            raise ValueError("fallback_model must not be empty")
        self._provider = provider
        self.micro_model = micro_model.strip()
        self.fallback_model = fallback_model.strip()
        self.min_confidence = min(1.0, max(0.0, min_confidence))

    def classify(
        self,
        user_text: str,
        *,
        options: Optional[AgentRequestOptions] = None,
    ) -> MicroFirstResult:
        cleaned_user = user_text.strip()
        if not cleaned_user:
            raise ValueError("user_text must not be empty")

        opts = options or AgentRequestOptions(
            temperature=0.0,
            top_p=None,
            top_k=None,
            response_format=None,
            max_output_tokens=128,
            stop_sequences=[],
            finish_instruction=None,
        )
        metrics = MicroFirstMetrics()

        micro_messages = [
            {"role": "system", "content": _MICRO_SYSTEM},
            {"role": "user", "content": cleaned_user},
        ]
        micro_data, micro_tried, micro_latency = self._provider.complete(
            micro_messages,
            opts,
            model_candidates=[self.micro_model],
        )
        metrics.llm_calls += 1
        metrics.micro_calls += 1
        metrics.micro_latency_sec = micro_latency
        p_tok, c_tok = _usage_tokens(micro_data)
        metrics.prompt_tokens += p_tok
        metrics.completion_tokens += c_tok

        micro_parsed_resp = parse_agent_response(
            micro_data, micro_tried, micro_latency, self._provider.provider
        )
        micro_parse = parse_classification_payload(micro_parsed_resp.answer)
        reasons = evaluate_fallback(micro_parse, min_confidence=self.min_confidence)
        metrics.fallback_reasons = list(reasons)

        micro_label = micro_parse.payload.label if micro_parse.payload else None
        micro_confidence = micro_parse.payload.confidence if micro_parse.payload else None
        micro_status = micro_parse.payload.status if micro_parse.payload else None

        if not reasons and micro_parse.payload is not None:
            return MicroFirstResult(
                label=micro_parse.payload.label,
                confidence=micro_parse.payload.confidence,
                status=micro_parse.payload.status,
                tier="micro",
                escalated=False,
                model_used=micro_tried,
                micro_model=self.micro_model,
                fallback_model=self.fallback_model,
                micro_label=micro_label,
                micro_confidence=micro_confidence,
                micro_status=micro_status,
                metrics=metrics,
            )

        micro_preview = strip_fences(micro_parsed_resp.answer)[:400]
        fallback_messages = [
            {"role": "system", "content": _FALLBACK_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"User query:\n{cleaned_user}\n\n"
                    f"Micro-model raw output (reasons: {', '.join(reasons)}):\n"
                    f"{micro_preview}\n\n"
                    "Return the final classification JSON."
                ),
            },
        ]
        fallback_data, fallback_tried, fallback_latency = self._provider.complete(
            fallback_messages,
            opts,
            model_candidates=[self.fallback_model],
        )
        metrics.llm_calls += 1
        metrics.fallback_calls += 1
        metrics.fallback_latency_sec = fallback_latency
        fp, fc = _usage_tokens(fallback_data)
        metrics.prompt_tokens += fp
        metrics.completion_tokens += fc

        fallback_parsed_resp = parse_agent_response(
            fallback_data, fallback_tried, fallback_latency, self._provider.provider
        )
        fallback_parse = parse_classification_payload(fallback_parsed_resp.answer)
        if not fallback_parse.ok or fallback_parse.payload is None:
            # Last resort: keep micro if it had a label, else other/UNSURE.
            if micro_parse.payload is not None:
                return MicroFirstResult(
                    label=micro_parse.payload.label,
                    confidence=micro_parse.payload.confidence,
                    status="UNSURE",
                    tier="fallback",
                    escalated=True,
                    model_used=fallback_tried,
                    micro_model=self.micro_model,
                    fallback_model=self.fallback_model,
                    micro_label=micro_label,
                    micro_confidence=micro_confidence,
                    micro_status=micro_status,
                    metrics=metrics,
                )
            return MicroFirstResult(
                label="other",
                confidence=0.0,
                status="FAIL",
                tier="fallback",
                escalated=True,
                model_used=fallback_tried,
                micro_model=self.micro_model,
                fallback_model=self.fallback_model,
                micro_label=micro_label,
                micro_confidence=micro_confidence,
                micro_status=micro_status,
                metrics=metrics,
            )

        return MicroFirstResult(
            label=fallback_parse.payload.label,
            confidence=fallback_parse.payload.confidence,
            status=fallback_parse.payload.status,
            tier="fallback",
            escalated=True,
            model_used=fallback_tried,
            micro_model=self.micro_model,
            fallback_model=self.fallback_model,
            micro_label=micro_label,
            micro_confidence=micro_confidence,
            micro_status=micro_status,
            metrics=metrics,
        )
