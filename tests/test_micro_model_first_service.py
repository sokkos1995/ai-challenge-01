from __future__ import annotations

import json
from typing import Any, Optional

from app.models import AgentRequestOptions
from app.services.micro_model_first_service import (
    MicroModelFirstService,
    evaluate_fallback,
    parse_classification_payload,
)


class FakeMicroProvider:
    def __init__(self, by_model: dict[str, list[str]], *, provider: str = "fake") -> None:
        self.provider = provider
        self._by_model = {k: list(v) for k, v in by_model.items()}
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        options: AgentRequestOptions,
        *,
        model_candidates: Optional[list[str]] = None,
    ) -> tuple[dict[str, Any], str, float]:
        if not model_candidates:
            raise RuntimeError("model_candidates required")
        model = model_candidates[0]
        queue = self._by_model.get(model)
        if not queue:
            raise RuntimeError(f"no scripted answers for model={model}")
        content = queue.pop(0)
        self.calls.append((model, messages))
        data = {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 9},
        }
        return data, model, 0.02


def test_parse_classification_payload_ok() -> None:
    parsed = parse_classification_payload(
        '{"label":"faq","confidence":0.9,"status":"OK"}'
    )
    assert parsed.ok is True
    assert parsed.payload is not None
    assert parsed.payload.label == "faq"
    assert parsed.payload.confidence == 0.9
    assert parsed.payload.status == "OK"


def test_parse_classification_payload_invalid_label() -> None:
    parsed = parse_classification_payload(
        '{"label":"unknown","confidence":0.9,"status":"OK"}'
    )
    assert parsed.ok is False
    assert parsed.reason == "invalid_label:unknown"


def test_evaluate_fallback_reasons() -> None:
    ok = parse_classification_payload('{"label":"faq","confidence":0.9,"status":"OK"}')
    assert evaluate_fallback(ok, min_confidence=0.6) == []

    unsure = parse_classification_payload(
        '{"label":"faq","confidence":0.4,"status":"UNSURE"}'
    )
    reasons = evaluate_fallback(unsure, min_confidence=0.6)
    assert "status_unsure" in reasons
    assert "low_confidence" in reasons

    bad = parse_classification_payload("not-json")
    assert evaluate_fallback(bad, min_confidence=0.6) == ["invalid_format"]


def test_micro_accepts_confident_ok() -> None:
    provider = FakeMicroProvider(
        {
            "micro": [
                json.dumps({"label": "chitchat", "confidence": 0.95, "status": "OK"}),
            ]
        }
    )
    service = MicroModelFirstService(provider, "micro", "fallback", min_confidence=0.6)
    result = service.classify("Hello!")
    assert result.escalated is False
    assert result.tier == "micro"
    assert result.label == "chitchat"
    assert result.metrics.llm_calls == 1
    assert result.metrics.fallback_calls == 0
    assert len(provider.calls) == 1


def test_micro_escalates_on_unsure() -> None:
    provider = FakeMicroProvider(
        {
            "micro": [
                json.dumps({"label": "other", "confidence": 0.3, "status": "UNSURE"}),
            ],
            "fallback": [
                json.dumps({"label": "code_help", "confidence": 0.9, "status": "OK"}),
            ],
        }
    )
    service = MicroModelFirstService(provider, "micro", "fallback", min_confidence=0.6)
    result = service.classify("pls fx my pythn scriipt")
    assert result.escalated is True
    assert result.tier == "fallback"
    assert result.label == "code_help"
    assert result.metrics.llm_calls == 2
    assert result.metrics.fallback_calls == 1
    assert "status_unsure" in result.metrics.fallback_reasons
    assert [c[0] for c in provider.calls] == ["micro", "fallback"]


def test_micro_escalates_on_invalid_format() -> None:
    provider = FakeMicroProvider(
        {
            "micro": ["totally broken"],
            "fallback": [
                json.dumps({"label": "faq", "confidence": 0.8, "status": "OK"}),
            ],
        }
    )
    service = MicroModelFirstService(provider, "micro", "fallback")
    result = service.classify("What is HTTP 404?")
    assert result.escalated is True
    assert result.label == "faq"
    assert result.metrics.fallback_reasons == ["invalid_format"]
