from __future__ import annotations

import json
from typing import Any, Optional

from app.models import AgentRequestOptions
from app.services.model_routing_service import (
    ModelRoutingService,
    evaluate_escalation,
    looks_uncertain,
)


class FakeRoutingProvider:
    """Scripted provider that records model_candidates overrides."""

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
            raise RuntimeError("FakeRoutingProvider expects model_candidates")
        model = model_candidates[0]
        queue = self._by_model.get(model)
        if not queue:
            raise RuntimeError(f"no scripted answers for model={model}")
        content = queue.pop(0)
        self.calls.append((model, messages))
        data = {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 12},
        }
        return data, model, 0.02


def test_looks_uncertain_markers() -> None:
    assert looks_uncertain("Я не уверен в ответе") is True
    assert looks_uncertain("I don't know the capital") is True
    assert looks_uncertain("Paris is the capital of France.") is False


def test_evaluate_escalation_heuristics() -> None:
    assert evaluate_escalation(
        answer="ok enough text here for threshold",
        confidence=0.9,
        status="OK",
        finish_reason="stop",
        min_answer_chars=10,
        min_confidence=0.55,
    ) == []

    # Short + high confidence OK → do not escalate on length alone.
    assert evaluate_escalation(
        answer="51",
        confidence=1.0,
        status="OK",
        finish_reason="stop",
        min_answer_chars=24,
        min_confidence=0.55,
    ) == []

    reasons = evaluate_escalation(
        answer="x",
        confidence=0.2,
        status="UNSURE",
        finish_reason="length",
        min_answer_chars=24,
        min_confidence=0.55,
    )
    assert "short_answer" in reasons
    assert "low_confidence" in reasons
    assert "status_unsure" in reasons
    assert "truncated" in reasons


def test_route_stays_on_cheap_when_confident() -> None:
    cheap_payload = json.dumps(
        {
            "answer": "Paris is the capital of France.",
            "confidence": 0.96,
            "status": "OK",
        },
        ensure_ascii=False,
    )
    provider = FakeRoutingProvider(
        {
            "cheap-model": [cheap_payload],
            "strong-model": ["SHOULD_NOT_BE_CALLED"],
        }
    )
    service = ModelRoutingService(provider, "cheap-model", "strong-model")
    result = service.route("What is the capital of France?")
    assert result.escalated is False
    assert result.tier == "cheap"
    assert result.model_used == "cheap-model"
    assert "Paris" in result.answer
    assert result.confidence == 0.96
    assert result.metrics.llm_calls == 1
    assert [c[0] for c in provider.calls] == ["cheap-model"]


def test_route_escalates_on_low_confidence() -> None:
    cheap_payload = json.dumps(
        {
            "answer": "Maybe somewhere in Europe?",
            "confidence": 0.3,
            "status": "UNSURE",
        },
        ensure_ascii=False,
    )
    provider = FakeRoutingProvider(
        {
            "cheap-model": [cheap_payload],
            "strong-model": ["Paris is the capital of France."],
        }
    )
    service = ModelRoutingService(provider, "cheap-model", "strong-model")
    result = service.route("What is the capital of France?")
    assert result.escalated is True
    assert result.tier == "strong"
    assert result.model_used == "strong-model"
    assert "Paris" in result.answer
    assert "low_confidence" in result.metrics.escalate_reasons
    assert "status_unsure" in result.metrics.escalate_reasons
    assert result.metrics.llm_calls == 2
    assert [c[0] for c in provider.calls] == ["cheap-model", "strong-model"]


def test_route_escalates_on_short_answer() -> None:
    # Short without high OK confidence still escalates.
    cheap_payload = json.dumps(
        {"answer": "ok", "confidence": 0.6, "status": "OK"},
        ensure_ascii=False,
    )
    provider = FakeRoutingProvider(
        {
            "cheap-model": [cheap_payload],
            "strong-model": ["A detailed strong-model answer that is long enough."],
        }
    )
    service = ModelRoutingService(
        provider, "cheap-model", "strong-model", min_answer_chars=24
    )
    result = service.route("Explain photosynthesis briefly")
    assert result.escalated is True
    assert "short_answer" in result.metrics.escalate_reasons
    assert result.model_used == "strong-model"


def test_route_escalates_on_uncertain_markers_without_json() -> None:
    provider = FakeRoutingProvider(
        {
            "cheap-model": ["I don't know how to solve this."],
            "strong-model": ["Use binary search on the sorted array."],
        }
    )
    service = ModelRoutingService(provider, "cheap-model", "strong-model")
    result = service.route("How do I find an element in a sorted list?")
    assert result.escalated is True
    assert "uncertain_markers" in result.metrics.escalate_reasons
    assert "binary search" in result.answer.lower()
