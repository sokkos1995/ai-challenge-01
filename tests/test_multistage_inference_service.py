from __future__ import annotations

import json
from typing import Any, Optional

from app.models import AgentRequestOptions
from app.services.multistage_inference_service import (
    MultistageInferenceService,
    decide_by_rules,
    parse_compact_kv,
    validate_triage_payload,
)


class FakeMultistageProvider:
    """Scripted provider that records model_candidates and returns queued answers."""

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
            raise RuntimeError("FakeMultistageProvider expects model_candidates")
        model = model_candidates[0]
        queue = self._by_model.get(model)
        if not queue:
            raise RuntimeError(f"no scripted answers for model={model}")
        content = queue.pop(0)
        self.calls.append((model, messages))
        data = {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }
        return data, model, 0.01


def test_parse_compact_kv() -> None:
    parsed = parse_compact_kv(
        "lang=ru;noise=1;product_hint=payments;core=card declined twice"
    )
    assert parsed["lang"] == "ru"
    assert parsed["noise"] == "1"
    assert parsed["product_hint"] == "payments"
    assert parsed["core"] == "card declined twice"


def test_parse_compact_kv_picks_line_from_prose() -> None:
    text = "Here you go:\nlang=en;noise=0;product_hint=auth;core=cannot login\nThanks"
    parsed = parse_compact_kv(text)
    assert parsed["lang"] == "en"
    assert parsed["product_hint"] == "auth"


def test_decide_by_rules() -> None:
    assert decide_by_rules("spam", "low") == "reject"
    assert decide_by_rules("bug", "critical") == "escalate"
    assert decide_by_rules("billing", "high") == "escalate"
    assert decide_by_rules("billing", "low") == "auto_reply"
    assert decide_by_rules("access", "medium") == "queue"


def test_validate_triage_rejects_unknown_enum() -> None:
    result = validate_triage_payload(
        {
            "intent": "not_a_real_intent",
            "urgency": "low",
            "product": "payments",
            "decision": "queue",
            "summary": "x",
        }
    )
    assert result.ok is False
    assert result.reason and result.reason.startswith("invalid_intent")


def test_validate_spam_must_reject() -> None:
    result = validate_triage_payload(
        {
            "intent": "spam",
            "urgency": "low",
            "product": "unknown",
            "decision": "queue",
            "summary": "promo blast",
        }
    )
    assert result.ok is False
    assert result.reason == "decision_rule_spam_must_reject"


def test_run_monolithic_one_call() -> None:
    payload = {
        "intent": "billing",
        "urgency": "low",
        "product": "payments",
        "decision": "auto_reply",
        "summary": "Refund FAQ for declined card",
    }
    provider = FakeMultistageProvider(
        {
            "strong": [json.dumps(payload, ensure_ascii=False)],
            "cheap": ["SHOULD_NOT_BE_CALLED"],
        }
    )
    service = MultistageInferenceService(provider, "cheap", "strong")
    result = service.run_monolithic("How do I get a refund for a declined card?")
    assert result.ok is True
    assert result.mode == "monolithic"
    assert result.fields is not None
    assert result.fields.intent == "billing"
    assert result.fields.decision == "auto_reply"
    assert result.metrics.llm_calls == 1
    assert len(provider.calls) == 1
    assert provider.calls[0][0] == "strong"


def test_run_multistage_three_calls() -> None:
    provider = FakeMultistageProvider(
        {
            "cheap": [
                "lang=en;noise=1;product_hint=api;core=500 errors on checkout",
                "intent=bug;urgency=critical;product=api",
            ],
            "strong": [
                "decision=escalate;summary=Critical API checkout 500s",
            ],
        }
    )
    service = MultistageInferenceService(provider, "cheap", "strong")
    result = service.run_multistage(
        "URGENT!!! api /checkout keeps returning 500 plz help ASAP!!!"
    )
    assert result.ok is True
    assert result.mode == "multistage"
    assert result.fields is not None
    assert result.fields.intent == "bug"
    assert result.fields.urgency == "critical"
    assert result.fields.decision == "escalate"
    assert result.metrics.llm_calls == 3
    assert [c[0] for c in provider.calls] == ["cheap", "cheap", "strong"]
    assert [s.name for s in result.metrics.stages] == ["normalize", "classify", "format"]


def test_run_multistage_rejects_bad_stage2_enum() -> None:
    provider = FakeMultistageProvider(
        {
            "cheap": [
                "lang=ru;noise=0;product_hint=auth;core=не могу войти",
                "intent=login_fail;urgency=medium;product=auth",
            ],
            "strong": ["decision=queue;summary=x"],
        }
    )
    service = MultistageInferenceService(provider, "cheap", "strong")
    result = service.run_multistage("не могу войти в аккаунт")
    assert result.ok is False
    assert result.error and "stage2:invalid_intent" in result.error
    assert result.metrics.llm_calls == 2


def test_empty_input() -> None:
    provider = FakeMultistageProvider({"cheap": [], "strong": []})
    service = MultistageInferenceService(provider, "cheap", "strong")
    mono = service.run_monolithic("   ")
    multi = service.run_multistage("")
    assert mono.ok is False and mono.error == "empty_input"
    assert multi.ok is False and multi.error == "empty_input"
    assert mono.metrics.llm_calls == 0
    assert multi.metrics.llm_calls == 0
