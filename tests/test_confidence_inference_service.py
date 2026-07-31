from __future__ import annotations

from typing import Any

from app.models import AgentRequestOptions
from app.services.confidence_inference_service import (
    ConfidenceInferenceService,
    majority_action,
    user_has_task_reference,
    validate_scored_payload,
)


class FakeProviderService:
    """Returns scripted OpenAI-shaped payloads for complete()."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[list[dict[str, str]], AgentRequestOptions]] = []

    def complete(
        self, messages: list[dict[str, str]], options: AgentRequestOptions
    ) -> tuple[dict[str, Any], str, float]:
        self.calls.append((messages, options))
        if not self.answers:
            raise RuntimeError("FakeProviderService: no more scripted answers")
        content = self.answers.pop(0)
        data = {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        return data, "fake-model", 0.01


def test_validate_rejects_empty_user_text() -> None:
    result = validate_scored_payload(
        {"action": "list", "confidence": 0.9, "status": "OK", "rationale": "x"},
        "   ",
    )
    assert result.ok is False
    assert result.reason == "empty_user_text"


def test_validate_rejects_bad_action_and_confidence() -> None:
    bad_action = validate_scored_payload(
        {"action": "delete", "confidence": 0.9, "status": "OK", "rationale": "x"},
        "удали всё",
    )
    assert bad_action.ok is False
    assert bad_action.reason and bad_action.reason.startswith("action_not_allowed")

    bad_conf = validate_scored_payload(
        {"action": "list", "confidence": 1.5, "status": "OK", "rationale": "x"},
        "покажи задачи",
    )
    assert bad_conf.ok is False
    assert bad_conf.reason == "confidence_out_of_range"


def test_validate_complete_requires_task_reference() -> None:
    no_ref = validate_scored_payload(
        {"action": "complete", "confidence": 0.95, "status": "OK", "rationale": "ok"},
        "закрой что-нибудь",
    )
    assert no_ref.ok is False
    assert no_ref.reason == "complete_without_task_reference"

    with_id = validate_scored_payload(
        {"action": "complete", "confidence": 0.95, "status": "OK", "rationale": "ok"},
        "закрой задачу 6h8HWJq8Hg8vVP37",
    )
    assert with_id.ok is True
    assert with_id.scored is not None
    assert with_id.scored.action == "complete"


def test_validate_model_status_fail() -> None:
    result = validate_scored_payload(
        {"action": "refuse", "confidence": 0.2, "status": "FAIL", "rationale": "unsafe"},
        "удали всё",
    )
    assert result.ok is False
    assert result.reason == "model_status_fail"
    assert result.scored is not None


def test_user_has_task_reference() -> None:
    assert user_has_task_reference("закрой задачу 6h8HWJq8Hg8vVP37")
    assert user_has_task_reference('закрой задачу «купить молоко»')
    assert not user_has_task_reference("закрой что-нибудь")


def test_majority_action() -> None:
    assert majority_action(["create", "create", "list"]) == "create"
    assert majority_action(["create", "list", "refuse"]) is None
    assert majority_action(["create", "list"]) is None
    assert majority_action([]) is None


def test_infer_empty_rejects_without_llm() -> None:
    provider = FakeProviderService([])
    service = ConfidenceInferenceService(provider, "fake")
    decision = service.infer_action("  ")
    assert decision.accepted is False
    assert decision.rejected_reason == "empty_user_text"
    assert provider.calls == []


def test_infer_high_confidence_accepts_after_self_check() -> None:
    scoring = (
        '{"action":"create","confidence":0.95,"status":"OK",'
        '"rationale":"explicit create request"}'
    )
    self_check = '{"agree":true,"corrected_action":"create","reason":"ok"}'
    provider = FakeProviderService([scoring, self_check])
    service = ConfidenceInferenceService(provider, "fake")
    decision = service.infer_action("создай задачу купить молоко")
    assert decision.accepted is True
    assert decision.action == "create"
    assert decision.metrics.scoring_calls == 1
    assert decision.metrics.self_check_calls == 1
    assert decision.metrics.redundancy_calls == 0
    assert "scoring" in decision.metrics.pathway
    assert "constraint" in decision.metrics.pathway
    assert "self_check" in decision.metrics.pathway


def test_infer_constraint_rejects_unparseable() -> None:
    provider = FakeProviderService(["not-json-at-all"])
    service = ConfidenceInferenceService(provider, "fake")
    decision = service.infer_action("создай задачу")
    assert decision.accepted is False
    assert decision.rejected_reason == "unparseable_scoring_json"


def test_infer_unsure_goes_to_redundancy_majority() -> None:
    first = (
        '{"action":"create","confidence":0.4,"status":"UNSURE",'
        '"rationale":"ambiguous"}'
    )
    r1 = '{"action":"create","confidence":0.7,"status":"OK","rationale":"a"}'
    r2 = '{"action":"create","confidence":0.6,"status":"OK","rationale":"b"}'
    r3 = '{"action":"list","confidence":0.5,"status":"OK","rationale":"c"}'
    provider = FakeProviderService([first, r1, r2, r3])
    service = ConfidenceInferenceService(provider, "fake", n_redundancy=3)
    decision = service.infer_action("может задачу на молоко?")
    assert decision.accepted is True
    assert decision.action == "create"
    assert "redundancy" in decision.metrics.pathway
    assert decision.metrics.redundancy_calls == 3
    assert decision.metrics.re_inference_count == 3


def test_infer_redundancy_no_majority_rejects() -> None:
    first = (
        '{"action":"create","confidence":0.6,"status":"OK",'
        '"rationale":"mid"}'
    )
    r1 = '{"action":"create","confidence":0.7,"status":"OK","rationale":"a"}'
    r2 = '{"action":"list","confidence":0.6,"status":"OK","rationale":"b"}'
    r3 = '{"action":"refuse","confidence":0.5,"status":"OK","rationale":"c"}'
    provider = FakeProviderService([first, r1, r2, r3])
    service = ConfidenceInferenceService(provider, "fake", n_redundancy=3)
    decision = service.infer_action("что-то с задачами")
    assert decision.accepted is False
    assert decision.rejected_reason == "mid_confidence_no_majority"


def test_infer_self_check_disagree_then_redundancy() -> None:
    scoring = (
        '{"action":"list","confidence":0.9,"status":"OK",'
        '"rationale":"maybe list"}'
    )
    self_check = '{"agree":false,"corrected_action":"create","reason":"actually create"}'
    r1 = '{"action":"create","confidence":0.8,"status":"OK","rationale":"a"}'
    r2 = '{"action":"create","confidence":0.7,"status":"OK","rationale":"b"}'
    r3 = '{"action":"create","confidence":0.75,"status":"OK","rationale":"c"}'
    provider = FakeProviderService([scoring, self_check, r1, r2, r3])
    service = ConfidenceInferenceService(provider, "fake")
    decision = service.infer_action("сделай задачу и потом посмотрим")
    assert decision.accepted is True
    assert decision.action == "create"
    assert decision.metrics.self_check_calls == 1
    assert decision.metrics.redundancy_calls == 3


def test_infer_complete_without_ref_rejected_by_constraint() -> None:
    scoring = (
        '{"action":"complete","confidence":0.99,"status":"OK",'
        '"rationale":"guess"}'
    )
    provider = FakeProviderService([scoring])
    service = ConfidenceInferenceService(provider, "fake")
    decision = service.infer_action("можно закрыть что-нибудь?")
    assert decision.accepted is False
    assert decision.rejected_reason == "complete_without_task_reference"
    assert len(provider.calls) == 1
