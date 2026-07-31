from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from app.models import AgentRequestOptions
from app.response_parser import parse_agent_response
from app.services.provider_service import ProviderService

ALLOWED_ACTIONS: frozenset[str] = frozenset({"create", "complete", "list", "refuse"})
ALLOWED_STATUSES: frozenset[str] = frozenset({"OK", "UNSURE", "FAIL"})

CONFIDENCE_HIGH = 0.85
CONFIDENCE_MID = 0.55
N_REDUNDANCY_SAMPLES = 3

# complete requires a concrete task reference (id-like token or quoted/named phrase).
_TASK_ID_RE = re.compile(r"\b[0-9a-zA-Z]{8,}\b")
_TASK_NAME_HINT_RE = re.compile(
    r"(задач[уиеа]|task)\s*[«\"'].+?[»\"']|(задач[уиеа]|task)\s+\S+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ScoredAction:
    action: str
    confidence: float
    status: str
    rationale: str


@dataclass
class InferenceMetrics:
    llm_calls: int = 0
    scoring_calls: int = 0
    self_check_calls: int = 0
    redundancy_calls: int = 0
    re_inference_count: int = 0
    latency_sec_total: float = 0.0
    baseline_latency_sec: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    pathway: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "llm_calls": self.llm_calls,
            "scoring_calls": self.scoring_calls,
            "self_check_calls": self.self_check_calls,
            "redundancy_calls": self.redundancy_calls,
            "re_inference_count": self.re_inference_count,
            "latency_sec_total": round(self.latency_sec_total, 4),
            "baseline_latency_sec": round(self.baseline_latency_sec, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "pathway": list(self.pathway),
        }


@dataclass(frozen=True)
class ConfidenceDecision:
    accepted: bool
    action: Optional[str]
    rejected_reason: Optional[str]
    scored: Optional[ScoredAction]
    metrics: InferenceMetrics


@dataclass(frozen=True)
class ConstraintResult:
    ok: bool
    reason: Optional[str] = None
    scored: Optional[ScoredAction] = None


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


def user_has_task_reference(user_text: str) -> bool:
    text = user_text.strip()
    if not text:
        return False
    if _TASK_ID_RE.search(text):
        return True
    if _TASK_NAME_HINT_RE.search(text):
        return True
    return False


def validate_scored_payload(payload: dict, user_text: str) -> ConstraintResult:
    """Constraint-based checks: format, allowed values, logical invariants."""
    if not user_text.strip():
        return ConstraintResult(ok=False, reason="empty_user_text")

    action_raw = payload.get("action")
    status_raw = payload.get("status")
    confidence_raw = payload.get("confidence")
    rationale = str(payload.get("rationale") or "").strip()

    if not isinstance(action_raw, str):
        return ConstraintResult(ok=False, reason="missing_or_invalid_action")
    action = action_raw.strip().lower()
    if action not in ALLOWED_ACTIONS:
        return ConstraintResult(ok=False, reason=f"action_not_allowed:{action}")

    if not isinstance(status_raw, str):
        return ConstraintResult(ok=False, reason="missing_or_invalid_status")
    status = status_raw.strip().upper()
    if status not in ALLOWED_STATUSES:
        return ConstraintResult(ok=False, reason=f"status_not_allowed:{status}")

    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        return ConstraintResult(ok=False, reason="invalid_confidence")
    if confidence < 0.0 or confidence > 1.0:
        return ConstraintResult(ok=False, reason="confidence_out_of_range")

    if action == "complete" and not user_has_task_reference(user_text):
        return ConstraintResult(ok=False, reason="complete_without_task_reference")

    if status == "FAIL":
        return ConstraintResult(
            ok=False,
            reason="model_status_fail",
            scored=ScoredAction(action=action, confidence=confidence, status=status, rationale=rationale),
        )

    return ConstraintResult(
        ok=True,
        scored=ScoredAction(action=action, confidence=confidence, status=status, rationale=rationale),
    )


def majority_action(actions: list[str]) -> Optional[str]:
    if not actions:
        return None
    counts = Counter(actions)
    top_action, top_count = counts.most_common(1)[0]
    if top_count < (len(actions) // 2) + 1:
        return None
    tied = [a for a, c in counts.items() if c == top_count]
    if len(tied) > 1:
        return None
    return top_action


class ConfidenceInferenceService:
    """
    High-stakes Todoist action classification with confidence gate.

    Approaches (all used in one pipeline):
    1. Scoring — model returns action + confidence + status
    2. Constraint-based — enum / format / invariants
    3. Self-check — second pass verifies the answer
    4. Redundancy — N independent scorings + majority vote
    """

    def __init__(
        self,
        provider_service: ProviderService,
        provider_name: str,
        *,
        confidence_high: float = CONFIDENCE_HIGH,
        confidence_mid: float = CONFIDENCE_MID,
        n_redundancy: int = N_REDUNDANCY_SAMPLES,
    ) -> None:
        self._provider_service = provider_service
        self._provider_name = provider_name
        self._confidence_high = confidence_high
        self._confidence_mid = confidence_mid
        self._n_redundancy = max(2, n_redundancy)

    @staticmethod
    def _scoring_options(*, temperature: float = 0.0) -> AgentRequestOptions:
        return AgentRequestOptions(
            temperature=temperature,
            top_p=None,
            top_k=None,
            response_format=None,
            max_output_tokens=220,
            stop_sequences=[],
            finish_instruction=None,
            count_tokens=True,
        )

    def _usage_tokens(self, raw_data: dict) -> tuple[int, int]:
        usage = raw_data.get("usage") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        return prompt, completion

    def _complete(
        self,
        messages: list[dict[str, str]],
        metrics: InferenceMetrics,
        *,
        temperature: float = 0.0,
    ) -> tuple[str, float]:
        data, tried_model, elapsed = self._provider_service.complete(
            messages, self._scoring_options(temperature=temperature)
        )
        response = parse_agent_response(data, tried_model, elapsed, self._provider_name)
        metrics.llm_calls += 1
        metrics.latency_sec_total += elapsed
        prompt_t, completion_t = self._usage_tokens(response.raw_data if isinstance(response.raw_data, dict) else {})
        metrics.prompt_tokens += prompt_t
        metrics.completion_tokens += completion_t
        return response.answer, elapsed

    def _score_once(
        self,
        user_text: str,
        metrics: InferenceMetrics,
        *,
        temperature: float = 0.0,
        track_as_redundancy: bool = False,
    ) -> tuple[Optional[dict], float]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You classify Todoist agent actions for a high-stakes tool gateway.\n"
                    "Allowed actions: create, complete, list, refuse.\n"
                    "Rules:\n"
                    "- create: user wants a new task\n"
                    "- complete: user wants to close a specific existing task\n"
                    "- list: user wants to see tasks\n"
                    "- refuse: unsafe, ambiguous, destructive, or unsupported request\n"
                    "Return JSON only with keys:\n"
                    '- action: "create"|"complete"|"list"|"refuse"\n'
                    "- confidence: number from 0 to 1\n"
                    '- status: "OK"|"UNSURE"|"FAIL"\n'
                    "- rationale: short explanation\n"
                    "Use FAIL when the request cannot be safely mapped. "
                    "Use UNSURE when ambiguous."
                ),
            },
            {"role": "user", "content": user_text.strip()},
        ]
        answer, elapsed = self._complete(messages, metrics, temperature=temperature)
        metrics.scoring_calls += 1
        if track_as_redundancy:
            metrics.redundancy_calls += 1
            metrics.re_inference_count += 1
        return extract_json_object(answer), elapsed

    def _self_check(
        self,
        user_text: str,
        scored: ScoredAction,
        metrics: InferenceMetrics,
    ) -> tuple[bool, Optional[str]]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You verify a previous Todoist action classification.\n"
                    "Return JSON only with keys:\n"
                    "- agree: boolean\n"
                    '- corrected_action: "create"|"complete"|"list"|"refuse"\n'
                    "- reason: short string\n"
                    "Set agree=true only if the proposed action is safe and correct."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_request": user_text.strip(),
                        "proposed": {
                            "action": scored.action,
                            "confidence": scored.confidence,
                            "status": scored.status,
                            "rationale": scored.rationale,
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        answer, _elapsed = self._complete(messages, metrics, temperature=0.0)
        metrics.self_check_calls += 1
        metrics.re_inference_count += 1
        payload = extract_json_object(answer)
        if payload is None:
            return False, None
        agree = bool(payload.get("agree"))
        corrected = str(payload.get("corrected_action") or "").strip().lower()
        if corrected and corrected not in ALLOWED_ACTIONS:
            return False, None
        if not agree:
            return False, corrected or None
        if corrected and corrected != scored.action:
            return False, corrected
        return True, scored.action

    def _redundancy_vote(
        self,
        user_text: str,
        metrics: InferenceMetrics,
    ) -> tuple[Optional[str], list[str]]:
        metrics.pathway.append("redundancy")
        actions: list[str] = []
        for _ in range(self._n_redundancy):
            payload, _elapsed = self._score_once(
                user_text,
                metrics,
                temperature=0.3,
                track_as_redundancy=True,
            )
            if payload is None:
                continue
            constrained = validate_scored_payload(payload, user_text)
            if not constrained.ok or constrained.scored is None:
                continue
            if constrained.scored.status == "FAIL":
                continue
            actions.append(constrained.scored.action)
        winner = majority_action(actions)
        return winner, actions

    def infer_action(self, user_text: str) -> ConfidenceDecision:
        metrics = InferenceMetrics()
        text = user_text if isinstance(user_text, str) else ""

        if not text.strip():
            metrics.pathway.append("constraint")
            return ConfidenceDecision(
                accepted=False,
                action=None,
                rejected_reason="empty_user_text",
                scored=None,
                metrics=metrics,
            )

        # 1) Scoring
        metrics.pathway.append("scoring")
        payload, baseline_elapsed = self._score_once(text, metrics, temperature=0.0)
        metrics.baseline_latency_sec = baseline_elapsed

        if payload is None:
            metrics.pathway.append("constraint")
            return ConfidenceDecision(
                accepted=False,
                action=None,
                rejected_reason="unparseable_scoring_json",
                scored=None,
                metrics=metrics,
            )

        # 2) Constraint-based
        metrics.pathway.append("constraint")
        constrained = validate_scored_payload(payload, text)
        if not constrained.ok or constrained.scored is None:
            return ConfidenceDecision(
                accepted=False,
                action=constrained.scored.action if constrained.scored else None,
                rejected_reason=constrained.reason or "constraint_failed",
                scored=constrained.scored,
                metrics=metrics,
            )

        scored = constrained.scored

        # Low confidence → redundancy (or reject if below mid after vote)
        if scored.confidence < self._confidence_mid or scored.status == "UNSURE":
            winner, _votes = self._redundancy_vote(text, metrics)
            if winner is None:
                return ConfidenceDecision(
                    accepted=False,
                    action=None,
                    rejected_reason="redundancy_no_majority",
                    scored=scored,
                    metrics=metrics,
                )
            return ConfidenceDecision(
                accepted=True,
                action=winner,
                rejected_reason=None,
                scored=scored,
                metrics=metrics,
            )

        # High confidence OK → self-check
        if scored.confidence >= self._confidence_high and scored.status == "OK":
            metrics.pathway.append("self_check")
            agree, corrected = self._self_check(text, scored, metrics)
            if agree:
                return ConfidenceDecision(
                    accepted=True,
                    action=scored.action,
                    rejected_reason=None,
                    scored=scored,
                    metrics=metrics,
                )
            winner, _votes = self._redundancy_vote(text, metrics)
            if winner is None:
                return ConfidenceDecision(
                    accepted=False,
                    action=corrected,
                    rejected_reason="self_check_failed_no_majority",
                    scored=scored,
                    metrics=metrics,
                )
            return ConfidenceDecision(
                accepted=True,
                action=winner,
                rejected_reason=None,
                scored=scored,
                metrics=metrics,
            )

        # Mid band → redundancy
        winner, _votes = self._redundancy_vote(text, metrics)
        if winner is None:
            return ConfidenceDecision(
                accepted=False,
                action=None,
                rejected_reason="mid_confidence_no_majority",
                scored=scored,
                metrics=metrics,
            )
        return ConfidenceDecision(
            accepted=True,
            action=winner,
            rejected_reason=None,
            scored=scored,
            metrics=metrics,
        )
