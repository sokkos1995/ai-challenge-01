import json
from dataclasses import dataclass
from typing import Optional

from app.models import AgentRequestOptions, AgentResponse
from app.response_parser import parse_agent_response
from app.services.provider_service import ProviderService


@dataclass
class InvariantConflict:
    violated_invariants: list[str]
    explanation: str
    safe_alternative: str
    raw_data: dict
    model: str
    latency_sec: float


class InvariantGuardService:
    def __init__(self, provider_service: ProviderService, provider_name: str) -> None:
        self._provider_service = provider_service
        self._provider_name = provider_name

    @staticmethod
    def _guard_options() -> AgentRequestOptions:
        return AgentRequestOptions(
            temperature=0.0,
            top_p=None,
            top_k=None,
            response_format=None,
            max_output_tokens=220,
            stop_sequences=[],
            finish_instruction=None,
            count_tokens=False,
        )

    @staticmethod
    def _extract_json_object(text: str) -> Optional[dict]:
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

    def check_request(self, user_request: str, invariants: list[str]) -> Optional[InvariantConflict]:
        clean_invariants = [item.strip() for item in invariants if item.strip()]
        if not clean_invariants:
            return None

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an invariant guard for an engineering assistant.\n"
                    "Check whether fulfilling the user's request would violate any invariant.\n"
                    "Return JSON only with keys:\n"
                    "- conflict: boolean\n"
                    "- violated_invariants: array of strings copied exactly from the invariant list\n"
                    "- explanation: short string in the same language as the user request\n"
                    "- safe_alternative: short string in the same language as the user request\n"
                    "Mark conflict=true only when complying with the request would require violating an invariant."
                ),
            },
            {
                "role": "user",
                "content": "\n".join(
                    [
                        "Invariants:",
                        *[f"- {item}" for item in clean_invariants],
                        "",
                        "User request:",
                        user_request.strip(),
                    ]
                ),
            },
        ]

        data, tried_model, elapsed = self._provider_service.complete(messages, self._guard_options())
        response = parse_agent_response(data, tried_model, elapsed, self._provider_name)
        payload = self._extract_json_object(response.answer)
        if payload is None or not bool(payload.get("conflict")):
            return None

        violated = payload.get("violated_invariants", [])
        if not isinstance(violated, list):
            violated = []

        explanation = str(payload.get("explanation", "")).strip()
        safe_alternative = str(payload.get("safe_alternative", "")).strip()
        return InvariantConflict(
            violated_invariants=[str(item).strip() for item in violated if str(item).strip()],
            explanation=explanation,
            safe_alternative=safe_alternative,
            raw_data=payload,
            model=tried_model,
            latency_sec=elapsed,
        )

    def build_refusal_response(self, conflict: InvariantConflict) -> AgentResponse:
        lines = ["Не могу предложить это решение: оно нарушает сохраненные инварианты."]
        if conflict.violated_invariants:
            lines.append("Конфликтующий инвариант:")
            lines.extend(f"- {item}" for item in conflict.violated_invariants)
        if conflict.explanation:
            lines.append(conflict.explanation)
        if conflict.safe_alternative:
            lines.append(f"Могу предложить безопасную альтернативу: {conflict.safe_alternative}")

        return AgentResponse(
            answer="\n".join(lines),
            raw_data=conflict.raw_data,
            model=conflict.model,
            latency_sec=conflict.latency_sec,
            provider=self._provider_name,
        )
