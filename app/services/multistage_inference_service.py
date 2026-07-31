from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

from app.models import AgentRequestOptions
from app.response_parser import parse_agent_response

ALLOWED_INTENTS: frozenset[str] = frozenset(
    {"billing", "bug", "access", "feature", "spam", "other"}
)
ALLOWED_URGENCIES: frozenset[str] = frozenset({"low", "medium", "high", "critical"})
ALLOWED_PRODUCTS: frozenset[str] = frozenset(
    {"payments", "auth", "api", "web", "mobile", "unknown"}
)
ALLOWED_DECISIONS: frozenset[str] = frozenset(
    {"auto_reply", "queue", "escalate", "reject"}
)
ALLOWED_LANGS: frozenset[str] = frozenset({"ru", "en", "mixed", "other"})

_MARKDOWN_FENCE_RE = re.compile(r"^```(?:json|text)?\s*|\s*```$", re.IGNORECASE)

_MONO_SYSTEM = (
    "You triage support tickets. Reply with a single JSON object only, no markdown:\n"
    '{"intent":"billing|bug|access|feature|spam|other",'
    '"urgency":"low|medium|high|critical",'
    '"product":"payments|auth|api|web|mobile|unknown",'
    '"decision":"auto_reply|queue|escalate|reject",'
    '"summary":"<one short operator phrase>"}\n'
    "Decision rules (must follow):\n"
    "- spam -> reject\n"
    "- critical + bug -> escalate\n"
    "- urgency high|critical (and not spam) -> escalate unless clearly a FAQ billing question "
    "that can auto_reply\n"
    "- clear low-urgency billing/feature FAQ -> auto_reply\n"
    "- otherwise -> queue\n"
)

_STAGE1_SYSTEM = (
    "Normalize a support ticket. Reply with ONE compact line only, no markdown:\n"
    "lang=ru|en|mixed|other;noise=0|1;product_hint=payments|auth|api|web|mobile|unknown;"
    "core=<short cleaned gist without fluff>\n"
    "noise=1 if typos, emoji spam, or mixed garbage; else 0."
)

_STAGE2_SYSTEM = (
    "Classify a normalized ticket. Reply with ONE compact line only, no markdown:\n"
    "intent=billing|bug|access|feature|spam|other;"
    "urgency=low|medium|high|critical;"
    "product=payments|auth|api|web|mobile|unknown\n"
    "Use only the allowed enum values."
)

_STAGE3_SYSTEM = (
    "Decide triage action. Reply with ONE compact line only, no markdown:\n"
    "decision=auto_reply|queue|escalate|reject;summary=<one short operator phrase>\n"
    "Hard rules: spam->reject; critical+bug->escalate; "
    "high|critical (non-spam)->prefer escalate; clear low billing/feature FAQ->auto_reply; "
    "else queue."
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
class InferenceStageMetrics:
    name: str
    model: str
    latency_sec: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": self.model,
            "latency_sec": round(self.latency_sec, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "raw": self.raw,
        }


@dataclass
class MultistageMetrics:
    llm_calls: int = 0
    latency_sec_total: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    models_used: list[str] = field(default_factory=list)
    stages: list[InferenceStageMetrics] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "llm_calls": self.llm_calls,
            "latency_sec_total": round(self.latency_sec_total, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "models_used": list(self.models_used),
            "stages": [s.to_dict() for s in self.stages],
        }


@dataclass(frozen=True)
class TicketTriageFields:
    intent: str
    urgency: str
    product: str
    decision: str
    summary: str


@dataclass(frozen=True)
class TicketTriageResult:
    ok: bool
    mode: str
    fields: Optional[TicketTriageFields]
    error: Optional[str]
    metrics: MultistageMetrics
    cheap_model: str
    strong_model: str

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "ok": self.ok,
            "mode": self.mode,
            "error": self.error,
            "cheap_model": self.cheap_model,
            "strong_model": self.strong_model,
            "metrics": self.metrics.to_dict(),
        }
        if self.fields is not None:
            payload["intent"] = self.fields.intent
            payload["urgency"] = self.fields.urgency
            payload["product"] = self.fields.product
            payload["decision"] = self.fields.decision
            payload["summary"] = self.fields.summary
        return payload


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    fields: Optional[TicketTriageFields] = None
    reason: Optional[str] = None


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


def parse_compact_kv(text: str) -> dict[str, str]:
    """Parse compact `k=v;k2=v2` lines. Values may contain `=` but not `;`."""
    clean = strip_fences(text)
    # If model wrapped compact line in prose, take the densest k=v; line.
    if "\n" in clean:
        lines = [ln.strip() for ln in clean.splitlines() if "=" in ln]
        if lines:
            clean = max(lines, key=lambda ln: ln.count("="))

    result: dict[str, str] = {}
    for part in clean.split(";"):
        chunk = part.strip()
        if not chunk or "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key:
            result[key] = value
    return result


def format_compact_kv(data: dict[str, str], keys: list[str]) -> str:
    parts: list[str] = []
    for key in keys:
        if key in data:
            parts.append(f"{key}={data[key]}")
    return ";".join(parts)


def decide_by_rules(intent: str, urgency: str) -> str:
    """Deterministic decision policy used for validation / correction hints."""
    if intent == "spam":
        return "reject"
    if urgency == "critical" and intent == "bug":
        return "escalate"
    if urgency in {"high", "critical"} and intent != "spam":
        return "escalate"
    if urgency == "low" and intent in {"billing", "feature"}:
        return "auto_reply"
    return "queue"


def validate_triage_payload(
    payload: dict,
    *,
    enforce_decision_rules: bool = True,
) -> ValidationResult:
    intent = str(payload.get("intent") or "").strip().lower()
    urgency = str(payload.get("urgency") or "").strip().lower()
    product = str(payload.get("product") or "").strip().lower()
    decision = str(payload.get("decision") or "").strip().lower()
    summary = str(payload.get("summary") or "").strip()

    if intent not in ALLOWED_INTENTS:
        return ValidationResult(ok=False, reason=f"invalid_intent:{intent or '?'}")
    if urgency not in ALLOWED_URGENCIES:
        return ValidationResult(ok=False, reason=f"invalid_urgency:{urgency or '?'}")
    if product not in ALLOWED_PRODUCTS:
        return ValidationResult(ok=False, reason=f"invalid_product:{product or '?'}")
    if decision not in ALLOWED_DECISIONS:
        return ValidationResult(ok=False, reason=f"invalid_decision:{decision or '?'}")
    if not summary:
        return ValidationResult(ok=False, reason="empty_summary")

    if enforce_decision_rules:
        expected = decide_by_rules(intent, urgency)
        # Allow escalate when rules say escalate; reject must match spam.
        if intent == "spam" and decision != "reject":
            return ValidationResult(ok=False, reason="decision_rule_spam_must_reject")
        if expected == "escalate" and decision not in {"escalate", "queue"}:
            # tolerate queue only when model is more cautious; escalate is preferred
            if decision == "auto_reply":
                return ValidationResult(ok=False, reason="decision_rule_high_risk_no_auto")
        if expected == "reject" and decision != "reject":
            return ValidationResult(ok=False, reason="decision_rule_reject_mismatch")

    return ValidationResult(
        ok=True,
        fields=TicketTriageFields(
            intent=intent,
            urgency=urgency,
            product=product,
            decision=decision,
            summary=summary,
        ),
    )


def validate_stage1(data: dict[str, str]) -> tuple[bool, Optional[str]]:
    lang = (data.get("lang") or "").strip().lower()
    noise = (data.get("noise") or "").strip()
    product_hint = (data.get("product_hint") or "").strip().lower()
    core = (data.get("core") or "").strip()
    if lang not in ALLOWED_LANGS:
        return False, f"invalid_lang:{lang or '?'}"
    if noise not in {"0", "1"}:
        return False, f"invalid_noise:{noise or '?'}"
    if product_hint not in ALLOWED_PRODUCTS:
        return False, f"invalid_product_hint:{product_hint or '?'}"
    if not core:
        return False, "empty_core"
    return True, None


def validate_stage2(data: dict[str, str]) -> tuple[bool, Optional[str]]:
    intent = (data.get("intent") or "").strip().lower()
    urgency = (data.get("urgency") or "").strip().lower()
    product = (data.get("product") or "").strip().lower()
    if intent not in ALLOWED_INTENTS:
        return False, f"invalid_intent:{intent or '?'}"
    if urgency not in ALLOWED_URGENCIES:
        return False, f"invalid_urgency:{urgency or '?'}"
    if product not in ALLOWED_PRODUCTS:
        return False, f"invalid_product:{product or '?'}"
    return True, None


def validate_stage3(data: dict[str, str]) -> tuple[bool, Optional[str]]:
    decision = (data.get("decision") or "").strip().lower()
    summary = (data.get("summary") or "").strip()
    if decision not in ALLOWED_DECISIONS:
        return False, f"invalid_decision:{decision or '?'}"
    if not summary:
        return False, "empty_summary"
    return True, None


def _usage_tokens(data: dict) -> tuple[int, int]:
    usage = data.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return prompt, completion


def _default_options(*, max_tokens: int) -> AgentRequestOptions:
    return AgentRequestOptions(
        temperature=0.1,
        top_p=None,
        top_k=None,
        response_format=None,
        max_output_tokens=max_tokens,
        stop_sequences=[],
        finish_instruction=None,
    )


class MultistageInferenceService:
    """
    Support-ticket triage: monolithic (1 call) vs multi-stage (normalize→classify→format).
    """

    def __init__(
        self,
        provider: SupportsComplete,
        cheap_model: str,
        strong_model: str,
    ) -> None:
        if not cheap_model.strip():
            raise ValueError("cheap_model must not be empty")
        if not strong_model.strip():
            raise ValueError("strong_model must not be empty")
        self._provider = provider
        self.cheap_model = cheap_model.strip()
        self.strong_model = strong_model.strip()

    def run_monolithic(
        self,
        text: str,
        *,
        options: Optional[AgentRequestOptions] = None,
    ) -> TicketTriageResult:
        cleaned = text.strip()
        metrics = MultistageMetrics()
        if not cleaned:
            return TicketTriageResult(
                ok=False,
                mode="monolithic",
                fields=None,
                error="empty_input",
                metrics=metrics,
                cheap_model=self.cheap_model,
                strong_model=self.strong_model,
            )

        opts = options or _default_options(max_tokens=256)
        messages = [
            {"role": "system", "content": _MONO_SYSTEM},
            {"role": "user", "content": cleaned},
        ]
        raw, model, latency, err = self._call(
            messages, opts, model=self.strong_model, stage="monolithic", metrics=metrics
        )
        if err:
            return TicketTriageResult(
                ok=False,
                mode="monolithic",
                fields=None,
                error=err,
                metrics=metrics,
                cheap_model=self.cheap_model,
                strong_model=self.strong_model,
            )

        payload = extract_json_object(raw or "")
        if not payload:
            return TicketTriageResult(
                ok=False,
                mode="monolithic",
                fields=None,
                error="unparseable_json",
                metrics=metrics,
                cheap_model=self.cheap_model,
                strong_model=self.strong_model,
            )

        validated = validate_triage_payload(payload)
        if not validated.ok:
            return TicketTriageResult(
                ok=False,
                mode="monolithic",
                fields=None,
                error=validated.reason,
                metrics=metrics,
                cheap_model=self.cheap_model,
                strong_model=self.strong_model,
            )

        return TicketTriageResult(
            ok=True,
            mode="monolithic",
            fields=validated.fields,
            error=None,
            metrics=metrics,
            cheap_model=self.cheap_model,
            strong_model=self.strong_model,
        )

    def run_multistage(
        self,
        text: str,
        *,
        options: Optional[AgentRequestOptions] = None,
    ) -> TicketTriageResult:
        cleaned = text.strip()
        metrics = MultistageMetrics()
        if not cleaned:
            return TicketTriageResult(
                ok=False,
                mode="multistage",
                fields=None,
                error="empty_input",
                metrics=metrics,
                cheap_model=self.cheap_model,
                strong_model=self.strong_model,
            )

        opts_short = options or _default_options(max_tokens=128)

        # Stage 1 — normalize (cheap)
        s1_messages = [
            {"role": "system", "content": _STAGE1_SYSTEM},
            {"role": "user", "content": cleaned},
        ]
        raw1, _, _, err1 = self._call(
            s1_messages, opts_short, model=self.cheap_model, stage="normalize", metrics=metrics
        )
        if err1:
            return self._fail("multistage", err1, metrics)
        stage1 = parse_compact_kv(raw1 or "")
        ok1, reason1 = validate_stage1(stage1)
        if not ok1:
            return self._fail("multistage", f"stage1:{reason1}", metrics)

        # Stage 2 — classify (cheap)
        s2_user = (
            f"normalized={format_compact_kv(stage1, ['lang', 'noise', 'product_hint', 'core'])}"
        )
        s2_messages = [
            {"role": "system", "content": _STAGE2_SYSTEM},
            {"role": "user", "content": s2_user},
        ]
        raw2, _, _, err2 = self._call(
            s2_messages, opts_short, model=self.cheap_model, stage="classify", metrics=metrics
        )
        if err2:
            return self._fail("multistage", err2, metrics)
        stage2 = parse_compact_kv(raw2 or "")
        ok2, reason2 = validate_stage2(stage2)
        if not ok2:
            return self._fail("multistage", f"stage2:{reason2}", metrics)

        # Stage 3 — decide + summary (strong)
        intent = stage2["intent"].strip().lower()
        urgency = stage2["urgency"].strip().lower()
        product = stage2["product"].strip().lower()
        hint = decide_by_rules(intent, urgency)
        s3_user = (
            f"intent={intent};urgency={urgency};product={product};"
            f"rule_hint={hint};core={stage1.get('core', '')}"
        )
        s3_messages = [
            {"role": "system", "content": _STAGE3_SYSTEM},
            {"role": "user", "content": s3_user},
        ]
        raw3, _, _, err3 = self._call(
            s3_messages, opts_short, model=self.strong_model, stage="format", metrics=metrics
        )
        if err3:
            return self._fail("multistage", err3, metrics)
        stage3 = parse_compact_kv(raw3 or "")
        ok3, reason3 = validate_stage3(stage3)
        if not ok3:
            return self._fail("multistage", f"stage3:{reason3}", metrics)

        payload = {
            "intent": intent,
            "urgency": urgency,
            "product": product,
            "decision": stage3["decision"].strip().lower(),
            "summary": stage3["summary"].strip(),
        }
        validated = validate_triage_payload(payload)
        if not validated.ok:
            return self._fail("multistage", validated.reason or "validation_failed", metrics)

        return TicketTriageResult(
            ok=True,
            mode="multistage",
            fields=validated.fields,
            error=None,
            metrics=metrics,
            cheap_model=self.cheap_model,
            strong_model=self.strong_model,
        )

    def _fail(self, mode: str, error: str, metrics: MultistageMetrics) -> TicketTriageResult:
        return TicketTriageResult(
            ok=False,
            mode=mode,
            fields=None,
            error=error,
            metrics=metrics,
            cheap_model=self.cheap_model,
            strong_model=self.strong_model,
        )

    def _call(
        self,
        messages: list[dict[str, str]],
        options: AgentRequestOptions,
        *,
        model: str,
        stage: str,
        metrics: MultistageMetrics,
    ) -> tuple[Optional[str], str, float, Optional[str]]:
        try:
            data, model_used, latency = self._provider.complete(
                messages,
                options,
                model_candidates=[model],
            )
        except Exception as exc:
            return None, model, 0.0, f"provider_error:{exc}"

        metrics.llm_calls += 1
        metrics.latency_sec_total += latency
        p_tok, c_tok = _usage_tokens(data)
        metrics.prompt_tokens += p_tok
        metrics.completion_tokens += c_tok
        metrics.models_used.append(model_used)

        parsed = parse_agent_response(
            data, model_used, latency, self._provider.provider
        )
        raw = strip_fences(parsed.answer)
        metrics.stages.append(
            InferenceStageMetrics(
                name=stage,
                model=model_used,
                latency_sec=latency,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                raw=raw[:500],
            )
        )
        if not raw.strip():
            return None, model_used, latency, f"empty_response:{stage}"
        return raw, model_used, latency, None
