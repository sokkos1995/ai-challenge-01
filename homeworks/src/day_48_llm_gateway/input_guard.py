"""Input Guard: detect and block/redact secrets in prompts."""
from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from typing import Literal

GuardMode = Literal["block", "redact"]

REDACTION_TOKENS: dict[str, str] = {
    "api_key": "[REDACTED_API_KEY]",
    "aws_key": "[REDACTED_AWS_KEY]",
    "github_token": "[REDACTED_GITHUB_TOKEN]",
    "email": "[REDACTED_EMAIL]",
    "phone": "[REDACTED_PHONE]",
    "card": "[REDACTED_CARD]",
    "base64_secret": "[REDACTED_BASE64_SECRET]",
}

_REDACT_PRIORITY = {
    "api_key": 100,
    "aws_key": 100,
    "github_token": 100,
    "base64_secret": 90,
    "card": 80,
    "email": 70,
    "phone": 60,
}

# OpenAI / project-style keys
_SK_KEY = re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{10,}\b")
_GHP = re.compile(r"\bghp_[A-Za-z0-9]{20,}\b")
_AKIA = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# RU / E.164-ish phones — require +/leading 7|8 or separators (avoid digits inside API keys)
_PHONE = re.compile(
    r"(?<![\w])("
    r"(?:\+\d{1,3}[\s\-()]*)?(?:\(?\d{2,4}\)?[\s\-.]*){2,4}\d{2,4}"
    r"|(?:(?:\+?7|8)[\s\-()]*)(?:\d[\s\-()]*){10}"
    r")(?![\w])"
)
# Candidate card digit runs (validated with Luhn)
_CARD_DIGITS = re.compile(r"(?<!\d)(?:\d[ \-]?){13,19}(?!\d)")
# Base64 blobs long enough to hide a key
_BASE64_BLOB = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")
# Split concatenation: "sk-" + "proj-abc..." or 'sk-' + 'proj-...'
_SPLIT_CONCAT = re.compile(
    r"""["']?(sk-(?:proj-)?)["']?\s*\+\s*["']([A-Za-z0-9_-]{6,})["']?""",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Finding:
    kind: str
    match: str
    start: int
    end: int


@dataclass
class GuardResult:
    ok: bool
    mode: GuardMode
    findings: list[Finding] = field(default_factory=list)
    cleaned_text: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def finding_kinds(self) -> list[str]:
        return sorted({f.kind for f in self.findings})


def _luhn_ok(digits: str) -> bool:
    if not digits.isdigit() or not (13 <= len(digits) <= 19):
        return False
    total = 0
    reverse = digits[::-1]
    for i, ch in enumerate(reverse):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _normalize_for_split(text: str) -> str:
    """Glue common split forms so regexes can see full secrets."""
    # "sk-" + "proj-abc" → sk-proj-abc
    glued = _SPLIT_CONCAT.sub(r"\1\2", text)
    # sk- + proj- (without quotes)
    glued = re.sub(
        r"(sk-(?:proj-)?)[\s]*\+[\s]*(proj-[A-Za-z0-9_-]+|[A-Za-z0-9_-]{8,})",
        r"\1\2",
        glued,
        flags=re.IGNORECASE,
    )
    return glued


def _scan_plain(text: str) -> list[Finding]:
    findings: list[Finding] = []

    for kind, pattern in (
        ("api_key", _SK_KEY),
        ("github_token", _GHP),
        ("aws_key", _AKIA),
        ("email", _EMAIL),
    ):
        for m in pattern.finditer(text):
            findings.append(Finding(kind=kind, match=m.group(0), start=m.start(), end=m.end()))

    for m in _PHONE.finditer(text):
        raw = m.group(0)
        digits = re.sub(r"\D", "", raw)
        # Accept 10–15 digits; reject bare runs that look like card/key fragments
        if not (10 <= len(digits) <= 15):
            continue
        # Must look like a phone: starts with +/7/8 or contains separators
        if not (raw.strip().startswith("+") or re.search(r"[\s\-()]", raw) or digits[0] in "78"):
            continue
        findings.append(Finding(kind="phone", match=raw.strip(), start=m.start(), end=m.end()))

    for m in _CARD_DIGITS.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if _luhn_ok(digits):
            findings.append(Finding(kind="card", match=m.group(0).strip(), start=m.start(), end=m.end()))

    return findings


def _scan_base64(text: str) -> list[Finding]:
    findings: list[Finding] = []
    for m in _BASE64_BLOB.finditer(text):
        blob = m.group(0)
        try:
            # pad if needed
            pad = "=" * ((4 - len(blob) % 4) % 4)
            decoded = base64.b64decode(blob + pad, validate=False).decode("utf-8", errors="ignore")
        except Exception:
            continue
        if not decoded or decoded == blob:
            continue
        inner = _scan_plain(decoded)
        if inner:
            findings.append(
                Finding(
                    kind="base64_secret",
                    match=blob[:24] + "…",
                    start=m.start(),
                    end=m.end(),
                )
            )
    return findings


def detect_secrets(text: str) -> list[Finding]:
    """Detect secrets including split and base64-encoded forms."""
    normalized = _normalize_for_split(text)
    findings = _scan_plain(normalized)
    findings.extend(_scan_base64(normalized))
    # Drop low-priority findings fully contained in higher-priority spans (e.g. digits in API keys)
    findings = _drop_contained_findings(findings)
    # Deduplicate by (kind, match, start)
    seen: set[tuple[str, str, int]] = set()
    unique: list[Finding] = []
    for f in findings:
        key = (f.kind, f.match, f.start)
        if key in seen:
            continue
        seen.add(key)
        unique.append(f)
    return unique


def _drop_contained_findings(findings: list[Finding]) -> list[Finding]:
    ranked = sorted(
        findings,
        key=lambda f: (-_REDACT_PRIORITY.get(f.kind, 0), -(f.end - f.start), f.start),
    )
    kept: list[Finding] = []
    for f in ranked:
        overlaps = False
        for k in kept:
            if f.start >= k.start and f.end <= k.end:
                overlaps = True
                break
            if not (f.end <= k.start or f.start >= k.end):
                overlaps = True
                break
        if overlaps:
            continue
        kept.append(f)
    return kept


def _redact_text(text: str, findings: list[Finding]) -> str:
    """Replace secret spans on normalized text."""
    work = _normalize_for_split(text)
    spans = detect_secrets(work)
    spans_sorted = sorted(spans, key=lambda f: f.start, reverse=True)
    for f in spans_sorted:
        token = REDACTION_TOKENS.get(f.kind, "[REDACTED]")
        work = work[: f.start] + token + work[f.end :]
    return work


def check_input(text: str, *, mode: GuardMode = "block") -> GuardResult:
    findings = detect_secrets(text)
    if not findings:
        return GuardResult(ok=True, mode=mode, findings=[], cleaned_text=text, warnings=[])

    kinds = sorted({f.kind for f in findings})
    warning = f"Input Guard: detected secrets ({', '.join(kinds)})"

    if mode == "block":
        return GuardResult(
            ok=False,
            mode=mode,
            findings=findings,
            cleaned_text=text,
            warnings=[warning, "Request blocked — nothing sent to LLM"],
        )

    cleaned = _redact_text(text, findings)
    return GuardResult(
        ok=True,
        mode=mode,
        findings=findings,
        cleaned_text=cleaned,
        warnings=[warning, "Secrets redacted; request forwarded to LLM"],
    )
