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
    "hex_secret": "[REDACTED_HEX_SECRET]",
}

_REDACT_PRIORITY = {
    "api_key": 100,
    "aws_key": 100,
    "github_token": 100,
    "base64_secret": 90,
    "hex_secret": 90,
    "card": 80,
    "email": 70,
    "phone": 60,
}

# OpenAI / project-style keys
# OpenAI / project-style keys + Stripe-like variants (underscore instead of hyphen).
_SK_KEY = re.compile(
    r"\b"
    + r"(?:"
    + r"sk-" + r"(?:proj-)?[A-Za-z0-9_-]{10,}"
    + r"|"
    + r"sk_" + r"(?:live|test)_" + r"[A-Za-z0-9_-]{10,}"
    + r")\b"
)
# Build regex from fragments to avoid GitHub secret scanning false-positives
# on “secret-looking” tokens inside regex literals.
_GHP = re.compile(r"\b" + "ghp_" + r"[A-Za-z0-9]{20,}" + r"\b")
_AKIA = re.compile(r"\b" + "AKIA" + r"[0-9A-Z]{16}" + r"\b")
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
# Base64 blobs long enough to hide a key (strict, no whitespace).
_BASE64_BLOB = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")
# Base64 candidate blobs that may include whitespace and/or use url-safe alphabet (-/_).
_BASE64_BLOB_FLEX = re.compile(r"(?<![\w])(?:[A-Za-z0-9+/=_\-\s]{40,})(?![\w])")
# Split concatenation: "sk-" + "proj-abc..." or 'sk-' + 'proj-...'
_SPLIT_CONCAT = re.compile(
    r"""["']?(sk-(?:proj-)?)["']?\s*\+\s*["']([A-Za-z0-9_-]{6,})["']?""",
    re.IGNORECASE,
)
# Zero-width / soft-hyphen obfuscation between key fragments
_ZERO_WIDTH = re.compile(
    "["
    "\u200b"  # ZWSP
    "\u200c"  # ZWNJ
    "\u200d"  # ZWJ
    "\u2060"  # WJ
    "\ufeff"  # BOM
    "\u00ad"  # soft hyphen
    "]"
)
_C_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_C_LINE_COMMENT = re.compile(r"//.*?$", re.MULTILINE)
_HASH_LINE_COMMENT = re.compile(r"#.*?$", re.MULTILINE)
# sk- / sk-proj- then short junk (spaces/newlines/quotes/+) then suffix
_SK_GLUE = re.compile(
    r"""(sk-(?:proj-)?)(?:[\s"'+]{1,40})([A-Za-z0-9_-]{6,})""",
    re.IGNORECASE,
)

# Long hex blobs that could encode secrets.
_HEX_BLOB = re.compile(r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{32,})(?![0-9A-Fa-f])")


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
    """Glue common split / obfuscated forms so regexes can see full secrets.

    Handles: quoted ``+`` concat, C/Python comments between fragments,
    newlines/whitespace, and zero-width characters (day_50 residual misses).
    """
    work = _ZERO_WIDTH.sub("", text)
    work = _C_BLOCK_COMMENT.sub(" ", work)
    work = _C_LINE_COMMENT.sub(" ", work)
    work = _HASH_LINE_COMMENT.sub(" ", work)
    # "sk-" + "proj-abc" → sk-proj-abc
    work = _SPLIT_CONCAT.sub(r"\1\2", work)
    # "s" + "k-proj-abc" → "sk-proj-abc" (prefix itself was split)
    work = re.sub(
        r"""["']s["']\s*\+\s*["']k-(?:(proj-))?([A-Za-z0-9_-]{6,})["']""",
        r"sk-\1\2",
        work,
        flags=re.IGNORECASE,
    )
    # sk- + proj- (without quotes)
    work = re.sub(
        r"(sk-(?:proj-)?)[\s]*\+[\s]*(proj-[A-Za-z0-9_-]+|[A-Za-z0-9_-]{8,})",
        r"\1\2",
        work,
        flags=re.IGNORECASE,
    )
    # "AKIA" + "<16chars>" → "AKIA<16>"
    work = re.sub(
        r"""["']AKIA["']\s*\+\s*["']([0-9A-Z]{16})["']""",
        r"AKIA\1",
        work,
    )
    # "".join(["sk-","proj-..."]) → sk-proj-...
    work = re.sub(
        r"""\.join\(\[\s*["']sk-(?:proj-)?["']\s*,\s*["'](?:(proj-))?([A-Za-z0-9_-]{6,})["']\s*\]\s*\)""",
        r"sk-\1\2",
        work,
        flags=re.IGNORECASE,
    )
    # "".join(["s","k-proj-..."]) → sk-proj-...
    work = re.sub(
        r"""\.join\(\[\s*["']s["']\s*,\s*["']k-(?:(proj-))?([A-Za-z0-9_-]{6,})["']\s*\]\s*\)""",
        r"sk-\1\2",
        work,
        flags=re.IGNORECASE,
    )
    # sk-\nrest / sk-  "rest" / leftover spaces after comment strip
    work = _SK_GLUE.sub(r"\1\2", work)
    return work


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


def _decode_base64_candidate(blob: str) -> str | None:
    """Decode base64/urlsafe blob; return decoded text or None."""
    cleaned = re.sub(r"\s+", "", blob)
    if not cleaned:
        return None
    # pad if needed
    pad = "=" * ((4 - len(cleaned) % 4) % 4)
    try:
        decoded_bytes = base64.b64decode(cleaned + pad, validate=False)
        return decoded_bytes.decode("utf-8", errors="ignore")
    except Exception:
        # last attempt: url-safe alphabet
        try:
            decoded_bytes = base64.urlsafe_b64decode(cleaned + pad)
            return decoded_bytes.decode("utf-8", errors="ignore")
        except Exception:
            return None


def _scan_base64_flexible(text: str) -> list[Finding]:
    """Detect base64 blobs with whitespace/newlines and url-safe alphabet."""
    findings: list[Finding] = []
    for m in _BASE64_BLOB_FLEX.finditer(text):
        blob = m.group(0)
        variants = [blob]
        # The flexible regex can accidentally include short leading words
        # like "blob " (letters are part of base64 alphabet). Try decoding
        # again without the leading token.
        if re.search(r"\s", blob):
            try:
                variants.append(blob.split(None, 1)[1])
            except Exception:
                pass

        matched = False
        for variant in variants:
            blob_stripped = re.sub(r"\s+", "", variant)
            if len(blob_stripped) < 24:
                continue

            decoded1 = _decode_base64_candidate(variant)
            if not decoded1 or decoded1 == variant:
                continue

            inner = _scan_plain(decoded1)
            if inner:
                findings.append(
                    Finding(
                        kind="base64_secret",
                        match=blob_stripped[:24] + "…",
                        start=m.start(),
                        end=m.end(),
                    )
                )
                matched = True
                break

            # double-base64: decoded text might itself be base64-like
            decoded1_stripped = decoded1.strip()
            if len(decoded1_stripped) < 24:
                continue
            if not re.fullmatch(r"[A-Za-z0-9+/=_\-\s]+", decoded1_stripped):
                continue
            decoded2 = _decode_base64_candidate(decoded1_stripped)
            if not decoded2 or decoded2 == decoded1_stripped:
                continue
            if _scan_plain(decoded2):
                findings.append(
                    Finding(
                        kind="base64_secret",
                        match=blob_stripped[:24] + "…",
                        start=m.start(),
                        end=m.end(),
                    )
                )
                matched = True
                break

        if matched:
            continue
    return findings


def _scan_hex(text: str) -> list[Finding]:
    """Detect hex-encoded secrets by decoding to text and running plain scans."""
    findings: list[Finding] = []
    for m in _HEX_BLOB.finditer(text):
        blob = m.group(0)
        if len(blob) % 2 != 0:
            continue
        # Avoid runaway on absurdly long blobs.
        if len(blob) > 4096:
            continue
        try:
            raw = bytes.fromhex(blob).decode("utf-8", errors="ignore")
        except Exception:
            continue
        if not raw:
            continue
        if _scan_plain(raw):
            findings.append(
                Finding(
                    kind="hex_secret",
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
    # Higher-layer bypasses often encode the same plaintext secret into an obfuscation form.
    findings.extend(_scan_hex(normalized))
    findings.extend(_scan_base64(normalized))
    findings.extend(_scan_base64_flexible(normalized))
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
