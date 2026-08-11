"""Heuristic / AST security scanner for Python (offline + LLM fallback)."""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass
from typing import Literal

Severity = Literal["Critical", "High", "Medium", "Low"]

_SK_KEY = re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{10,}\b")
_GHP = re.compile(r"\bghp_[A-Za-z0-9]{20,}\b")
_AKIA = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_BEARER = re.compile(r"(?i)(?:bearer|authorization)[\"'\s:=]+[A-Za-z0-9._\-]{8,}")
_HTTP_URL = re.compile(r"""['"]http://[^'"]+['"]""")
_SQL_FSTRING = re.compile(
    r"""(?:execute|executemany)\s*\(\s*f['"].*(?:SELECT|INSERT|UPDATE|DELETE|WHERE)""",
    re.IGNORECASE,
)
_SQL_PERCENT = re.compile(
    r"""(?:execute|executemany)\s*\(\s*['"].*%[sd].*['"]\s*%""",
    re.IGNORECASE,
)
_LOG_SENSITIVE = re.compile(
    r"""(?:print|logging\.\w+|logger\.\w+)\s*\([^)]*(?:token|password|authorization|api_key|secret)""",
    re.IGNORECASE,
)
_TOKEN_FILE_WRITE = re.compile(
    r"""(?:open|Path\([^)]+\)\.write_text)\s*\([^)]*(?:token|secret|\.token)""",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SecurityFinding:
    severity: Severity
    line: int | None
    detail: str
    kind: str
    source: str = "heuristic"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _line_of(text: str, start: int) -> int:
    return text.count("\n", 0, start) + 1


def _add_regex(
    findings: list[SecurityFinding],
    text: str,
    pattern: re.Pattern[str],
    *,
    severity: Severity,
    kind: str,
    detail: str,
) -> None:
    for m in pattern.finditer(text):
        findings.append(
            SecurityFinding(
                severity=severity,
                line=_line_of(text, m.start()),
                detail=detail,
                kind=kind,
            )
        )


def _ast_findings(code: str) -> list[SecurityFinding]:
    out: list[SecurityFinding] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return out

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr

            if name in {"eval", "exec"}:
                out.append(
                    SecurityFinding(
                        severity="Critical",
                        line=getattr(node, "lineno", None),
                        detail=f"unsafe {name}() call",
                        kind="eval_exec",
                    )
                )
            if name == "loads" and isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name) and func.value.id == "pickle":
                    out.append(
                        SecurityFinding(
                            severity="Critical",
                            line=getattr(node, "lineno", None),
                            detail="pickle.loads on untrusted data",
                            kind="pickle_loads",
                        )
                    )
            if name in {"run", "Popen", "call", "system"}:
                for kw in node.keywords:
                    if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        out.append(
                            SecurityFinding(
                                severity="High",
                                line=getattr(node, "lineno", None),
                                detail="subprocess with shell=True",
                                kind="shell_true",
                            )
                        )
    return out


def _normalize_secret_obfuscation(code: str) -> str:
    """Align with day_48 input_guard: strip ZW/comments and glue sk- splits."""
    work = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]", "", code)
    work = re.sub(r"/\*.*?\*/", " ", work, flags=re.DOTALL)
    work = re.sub(r"//.*?$", " ", work, flags=re.MULTILINE)
    work = re.sub(
        r"""["']?(sk-(?:proj-)?)["']?\s*\+\s*["']([A-Za-z0-9_-]{6,})["']?""",
        r"\1\2",
        work,
        flags=re.IGNORECASE,
    )
    work = re.sub(
        r"""(sk-(?:proj-)?)(?:[\s"'+]{1,40})([A-Za-z0-9_-]{6,})""",
        r"\1\2",
        work,
        flags=re.IGNORECASE,
    )
    return work


def scan_code(code: str) -> list[SecurityFinding]:
    """Return heuristic findings ordered Critical → Low."""
    findings: list[SecurityFinding] = []
    secret_view = _normalize_secret_obfuscation(code)
    _add_regex(
        findings,
        secret_view,
        _SK_KEY,
        severity="Critical",
        kind="hardcoded_api_key",
        detail="hardcoded sk- API key in source",
    )
    _add_regex(
        findings,
        secret_view,
        _GHP,
        severity="Critical",
        kind="hardcoded_github_token",
        detail="hardcoded GitHub token in source",
    )
    _add_regex(
        findings,
        secret_view,
        _AKIA,
        severity="Critical",
        kind="hardcoded_aws_key",
        detail="hardcoded AWS access key in source",
    )
    _add_regex(
        findings,
        code,
        _SQL_FSTRING,
        severity="Critical",
        kind="sql_injection_fstring",
        detail="SQL built with f-string (use ? placeholders)",
    )
    _add_regex(
        findings,
        code,
        _SQL_PERCENT,
        severity="Critical",
        kind="sql_injection_percent",
        detail="SQL built with % formatting (use ? placeholders)",
    )
    _add_regex(
        findings,
        code,
        _BEARER,
        severity="High",
        kind="auth_in_source",
        detail="Authorization/Bearer token literal in source",
    )
    _add_regex(
        findings,
        code,
        _LOG_SENSITIVE,
        severity="High",
        kind="pii_in_logs",
        detail="logging/printing sensitive fields (token/password/Authorization)",
    )
    _add_regex(
        findings,
        code,
        _TOKEN_FILE_WRITE,
        severity="High",
        kind="secret_file_write",
        detail="writing token/secret to a non-.env file",
    )
    _add_regex(
        findings,
        code,
        _HTTP_URL,
        severity="Medium",
        kind="http_not_https",
        detail="HTTP URL used instead of HTTPS",
    )
    findings.extend(_ast_findings(code))

    # de-dupe by (kind, line, detail)
    seen: set[tuple[str, int | None, str]] = set()
    unique: list[SecurityFinding] = []
    for f in findings:
        key = (f.kind, f.line, f.detail)
        if key in seen:
            continue
        seen.add(key)
        unique.append(f)

    order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    unique.sort(key=lambda f: (order.get(f.severity, 9), f.line or 0, f.kind))
    return unique


def max_severity(findings: list[SecurityFinding]) -> Severity | None:
    if not findings:
        return None
    order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    return min((f.severity for f in findings), key=lambda s: order[s])


def needs_regen(findings: list[SecurityFinding]) -> bool:
    return any(f.severity in {"Critical", "High"} for f in findings)


def decision_for(findings: list[SecurityFinding]) -> str:
    """regen | warn | commit"""
    if needs_regen(findings):
        return "regen"
    if findings:
        return "warn"
    return "commit"


def feedback_from_findings(findings: list[SecurityFinding]) -> str:
    parts: list[str] = []
    for f in findings:
        if f.severity not in {"Critical", "High"}:
            continue
        loc = f"в строке {f.line}" if f.line else "в коде"
        parts.append(f"исправь: {f.detail} {loc}")
    return "; ".join(parts) if parts else "исправь: security Critical/High findings"
