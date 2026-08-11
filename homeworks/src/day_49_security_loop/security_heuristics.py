"""Heuristic / AST security scanner for Python (offline + LLM fallback)."""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass
from typing import Literal

Severity = Literal["Critical", "High", "Medium", "Low"]

# OpenAI / project-style keys + Stripe-like variants (underscore instead of hyphen).
# Build regex from fragments to avoid GitHub secret-scanning false-positives
# on “secret-looking” tokens inside regex literals.
_SK_KEY = re.compile(
    r"\b"
    + r"(?:"
    + r"sk-" + r"(?:proj-)?[A-Za-z0-9_-]{10,}"
    + r"|"
    + r"sk_" + r"(?:live|test)_" + r"[A-Za-z0-9_-]{10,}"
    + r")\b"
)
_GHP = re.compile(r"\b" + "ghp_" + r"[A-Za-z0-9]{20,}" + r"\b")
_AKIA = re.compile(r"\b" + "AKIA" + r"[0-9A-Z]{16}" + r"\b")
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

_SQL_KEYWORDS = re.compile(
    r"(?i)\b(SELECT|INSERT|UPDATE|DELETE|WHERE|FROM|JOIN|GROUP\s+BY|ORDER\s+BY)\b"
)

_SHELL_KW = "shell"

_PRINT_FUNC = "print"
_LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}


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

    def _const_str(n: ast.AST) -> str | None:
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            return n.value
        return None

    def _concat_const_str(n: ast.AST) -> str | None:
        """Support 'a' + 'b' constant folding in AST."""
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            left = _concat_const_str(n.left)
            right = _concat_const_str(n.right)
            if left is not None and right is not None:
                return left + right
        return _const_str(n)

    def _is_secret_str(s: str) -> bool:
        return bool(_SK_KEY.search(s) or _GHP.search(s) or _AKIA.search(s))

    # Pass 1: detect tainted vars assigned to literal secret strings.
    tainted: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            val = node.value
            if not (isinstance(val, ast.Constant) and isinstance(val.value, str)):
                continue
            if not _is_secret_str(val.value):
                continue
            for t in node.targets:
                if isinstance(t, ast.Name):
                    tainted.add(t.id)
        if isinstance(node, ast.AnnAssign):
            val = node.value
            if not (isinstance(val, ast.Constant) and isinstance(val.value, str)):
                continue
            if not _is_secret_str(val.value):
                continue
            if isinstance(node.target, ast.Name):
                tainted.add(node.target.id)

    # Pass 2: find dangerous patterns + use taint for print/logging.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # eval/exec (direct)
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

        # getattr(__builtins__, "ev"+"al")(...), getattr(__builtins__, 'ex'+'ec')(...)
        if isinstance(func, ast.Call) and isinstance(func.func, ast.Name) and func.func.id == "getattr":
            if len(func.args) >= 2:
                first = func.args[0]
                second = func.args[1]

                module = _const_str(first) if isinstance(first, ast.Constant) else None
                if isinstance(first, ast.Name):
                    module = first.id

                attr = _concat_const_str(second)
                if module == "__builtins__" and attr in {"eval", "exec"}:
                    # Outer node is node itself, not inner getattr().
                    out.append(
                        SecurityFinding(
                            severity="Critical",
                            line=getattr(node, "lineno", None),
                            detail=f"obfuscated getattr(__builtins__, {attr}) call",
                            kind="eval_exec_obfuscated",
                        )
                    )

        # os.system / subprocess via getattr(os, "sys"+"tem")(...)
        if isinstance(func, ast.Call) and isinstance(func.func, ast.Name) and func.func.id == "getattr":
            if len(func.args) >= 2:
                first = func.args[0]
                second = func.args[1]
                module = first.id if isinstance(first, ast.Name) else None
                attr = _concat_const_str(second)
                if module == "os" and attr == "system":
                    out.append(
                        SecurityFinding(
                            severity="High",
                            line=getattr(node, "lineno", None),
                            detail="obfuscated os.system via getattr()",
                            kind="os_system_obfuscated",
                        )
                    )

        # pickle.loads on untrusted input
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

        # subprocess/process execution: shell=True
        if name in {"run", "Popen", "call", "system"} or (
            isinstance(func, ast.Attribute) and func.attr in {"run", "Popen", "call", "system"}
        ):
            for kw in node.keywords:
                if kw.arg != _SHELL_KW:
                    continue
                if isinstance(kw.value, ast.Constant) and kw.value.value is False:
                    continue
                out.append(
                    SecurityFinding(
                        severity="High",
                        line=getattr(node, "lineno", None),
                        detail="subprocess with shell=True (not a constant false)",
                        kind="shell_true",
                    )
                )

        # SQL injection: execute/executemany with .format() or string concatenation.
        if isinstance(func, ast.Attribute) and func.attr in {"execute", "executemany"}:
            if node.args:
                arg0 = node.args[0]
                sql_target: str | None = None
                # 'SELECT ... {x}'.format(...)
                if (
                    isinstance(arg0, ast.Call)
                    and isinstance(arg0.func, ast.Attribute)
                    and arg0.func.attr == "format"
                    and isinstance(arg0.func.value, ast.Constant)
                    and isinstance(arg0.func.value.value, str)
                ):
                    sql_target = arg0.func.value.value
                # 'SELECT ...' + user_input
                elif isinstance(arg0, ast.BinOp) and isinstance(arg0.op, ast.Add):
                    left = _concat_const_str(arg0.left)
                    right = _concat_const_str(arg0.right)
                    if left is not None and right is not None:
                        sql_target = left + right
                    else:
                        # If only one side is constant, still check that side.
                        if left is not None:
                            sql_target = left
                        elif right is not None:
                            sql_target = right

                if sql_target and _SQL_KEYWORDS.search(sql_target):
                    out.append(
                        SecurityFinding(
                            severity="Critical",
                            line=getattr(node, "lineno", None),
                            detail="SQL built with .format()/string concatenation (use ? placeholders)",
                            kind="sql_injection_format_concat",
                        )
                    )

        # taint sink: print(secret_var), logging.info(secret_var), ...
        if isinstance(func, ast.Name) and func.id == _PRINT_FUNC:
            for a in node.args:
                if isinstance(a, ast.Name) and a.id in tainted:
                    out.append(
                        SecurityFinding(
                            severity="High",
                            line=getattr(node, "lineno", None),
                            detail=f"logging/printing tainted secret variable {a.id}",
                            kind="pii_in_logs_tainted",
                        )
                    )

        if isinstance(func, ast.Attribute) and func.attr in _LOG_LEVELS:
            if isinstance(func.value, ast.Name) and func.value.id in {"logging", "logger"}:
                for a in node.args:
                    if isinstance(a, ast.Name) and a.id in tainted:
                        out.append(
                            SecurityFinding(
                                severity="High",
                                line=getattr(node, "lineno", None),
                                detail=f"logging.{func.attr}() of tainted secret variable {a.id}",
                                kind="pii_in_logs_tainted",
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
