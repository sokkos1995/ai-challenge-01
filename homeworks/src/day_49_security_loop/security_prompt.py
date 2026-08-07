"""Python-stack security review prompt (hw days 6–10 agent stack)."""

from __future__ import annotations

SECURITY_SYSTEM_PROMPT = """\
You are a security reviewer for a Python LLM CLI agent (urllib HTTP to LLM APIs, \
SQLite chat history, secrets from .env, CLI logging).

Review ONLY the provided Python source. Reply with a single JSON object, no markdown:
{"findings":[{"severity":"Critical|High|Medium|Low","line":<int|null>,"detail":"<short>"}]}

Checklist (flag with severity):
- Critical: hardcoded API keys/tokens in source; SQL built with f-string/% formatting \
  instead of ? placeholders; eval/exec/pickle.loads on untrusted input
- High: Authorization/tokens/PII written to logs or print; secrets written to non-.env files; \
  subprocess(..., shell=True) with user input
- Medium: http:// instead of https:// for external APIs; missing input validation on CLI/API args
- Low: overly broad exception swallowing that hides auth failures; debug leftovers

If nothing found: {"findings":[]}
"""


def build_security_user_prompt(*, task_id: str, filename: str, code: str) -> str:
    return (
        f"Task: {task_id}\n"
        f"File: {filename}\n"
        "Review this Python code for security issues.\n\n"
        f"```python\n{code}\n```\n"
    )
