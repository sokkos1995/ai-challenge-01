"""Unified Level-1 + Level-2 QA flow for day_38 (post-PR / after feature).

Writes homeworks/artifacts/day_38/qa_report.{md,json}.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "homeworks" / "artifacts" / "day_38"
LEVEL1_TESTS = [
    "tests/test_todoist_hw_service.py",
    "tests/test_personalization_service.py",
    "tests/test_storage.py",
]


def _run(cmd: list[str], *, cwd: Path = ROOT) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    return {
        "command": " ".join(cmd),
        "exit_code": proc.returncode,
        "ok": proc.returncode == 0,
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
    }


def run_level1() -> dict[str, Any]:
    python = str(ROOT / ".venv" / "bin" / "python")
    if not Path(python).exists():
        python = sys.executable
    result = _run([python, "-m", "pytest", *LEVEL1_TESTS, "-q"])
    result["level"] = 1
    result["name"] = "unit_integration"
    result["tests"] = LEVEL1_TESTS
    return result


def run_level2() -> dict[str, Any]:
    python = str(ROOT / ".venv" / "bin" / "python")
    if not Path(python).exists():
        python = sys.executable
    smoke_json = ARTIFACTS / "smoke_report.json"
    smoke_md = ARTIFACTS / "smoke_report.md"
    if smoke_json.exists():
        smoke_json.unlink()
    if smoke_md.exists():
        smoke_md.unlink()
    result = _run([python, "-m", "homeworks.src.day_38_smoke.run_smoke"])
    result["level"] = 2
    result["name"] = "ui_smoke"
    if smoke_json.exists():
        try:
            result["smoke_report"] = json.loads(smoke_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result["smoke_report"] = None
    else:
        result["smoke_report"] = None
    return result


def write_unified_report(
    *,
    trigger: str,
    level1: dict[str, Any],
    level2: dict[str, Any],
    notes: str = "",
) -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    all_ok = bool(level1.get("ok")) and bool(level2.get("ok"))
    payload: dict[str, Any] = {
        "generated_at": stamp,
        "trigger": trigger,
        "all_passed": all_ok,
        "level1": {
            "ok": level1.get("ok"),
            "exit_code": level1.get("exit_code"),
            "command": level1.get("command"),
            "stdout": level1.get("stdout"),
            "stderr": level1.get("stderr"),
            "tests": level1.get("tests"),
        },
        "level2": {
            "ok": level2.get("ok"),
            "exit_code": level2.get("exit_code"),
            "command": level2.get("command"),
            "stdout": level2.get("stdout"),
            "stderr": level2.get("stderr"),
            "smoke_all_passed": (level2.get("smoke_report") or {}).get("all_passed"),
            "scenarios": (level2.get("smoke_report") or {}).get("scenarios"),
        },
        "notes": notes,
        "artifacts": {
            "qa_report_md": "qa_report.md",
            "qa_report_json": "qa_report.json",
            "smoke_report_md": "smoke_report.md",
            "smoke_report_json": "smoke_report.json",
        },
    }
    json_path = ARTIFACTS / "qa_report.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Day 38 unified QA report",
        "",
        f"Generated: `{stamp}`",
        f"Trigger: `{trigger}`",
        f"**Overall:** {'PASS' if all_ok else 'FAIL'}",
        "",
        "## Level 1 — unit/integration",
        "",
        f"- Status: {'PASS' if level1.get('ok') else 'FAIL'} (exit `{level1.get('exit_code')}`)",
        f"- Command: `{level1.get('command')}`",
        f"- Output: `{level1.get('stdout') or level1.get('stderr') or '(empty)'}`",
        "",
        "## Level 2 — UI smoke",
        "",
        f"- Status: {'PASS' if level2.get('ok') else 'FAIL'} (exit `{level2.get('exit_code')}`)",
        f"- Command: `{level2.get('command')}`",
        f"- Smoke overall: `{(level2.get('smoke_report') or {}).get('all_passed')}`",
        "- Details: [smoke_report.md](smoke_report.md)",
        "",
    ]
    scenarios = (level2.get("smoke_report") or {}).get("scenarios") or []
    for item in scenarios:
        mark = "PASS" if item.get("ok") else "FAIL"
        lines.append(f"- {item.get('id')} {item.get('title')}: {mark}")
        if item.get("failure"):
            lines.append(f"  - Failure: `{item['failure']}`")
            lines.append(
                f"  - Suggested area: UI handler / `data-testid` for `{item.get('id')}`"
            )
    if notes.strip():
        lines.extend(["", "## Notes", "", notes.strip(), ""])
    if not all_ok:
        lines.extend(
            [
                "",
                "## Where to look",
                "",
                "- Level 1 fail → pytest output above / corresponding `tests/test_*.py`",
                "- Level 2 fail → `smoke_report.md` + `*_fail.png` screenshots",
                "",
            ]
        )
    md_path = ARTIFACTS / "qa_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    trigger = "post-pr"
    notes = ""
    if args:
        trigger = args[0]
    if len(args) > 1:
        notes = " ".join(args[1:])

    level1 = run_level1()
    level2 = run_level2()
    report = write_unified_report(trigger=trigger, level1=level1, level2=level2, notes=notes)
    print(report)
    print("PASS" if level1["ok"] and level2["ok"] else "FAIL")
    return 0 if level1["ok"] and level2["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
