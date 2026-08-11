"""Execution loop: generate → tests → security review → regen/commit via gateway."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .gateway_client import CompleterFn, GatewayClient
from .security_heuristics import (
    SecurityFinding,
    decision_for,
    feedback_from_findings,
    scan_code,
)
from .security_prompt import SECURITY_SYSTEM_PROMPT, build_security_user_prompt
from .tasks import TASKS, TaskSpec

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ART = ROOT / "homeworks" / "artifacts" / "day_49"

_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class IterationRecord:
    n: int
    phase: str
    ok: bool
    detail: str = ""
    gateway_status: str = ""
    security_decision: str = ""
    findings: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TaskResult:
    task_id: str
    prompt: str
    ok: bool
    commit_status: str  # committed | skipped | blocked | failed
    iterations: list[IterationRecord] = field(default_factory=list)
    security_findings: list[dict[str, Any]] = field(default_factory=list)
    security_caught: list[str] = field(default_factory=list)
    gateway_events: list[dict[str, Any]] = field(default_factory=list)
    gateway_caught: list[str] = field(default_factory=list)
    gateway_clean_count: int = 0
    missed_by_both: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    final_path: str = ""
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["iterations"] = [asdict(i) for i in self.iterations]
        return d


def extract_python(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    m = _CODE_FENCE.search(text)
    if m:
        return m.group(1).strip() + "\n"
    # strip leading prose lines until a Python-looking line
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith(("import ", "from ", '"""', "'''", "def ", "class ", "#")):
            start = i
            break
    return "\n".join(lines[start:]).strip() + "\n"


def parse_security_json(raw: str) -> list[SecurityFinding]:
    text = (raw or "").strip()
    if not text:
        return []
    # strip fences
    fence = _CODE_FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        payload = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return []
    findings_raw = payload.get("findings") if isinstance(payload, dict) else None
    if not isinstance(findings_raw, list):
        return []
    out: list[SecurityFinding] = []
    for item in findings_raw:
        if not isinstance(item, dict):
            continue
        sev = str(item.get("severity") or "Medium")
        if sev not in {"Critical", "High", "Medium", "Low"}:
            sev = "Medium"
        line = item.get("line")
        line_i = int(line) if isinstance(line, int) else None
        detail = str(item.get("detail") or "security finding")
        out.append(
            SecurityFinding(
                severity=sev,  # type: ignore[arg-type]
                line=line_i,
                detail=detail,
                kind=str(item.get("kind") or "llm"),
                source="llm",
            )
        )
    return out


_SEV_RANK = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


def merge_findings(
    llm: list[SecurityFinding],
    heuristic: list[SecurityFinding],
) -> list[SecurityFinding]:
    """Union by kind; if both report the same kind, keep the worse severity.

    Prevents LLM «test-only / Low» from downgrading a heuristic Critical/High.
    """
    if not llm:
        return list(heuristic)
    merged = list(llm)
    for h in heuristic:
        same_idxs = [i for i, f in enumerate(merged) if f.kind == h.kind]
        if not same_idxs:
            merged.append(h)
            continue
        for i in same_idxs:
            cur = merged[i]
            if _SEV_RANK.get(h.severity, 9) < _SEV_RANK.get(cur.severity, 9):
                merged[i] = SecurityFinding(
                    severity=h.severity,
                    line=cur.line if cur.line is not None else h.line,
                    detail=h.detail,
                    kind=h.kind,
                    source="heuristic",
                )
    return merged


def verify_task(task: TaskSpec, path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "file missing"
    code = path.read_text(encoding="utf-8")
    try:
        compile(code, str(path), "exec")
    except SyntaxError as exc:
        return False, f"syntax: {exc}"

    if task.id == "save_auth_token":
        script = (
            "import importlib.util\n"
            f"s=importlib.util.spec_from_file_location('m', r'{path}')\n"
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
            "assert hasattr(m,'save_token') and hasattr(m,'load_token')\n"
        )
    elif task.id == "log_all_requests":
        script = (
            "import importlib.util\n"
            f"s=importlib.util.spec_from_file_location('m', r'{path}')\n"
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
            "assert hasattr(m,'log_request')\n"
            "out=m.log_request('GET','https://example.com',{'Authorization':'secret'})\n"
            "assert isinstance(out,str)\n"
        )
    elif task.id == "api_request":
        script = (
            "import importlib.util\n"
            f"s=importlib.util.spec_from_file_location('m', r'{path}')\n"
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
            "assert hasattr(m,'fetch')\n"
        )
    else:
        return False, "unknown task"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=15,
    )
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "verify failed")[:500]
    return True, "ok"


def build_generate_prompt(task: TaskSpec, *, feedback: str = "") -> str:
    base = (
        f"Задача: {task.prompt}\n"
        f"Стек: Python CLI LLM-агент (дни 6–10): urllib, SQLite, секреты из .env.\n"
        f"Создай один файл `{task.filename}`.\n"
        f"Требование Done: {task.done_hint}\n"
        "Ответь только Python-кодом файла (можно в ```python).\n"
    )
    if feedback:
        base += f"\nФидбек предыдущей итерации:\n{feedback}\n"
    return base


def offline_completer_factory(
    tasks: list[TaskSpec],
) -> CompleterFn:
    """Deterministic generator: insecure first, secure after security feedback."""

    state: dict[str, str] = {t.id: "insecure" for t in tasks}

    def _detect_task(messages: list[dict[str, str]]) -> TaskSpec | None:
        blob = "\n".join(m.get("content", "") for m in messages)
        for t in tasks:
            if t.id in blob or t.prompt in blob or t.filename in blob:
                return t
        # security review prompts include filename
        for t in tasks:
            if f"File: {t.filename}" in blob:
                return t
        return None

    def complete(messages: list[dict[str, str]], model: str | None = None) -> str:
        blob = "\n".join(m.get("content", "") for m in messages)
        is_security = "security reviewer" in blob.lower() or "Review this Python code" in blob
        task = _detect_task(messages)
        if task is None:
            return '{"findings":[]}' if is_security else "# empty\n"

        if is_security:
            # Review whatever code is in the prompt fence; use heuristics via JSON
            code_m = _CODE_FENCE.search(blob)
            code = code_m.group(1) if code_m else ""
            findings = scan_code(code)
            payload = {
                "findings": [
                    {
                        "severity": f.severity,
                        "line": f.line,
                        "detail": f.detail,
                        "kind": f.kind,
                    }
                    for f in findings
                ]
            }
            return json.dumps(payload, ensure_ascii=False)

        # generation
        if "исправь:" in blob or "Critical" in blob or "High" in blob:
            state[task.id] = "secure"
        if state[task.id] == "secure":
            return task.secure_fixture
        return task.insecure_fixture

    return complete


class SecurityLoop:
    def __init__(
        self,
        *,
        art_dir: Path | None = None,
        gateway: GatewayClient | None = None,
        max_iters: int = 3,
        offline: bool = False,
        tasks: list[TaskSpec] | None = None,
    ) -> None:
        self.art_dir = art_dir or DEFAULT_ART
        self.workspace = self.art_dir / "workspace"
        self.committed = self.art_dir / "committed"
        self.max_iters = max(1, max_iters)
        self.tasks = tasks or list(TASKS)
        self.offline = offline
        if gateway is not None:
            self.gateway = gateway
        elif offline:
            self.gateway = GatewayClient(
                mode="redact",
                in_process=offline_completer_factory(self.tasks),
                use_input_guard=True,
            )
        else:
            self.gateway = GatewayClient(mode="redact")

    def run(self) -> list[TaskResult]:
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.committed.mkdir(parents=True, exist_ok=True)
        results: list[TaskResult] = []
        for task in self.tasks:
            results.append(self.run_task(task))
        self._write_artifacts(results)
        return results

    def run_task(self, task: TaskSpec) -> TaskResult:
        t0 = time.perf_counter()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.committed.mkdir(parents=True, exist_ok=True)
        task_dir = self.workspace / task.id
        task_dir.mkdir(parents=True, exist_ok=True)
        path = task_dir / task.filename
        result = TaskResult(
            task_id=task.id,
            prompt=task.prompt,
            ok=False,
            commit_status="failed",
        )
        feedback = ""
        security_feedback = ""

        for n in range(1, self.max_iters + 1):
            # --- generate ---
            gen_prompt = build_generate_prompt(
                task,
                feedback="\n".join(x for x in (feedback, security_feedback) if x),
            )
            # Deliberately include a benign marker; fixtures may contain secrets that
            # gateway should redact/block depending on mode.
            chat = self.gateway.chat(prompt=gen_prompt, stage=f"generate:{task.id}:{n}")
            result.gateway_events.append(chat.event.to_dict())
            if chat.event.blocked and not chat.answer:
                # In offline redact mode secrets in fixture answers are fine;
                # blocked input (e.g. if prompt itself had secrets) — inject fixture.
                if self.offline:
                    code = task.insecure_fixture if n == 1 else task.secure_fixture
                    path.write_text(code, encoding="utf-8")
                    result.iterations.append(
                        IterationRecord(
                            n=n,
                            phase="generate",
                            ok=True,
                            detail="offline fallback after gateway block",
                            gateway_status=chat.event.status,
                        )
                    )
                else:
                    result.iterations.append(
                        IterationRecord(
                            n=n,
                            phase="generate",
                            ok=False,
                            detail="gateway blocked generate",
                            gateway_status=chat.event.status,
                        )
                    )
                    result.commit_status = "blocked"
                    break
            else:
                code = extract_python(chat.answer) or (
                    task.insecure_fixture if self.offline else ""
                )
                if not code.strip():
                    result.iterations.append(
                        IterationRecord(
                            n=n,
                            phase="generate",
                            ok=False,
                            detail="empty generation",
                            gateway_status=chat.event.status,
                        )
                    )
                    feedback = "пустой ответ — сгенерируй полный Python-файл"
                    continue
                path.write_text(code, encoding="utf-8")
                result.iterations.append(
                    IterationRecord(
                        n=n,
                        phase="generate",
                        ok=True,
                        detail=f"wrote {path.name}",
                        gateway_status=chat.event.status,
                    )
                )

            # --- tests ---
            ok, detail = verify_task(task, path)
            result.iterations.append(
                IterationRecord(n=n, phase="tests", ok=ok, detail=detail[:300])
            )
            if not ok:
                feedback = f"тесты упали: {detail[:400]}"
                security_feedback = ""
                continue
            feedback = ""

            # --- security review ---
            code_now = path.read_text(encoding="utf-8")
            sec_user = build_security_user_prompt(
                task_id=task.id,
                filename=task.filename,
                code=code_now,
            )
            sec_chat = self.gateway.chat(
                prompt=sec_user,
                stage=f"security:{task.id}:{n}",
                system=SECURITY_SYSTEM_PROMPT,
            )
            result.gateway_events.append(sec_chat.event.to_dict())

            llm_findings = parse_security_json(sec_chat.answer) if sec_chat.answer else []
            heur = scan_code(code_now)
            findings = merge_findings(llm_findings, heur)
            decision = decision_for(findings)
            # Accumulate across iterations (final clean pass must not erase earlier catches).
            prev_kinds = set(result.security_caught)
            new_kinds = {f.kind for f in findings}
            result.security_caught = sorted(prev_kinds | new_kinds)
            seen_finding_keys = {
                (d.get("kind"), d.get("line"), d.get("detail"))
                for d in result.security_findings
            }
            for f in findings:
                key = (f.kind, f.line, f.detail)
                if key not in seen_finding_keys:
                    result.security_findings.append(f.to_dict())
                    seen_finding_keys.add(key)
            result.iterations.append(
                IterationRecord(
                    n=n,
                    phase="security",
                    ok=decision != "regen",
                    detail=feedback_from_findings(findings) if decision == "regen" else decision,
                    gateway_status=sec_chat.event.status,
                    security_decision=decision,
                    findings=[f.to_dict() for f in findings],
                )
            )

            if decision == "regen":
                security_feedback = feedback_from_findings(findings)
                continue

            if decision == "warn":
                result.warnings.append(
                    "Medium/Low security findings: "
                    + "; ".join(f"{f.severity}:{f.detail}" for f in findings)
                )

            # --- commit (dry-run) ---
            dest = self.committed / f"{task.id}_{task.filename}"
            shutil.copy2(path, dest)
            result.ok = True
            result.commit_status = "committed"
            result.final_path = str(dest)
            result.iterations.append(
                IterationRecord(
                    n=n,
                    phase="commit",
                    ok=True,
                    detail=f"dry-run commit → {dest.name}",
                    security_decision=decision,
                )
            )
            break
        else:
            result.commit_status = "failed"

        result.elapsed_sec = round(time.perf_counter() - t0, 3)
        result.gateway_caught = sorted(
            {
                f
                for ev in result.gateway_events
                for f in (ev.get("findings") or [])
                if ev.get("status") in {"blocked", "redacted"} or ev.get("blocked")
            }
        )
        result.gateway_clean_count = sum(
            1 for ev in result.gateway_events if ev.get("status") == "clean"
        )
        # smells present in first insecure artifact vs caught
        result.missed_by_both = self._missed_by_both(task, result)
        return result

    def _missed_by_both(self, task: TaskSpec, result: TaskResult) -> list[str]:
        """Expected insecure smells from fixture that neither layer reported."""
        expected = {f.kind for f in scan_code(task.insecure_fixture)}
        sec = set(result.security_caught)
        gw = set(result.gateway_caught)
        # Map gateway kinds loosely: api_key etc. vs hardcoded_api_key
        gw_mapped = set(gw)
        for g in gw:
            if "api" in g or "key" in g or "token" in g or "secret" in g:
                gw_mapped.add("hardcoded_api_key")
                gw_mapped.add("auth_in_source")
        missed = sorted(expected - sec - gw_mapped)
        return missed

    def _write_artifacts(self, results: list[TaskResult]) -> None:
        self.art_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "offline": self.offline,
            "max_iters": self.max_iters,
            "tasks": [r.to_dict() for r in results],
            "summary": {
                "committed": sum(1 for r in results if r.commit_status == "committed"),
                "failed": sum(1 for r in results if r.commit_status == "failed"),
                "blocked": sum(1 for r in results if r.commit_status == "blocked"),
            },
        }
        (self.art_dir / "results.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._write_execution_log(results)
        self._write_caught_table(results)

    def _write_execution_log(self, results: list[TaskResult]) -> None:
        lines = [
            "# Execution log day_49 — security step + LLM gateway",
            "",
            f"Offline: `{self.offline}` · max_iters={self.max_iters}",
            "",
        ]
        for r in results:
            lines.append(f"## {r.task_id}")
            lines.append(f"- prompt: {r.prompt}")
            lines.append(f"- commit_status: **{r.commit_status}** · ok={r.ok} · {r.elapsed_sec}s")
            if r.warnings:
                lines.append(f"- warnings: {'; '.join(r.warnings)}")
            lines.append("- iterations:")
            for it in r.iterations:
                lines.append(
                    f"  - [{it.n}] {it.phase}: ok={it.ok} gw={it.gateway_status or '-'} "
                    f"sec={it.security_decision or '-'} — {it.detail[:120]}"
                )
            lines.append(f"- security_caught: {r.security_caught or '—'}")
            lines.append(f"- gateway_caught: {r.gateway_caught or '—'}")
            lines.append(f"- missed_by_both: {r.missed_by_both or '—'}")
            lines.append("")
        (self.art_dir / "execution_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_caught_table(self, results: list[TaskResult]) -> None:
        lines = [
            "# Caught vs missed — day_49",
            "",
            "| Task | Security step | Gateway | Missed by both | Commit |",
            "|------|---------------|---------|----------------|--------|",
        ]
        for r in results:
            lines.append(
                "| {task} | {sec} | {gw} | {miss} | {st} |".format(
                    task=r.task_id,
                    sec=", ".join(r.security_caught) or "—",
                    gw=", ".join(r.gateway_caught) or "clean",
                    miss=", ".join(r.missed_by_both) or "—",
                    st=r.commit_status,
                )
            )
        lines.append("")
        lines.append(
            "Security step = heuristic+LLM code review (Critical/High → regen). "
            "Gateway = input/output secret/PII guards on all LLM calls."
        )
        (self.art_dir / "caught_vs_missed.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def run_offline(
    art_dir: Path | None = None,
    *,
    max_iters: int = 3,
) -> list[TaskResult]:
    loop = SecurityLoop(art_dir=art_dir, offline=True, max_iters=max_iters)
    return loop.run()
