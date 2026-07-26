#!/usr/bin/env python3
"""day_40 local execution loop: Ollama generates solutions, harness verifies Done."""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "homeworks" / "artifacts" / "day_40_local"
RULES = (ROOT / ".continue" / "rules" / "hw01-agents.md").read_text(encoding="utf-8")
LOOP_RULE = (ROOT / ".cursor" / "rules" / "execution-loop.mdc").read_text(encoding="utf-8")
OLLAMA = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen2.5-coder:7b"
SYS = RULES + "\n\n" + LOOP_RULE + "\n\nОтвечай только кодом файла (или markdown для docs). Без пояснений вокруг."


@dataclass
class TaskSpec:
    key: str
    kind: str  # bug|feature|refactor|test|docs
    relpath: str
    done: str
    prompt_extra: str = ""
    starter: str | None = None


TASKS: list[TaskSpec] = [
    TaskSpec(
        "broken_avg",
        "bug",
        "broken_avg.py",
        "average([]) is None; average([1,3])==2; no ZeroDivisionError",
        starter='''"""Broken average."""
from __future__ import annotations

def average(nums: list[float]) -> float | None:
    return sum(nums) / len(nums)
''',
    ),
    TaskSpec(
        "broken_slice",
        "bug",
        "broken_slice.py",
        "last_n([1,2,3,4,5],2)==[4,5]; last_n([1,2],5)==[1,2]",
        starter='''"""Broken last_n."""
from __future__ import annotations
from typing import TypeVar
T = TypeVar("T")

def last_n(items: list[T], n: int) -> list[T]:
    if n <= 0:
        return []
    return items[-(n + 1):]
''',
    ),
    TaskSpec(
        "broken_append",
        "bug",
        "broken_append.py",
        "append_item('a')==['a'] and append_item('b')==['b'] (no shared default)",
        starter='''"""Broken append_item."""
from __future__ import annotations

def append_item(value: str, bucket: list[str] = []) -> list[str]:
    bucket.append(value)
    return bucket
''',
    ),
    TaskSpec(
        "fizzbuzz",
        "feature",
        "fizzbuzz.py",
        "fizzbuzz(3)==['1','2','Fizz']; CLI without N prints usage and exit 2; main(['5']) prints 1..Buzz",
        prompt_extra="Provide fizzbuzz(n)->list[str] and main(argv=None)->int CLI.",
    ),
    TaskSpec(
        "wordcount",
        "feature",
        "wordcount.py",
        "count_text + argparse CLI with optional path and --help; words/lines/chars",
        prompt_extra="Provide count_text(text)->(words,lines,chars) and main(argv=None)->int.",
    ),
    TaskSpec(
        "json_pretty",
        "feature",
        "json_pretty.py",
        "pretty-print JSON indent=2; invalid JSON -> stderr + exit 1",
        prompt_extra="Provide main(argv=None)->int reading file arg or stdin.",
    ),
    TaskSpec(
        "temp_convert",
        "feature",
        "temp_convert.py",
        "c2f/f2c one number round 1 decimal; else usage exit 2",
        prompt_extra="Provide c2f, f2c, main(argv=None)->int.",
    ),
    TaskSpec(
        "csv_to_md",
        "feature",
        "csv_to_md.py",
        "first row header; GitHub markdown table to stdout",
        prompt_extra="Provide csv_to_md(text)->str and main(argv=None)->int.",
    ),
    TaskSpec(
        "password_gen",
        "feature",
        "password_gen.py",
        "--length N default 16; length<8 -> exit 2; letters+digits+punctuation",
        prompt_extra="Provide generate(length)->str and main(argv=None)->int with argparse.",
    ),
    TaskSpec(
        "anagrams",
        "feature",
        "anagrams.py",
        "two strings -> YES/NO ignoring case and spaces",
        prompt_extra="Provide is_anagram and main(argv=None)->int.",
    ),
    TaskSpec(
        "monolith_stats",
        "refactor",
        "monolith_stats.py",
        "public mean/median/mode + CLI preserving stats behavior",
        starter='''"""Monolith stats."""
from __future__ import annotations
import sys
from collections import Counter

def stats(nums: list[float]) -> dict[str, float | None]:
    if not nums:
        return {"mean": None, "median": None, "mode": None}
    mean = sum(nums) / len(nums)
    ordered = sorted(nums)
    mid = len(ordered) // 2
    median = float(ordered[mid]) if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    mode = float(Counter(ordered).most_common(1)[0][0])
    return {"mean": mean, "median": median, "mode": mode}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: monolith_stats.py <num> [num...]", file=sys.stderr)
        raise SystemExit(2)
    result = stats([float(x) for x in sys.argv[1:]])
    print(f"mean={result['mean']} median={result['median']} mode={result['mode']}")
''',
    ),
    TaskSpec(
        "validate_email",
        "refactor",
        "validate_email.py",
        "is_valid_email(s)->bool used by CLI; no duplicated regex in main",
        starter='''"""Email CLI without helper."""
from __future__ import annotations
import re, sys
_PATTERN = r"^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$"

def main() -> None:
    if len(sys.argv) != 2:
        print("usage: validate_email.py <email>", file=sys.stderr)
        raise SystemExit(2)
    if re.match(_PATTERN, sys.argv[1]):
        print("OK")
    else:
        print("INVALID")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
''',
    ),
    TaskSpec(
        "student_grade",
        "refactor",
        "student_grade.py",
        "@dataclass Student with average() method; CLI prints name and average",
        starter='''"""Student as dict."""
from __future__ import annotations
import json, sys

def average(student: dict) -> float:
    grades = student["grades"]
    return 0.0 if not grades else sum(grades) / len(grades)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('usage: student_grade.py \\'{"name":"Ada","grades":[5,4,5]}\\'', file=sys.stderr)
        raise SystemExit(2)
    data = json.loads(sys.argv[1])
    print(f"{data['name']}: {average(data):.2f}")
''',
    ),
    TaskSpec(
        "test_fizzbuzz",
        "test",
        "tests/test_day40_local_fizzbuzz.py",
        "pytest file with >=3 cases for day_40_local/fizzbuzz.py green",
        prompt_extra="Write pytest module loading homeworks/artifacts/day_40_local/fizzbuzz.py via importlib.",
    ),
    TaskSpec(
        "test_wordcount",
        "test",
        "tests/test_day40_local_wordcount.py",
        "pytest covers file + stdin for wordcount green",
        prompt_extra="Write pytest module loading homeworks/artifacts/day_40_local/wordcount.py via importlib.",
    ),
    TaskSpec(
        "test_bugfixes",
        "test",
        "tests/test_day40_local_bugfixes.py",
        "pytest covers average/last_n/append_item green",
        prompt_extra="Write pytest loading broken_avg.py, broken_slice.py, broken_append.py from day_40_local.",
    ),
    TaskSpec(
        "readme",
        "docs",
        "README.md",
        "markdown list of scripts with one example command each",
    ),
    TaskSpec(
        "typing_cheatsheet",
        "docs",
        "typing_cheatsheet.md",
        ">=8 typing annotation examples (list/dict/Optional/Union/Callable/TypedDict/...)",
    ),
]


def chat(model: str, user: str) -> dict:
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0.1, "top_p": 0.9, "num_ctx": 8192},
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": user},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - started
    content = body.get("message", {}).get("content", "") or ""
    return {"elapsed_sec": round(elapsed, 2), "content": content, "eval_count": body.get("eval_count")}


def extract_content(raw: str, *, docs: bool) -> str:
    fences = re.findall(r"```(?:python|py|markdown|md)?\n(.*?)```", raw, flags=re.S | re.I)
    if fences:
        # pick largest fence
        return max(fences, key=len).strip() + "\n"
    return raw.strip() + "\n"


def build_prompt(task: TaskSpec, path: Path) -> str:
    parts = [
        f"Task type: {task.kind}",
        f"Write the FULL file for: {path.relative_to(ROOT)}",
        f"Done criteria: {task.done}",
    ]
    if task.prompt_extra:
        parts.append(task.prompt_extra)
    if task.starter:
        parts.append("Current (broken/incomplete) file:\n```python\n" + task.starter + "\n```")
        parts.append("Fix/refactor this file in place. Return the full corrected file.")
    else:
        parts.append("Create the file from scratch. Return the full file contents only.")
    return "\n\n".join(parts)


def run_py(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )


def verify(task: TaskSpec, path: Path) -> tuple[bool, str]:
    key = task.key
    try:
        if key == "broken_avg":
            p = run_py(
                "import importlib.util\n"
                f"s=importlib.util.spec_from_file_location('m', r'{path}')\n"
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
                "assert m.average([]) is None\n"
                "assert m.average([1.0,3.0])==2.0\n"
            )
            return p.returncode == 0, p.stderr or p.stdout
        if key == "broken_slice":
            p = run_py(
                "import importlib.util\n"
                f"s=importlib.util.spec_from_file_location('m', r'{path}')\n"
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
                "assert m.last_n([1,2,3,4,5],2)==[4,5]\n"
                "assert m.last_n([1,2],5)==[1,2]\n"
            )
            return p.returncode == 0, p.stderr or p.stdout
        if key == "broken_append":
            p = run_py(
                "import importlib.util\n"
                f"s=importlib.util.spec_from_file_location('m', r'{path}')\n"
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
                "assert m.append_item('a')==['a']\n"
                "assert m.append_item('b')==['b']\n"
            )
            return p.returncode == 0, p.stderr or p.stdout
        if key == "fizzbuzz":
            p = run_py(
                "import importlib.util,sys\n"
                f"s=importlib.util.spec_from_file_location('m', r'{path}')\n"
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
                "assert m.fizzbuzz(3)==['1','2','Fizz']\n"
                "assert m.main([])==2\n"
                "assert m.main(['5'])==0\n"
            )
            return p.returncode == 0, p.stderr or p.stdout
        if key == "wordcount":
            p = run_py(
                "import importlib.util\n"
                f"s=importlib.util.spec_from_file_location('m', r'{path}')\n"
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
                "w,l,c=m.count_text('hello world\\nfoo')\n"
                "assert w==3 and l==2 and c==15\n"
            )
            p2 = subprocess.run(
                [sys.executable, str(path), "--help"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=10,
            )
            help_text = (p2.stdout + p2.stderr).lower()
            ok = p.returncode == 0 and p2.returncode == 0 and ("usage" in help_text or "help" in help_text)
            return ok, (p.stderr or p.stdout) + p2.stdout
        if key == "json_pretty":
            p = subprocess.run(
                [sys.executable, str(path)],
                input='{"a":1}\n',
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=10,
            )
            p_bad = subprocess.run(
                [sys.executable, str(path)],
                input="not-json\n",
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=10,
            )
            ok = p.returncode == 0 and '"a"' in p.stdout and p_bad.returncode == 1 and bool(p_bad.stderr)
            return ok, p.stdout + p_bad.stderr
        if key == "temp_convert":
            p = subprocess.run(
                [sys.executable, str(path), "c2f", "0"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=10,
            )
            p2 = subprocess.run(
                [sys.executable, str(path)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=10,
            )
            ok = p.returncode == 0 and "32" in p.stdout and p2.returncode == 2
            return ok, p.stdout + p2.stderr
        if key == "csv_to_md":
            p = subprocess.run(
                [sys.executable, str(path)],
                input="h1,h2\na,b\n",
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=10,
            )
            ok = p.returncode == 0 and "| h1 | h2 |" in p.stdout and "---" in p.stdout
            return ok, p.stdout + p.stderr
        if key == "password_gen":
            p = subprocess.run(
                [sys.executable, str(path), "--length", "12"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=10,
            )
            p2 = subprocess.run(
                [sys.executable, str(path), "--length", "4"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=10,
            )
            out = p.stdout.strip()
            ok = p.returncode == 0 and len(out) == 12 and p2.returncode == 2
            return ok, p.stdout + p2.stderr
        if key == "anagrams":
            p = subprocess.run(
                [sys.executable, str(path), "Listen", "Silent"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=10,
            )
            p2 = subprocess.run(
                [sys.executable, str(path), "ab", "ac"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=10,
            )
            ok = p.returncode == 0 and p.stdout.strip() == "YES" and p2.stdout.strip() == "NO"
            return ok, p.stdout + p2.stdout
        if key == "monolith_stats":
            p = run_py(
                "import importlib.util\n"
                f"s=importlib.util.spec_from_file_location('m', r'{path}')\n"
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
                "assert hasattr(m,'mean') and hasattr(m,'median') and hasattr(m,'mode')\n"
                "assert m.mean([1,2,3])==2\n"
                "assert m.median([1,2,3,4])==2.5\n"
                "assert m.mode([1,2,2,3])==2.0\n"
            )
            return p.returncode == 0, p.stderr or p.stdout
        if key == "validate_email":
            p = run_py(
                "import importlib.util\n"
                f"s=importlib.util.spec_from_file_location('m', r'{path}')\n"
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
                "assert m.is_valid_email('a@b.com')\n"
                "assert not m.is_valid_email('x')\n"
            )
            return p.returncode == 0, p.stderr or p.stdout
        if key == "student_grade":
            p = run_py(
                "import importlib.util\n"
                f"s=importlib.util.spec_from_file_location('m', r'{path}')\n"
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
                "assert hasattr(m,'Student')\n"
                "s=m.Student(name='Ada', grades=[5,4,5])\n"
                "assert abs(s.average()-4.6667)<0.01\n"
            )
            cli = subprocess.run(
                [sys.executable, str(path), '{"name":"Ada","grades":[5,4,5]}'],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=10,
            )
            ok = p.returncode == 0 and cli.returncode == 0 and "Ada" in cli.stdout
            return ok, (p.stderr or "") + cli.stdout
        if key.startswith("test_"):
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", str(path), "-q"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=60,
            )
            return proc.returncode == 0, proc.stdout + proc.stderr
        if key == "readme":
            text = path.read_text(encoding="utf-8")
            ok = "fizzbuzz" in text.lower() and "python" in text.lower() and len(text) > 80
            return ok, "readme len=%s" % len(text)
        if key == "typing_cheatsheet":
            text = path.read_text(encoding="utf-8")
            markers = ["list[", "dict[", "Optional", "Union", "Callable", "TypedDict"]
            hit = sum(1 for m in markers if m in text)
            ok = hit >= 6 and text.count(":") >= 8
            return ok, f"markers={hit}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    return False, "unknown task"


def resolve_path(task: TaskSpec) -> Path:
    if task.relpath.startswith("tests/"):
        return ROOT / task.relpath
    return ART / task.relpath


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    # seed starters
    for task in TASKS:
        path = resolve_path(task)
        path.parent.mkdir(parents=True, exist_ok=True)
        if task.starter and not path.exists():
            path.write_text(task.starter, encoding="utf-8")

    started = time.time()
    results: list[dict] = []
    streak = 0
    broke_on: str | None = None
    broke_reason: str | None = None
    first_try_ok = 0

    for idx, task in enumerate(TASKS, 1):
        path = resolve_path(task)
        print(f"=== [{idx}/18] {task.kind} {task.key} ===", flush=True)
        prompt = build_prompt(task, path)
        t0 = time.perf_counter()
        try:
            resp = chat(MODEL, prompt)
            err = None
        except urllib.error.URLError as exc:
            resp = {"elapsed_sec": None, "content": "", "eval_count": None}
            err = str(exc)
        gen_sec = time.perf_counter() - t0

        raw = resp.get("content") or ""
        content = extract_content(raw, docs=task.kind == "docs")
        if content.strip():
            path.write_text(content, encoding="utf-8")
        else:
            err = err or "empty model output"

        ok, detail = (False, err or "no content") if err else verify(task, path)
        if ok:
            first_try_ok += 1
            if broke_on is None:
                streak += 1
        else:
            if broke_on is None:
                broke_on = task.key
                broke_reason = (detail or "verify failed")[:500]

        row = {
            "n": idx,
            "key": task.key,
            "kind": task.kind,
            "ok": ok,
            "first_try": ok,
            "gen_sec": round(gen_sec, 2),
            "eval_count": resp.get("eval_count"),
            "error": None if ok else (detail or "")[:500],
            "path": str(path.relative_to(ROOT)),
        }
        results.append(row)
        print(f"  -> {'OK' if ok else 'FAIL'} in {row['gen_sec']}s", flush=True)
        if not ok:
            print(f"  detail: {row['error'][:200]}", flush=True)

    elapsed = time.time() - started
    minutes = max(1, math.ceil(elapsed / 60))
    metrics = {
        "run": 3,
        "model": MODEL,
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_minutes": minutes,
        "tasks_total": len(TASKS),
        "tasks_completed_streak": streak,
        "tasks_ok_total": first_try_ok,
        "broke_on_task": broke_on,
        "broke_reason": broke_reason,
        "first_try_percent": round(100.0 * first_try_ok / len(TASKS), 1),
        "avg_minutes_per_task": round(elapsed / 60 / len(TASKS), 3),
        "results": results,
        "vs_cloud": {
            "run1_minutes": 3,
            "run2_minutes": 2,
            "run1_streak": 18,
            "run2_streak": 18,
            "local_streak": streak,
            "local_minutes": minutes,
        },
    }
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (ROOT / "homeworks" / "artifacts" / "day_40" / "metrics_run3_local.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        f"# Execution log day_40 — прогон 3 (local `{MODEL}`)",
        "",
        f"Без пауз: **{minutes} мин** ({elapsed:.0f} с) · streak={streak}/18 · ok={first_try_ok}/18 · first_try={metrics['first_try_percent']}%",
        "",
        "| # | key | type | result | gen_sec | notes |",
        "|---|-----|------|--------|---------|-------|",
    ]
    for r in results:
        lines.append(
            f"| {r['n']} | {r['key']} | {r['kind']} | {'ok' if r['ok'] else 'fail'} | {r['gen_sec']} | {(r['error'] or '')[:80]} |"
        )
    if broke_on:
        lines += ["", f"Сломался на: `{broke_on}` — {broke_reason}"]
    else:
        lines += ["", "Сломался: нет (18/18)."]
    (ART / "execution_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("elapsed_minutes", "tasks_completed_streak", "broke_on_task", "first_try_percent")}, indent=2))
    return 0 if first_try_ok == len(TASKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
