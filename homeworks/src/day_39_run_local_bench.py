#!/usr/bin/env python3
"""Run day_39 feature + agent prompts against local Ollama models."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "homeworks" / "artifacts" / "day_39"
RULES = (ROOT / ".continue" / "rules" / "hw01-agents.md").read_text(encoding="utf-8")

MODELS = [
    "qwen2.5-coder:7b",
    "deepseek-coder:6.7b",
    "qwen2.5-coder:3b",
]

FEATURE_PROMPT = """Сгенерируй полный файл app/services/slug_normalize_service.py для hw01.

Требования:
- from __future__ import annotations
- frozen dataclass SlugNormalizeResult с полями: ok: bool, slug: str, detail: str
- класс SlugNormalizeService:
  - __init__(self, max_length: int = 64)
  - normalize(self, text: str) -> SlugNormalizeResult
  - поведение: trim, lower, заменить пробелы/не-алфанумерик на "-", схлопнуть "--", обрезать по max_length, strip "-" по краям
  - пустой вход -> ValueError("text must not be empty")
- без print, без SQL, без I/O
- только код файла, без пояснений вокруг
"""

PERSONALIZATION_SNIPPET = '''
```python
# app/services/personalization_service.py (фрагмент)
class PersonalizationService:
    REQUIRED_PROFILE_KEYS = ("role", "stack", "answer_detail", "answer_format")

    def _ensure_loaded(self) -> None:
        if self._loaded or not self._user_id:
            return
        self._profile, self._interview_completed = load_user_profile(self._user_db_path, self._user_id)
        self._loaded = True

    def _all_required_fields_present(self, profile: dict[str, str]) -> bool:
        for key in self.REQUIRED_PROFILE_KEYS:
            if not profile.get(key, "").strip():
                return False
        return True

    def _is_interview_completed(self) -> bool:
        return self._all_required_fields_present(self._profile)

    def _refresh_completion_flag(self) -> None:
        if not self._user_id:
            return
        self._loaded = False
        self._ensure_loaded()
        completed = self._is_interview_completed()
        if completed != self._interview_completed:
            set_user_interview_completed(self._user_db_path, self._user_id, completed)
            self._interview_completed = completed

    def needs_interview(self) -> bool:
        if not self._user_id:
            return False
        self._ensure_loaded()
        return not self._is_interview_completed() or not self._interview_completed

    def save_interview_answers(self, answers: dict[str, str]) -> None:
        user_id = self._require_user_id()
        self.ensure_user_exists()
        upsert_user_profile_entries(self._user_db_path, user_id, answers)
        self._refresh_completion_flag()
```
'''

AGENT_PROMPT = f"""Режим Bug Fix (только анализ, код не пиши целиком).

Симптом: PersonalizationService.needs_interview() иногда возвращает True даже после того,
как пользователь заполнил все REQUIRED_PROFILE_KEYS через save_interview_answers.

Ниже фрагмент app/services/personalization_service.py. Найди вероятную причину и план фикса.

{PERSONALIZATION_SNIPPET}

Ответ строго в формате:
Причина: ...
Что починить: ...
Что проверить: ...
Риски: ...
"""

OLLAMA = "http://127.0.0.1:11434/api/chat"


def chat(model: str, user: str, *, temperature: float = 0.2, top_p: float = 0.9) -> dict:
    payload = {
        "model": model,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_ctx": 8192,
        },
        "messages": [
            {"role": "system", "content": RULES},
            {"role": "user", "content": user},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - started
    content = body.get("message", {}).get("content", "")
    return {
        "model": model,
        "elapsed_sec": round(elapsed, 2),
        "eval_count": body.get("eval_count"),
        "prompt_eval_count": body.get("prompt_eval_count"),
        "eval_duration_ns": body.get("eval_duration"),
        "content": content,
    }


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    for model in MODELS:
        safe = model.replace(":", "_").replace(".", "_")
        print(f"=== feature {model} ===", flush=True)
        try:
            feat = chat(model, FEATURE_PROMPT)
        except urllib.error.URLError as exc:
            feat = {"model": model, "error": str(exc), "elapsed_sec": None, "content": ""}
        (ART / f"feature_{safe}.md").write_text(
            f"# Feature — {model}\n\n"
            f"- elapsed_sec: {feat.get('elapsed_sec')}\n"
            f"- prompt_eval_count: {feat.get('prompt_eval_count')}\n"
            f"- eval_count: {feat.get('eval_count')}\n\n"
            f"```\n{feat.get('content', '')}\n```\n",
            encoding="utf-8",
        )
        print(f"  done in {feat.get('elapsed_sec')}s", flush=True)

        print(f"=== agent {model} ===", flush=True)
        try:
            ag = chat(model, AGENT_PROMPT)
        except urllib.error.URLError as exc:
            ag = {"model": model, "error": str(exc), "elapsed_sec": None, "content": ""}
        (ART / f"agent_{safe}.md").write_text(
            f"# Agent — {model}\n\n"
            f"- elapsed_sec: {ag.get('elapsed_sec')}\n"
            f"- prompt_eval_count: {ag.get('prompt_eval_count')}\n"
            f"- eval_count: {ag.get('eval_count')}\n\n"
            f"{ag.get('content', '')}\n",
            encoding="utf-8",
        )
        print(f"  done in {ag.get('elapsed_sec')}s", flush=True)

        summary.append(
            {
                "model": model,
                "feature_elapsed_sec": feat.get("elapsed_sec"),
                "feature_eval_count": feat.get("eval_count"),
                "agent_elapsed_sec": ag.get("elapsed_sec"),
                "agent_eval_count": ag.get("eval_count"),
                "feature_error": feat.get("error"),
                "agent_error": ag.get("error"),
            }
        )

    (ART / "timings.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", ART / "timings.json", flush=True)


if __name__ == "__main__":
    main()
