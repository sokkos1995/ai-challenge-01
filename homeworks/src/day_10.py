import os
import re
import subprocess
from statistics import mean

TOKEN_RE = re.compile(r"^(prompt_tokens_total|dialog_history_tokens|total_tokens)=(.+)$")
LAT_RE = re.compile(r"^latency_ms=(.+)$")

BASE_ENV = os.environ.copy()
BASE_ENV["LLM_CHAT_KEEP_LAST_N"] = BASE_ENV.get("LLM_CHAT_KEEP_LAST_N", "10")
BASE_ENV["LLM_CHAT_SUMMARY_BATCH_SIZE"] = BASE_ENV.get("LLM_CHAT_SUMMARY_BATCH_SIZE", "10")

SCENARIO = [
    "Запомни: мой стек Python + FastAPI + Postgres.",
    "Запомни: дедлайн проекта 15 мая.",
    "Предпочитаю ответы коротко, списком.",
    "Составь план MVP на 2 недели.",
    "Добавь риски и как их снизить.",
    "Что я говорил про стек?",
    "Какой дедлайн?",
    "Переформулируй план в 5 пунктах.",
    "Добавь чеклист на запуск.",
    "Сделай итог: стек, дедлайн, план.",
]

# Для branching: общий префикс + разветвление
PREFIX = SCENARIO[:6]  # до момента checkpoint
BRANCH_1_TAIL = SCENARIO[6:] + ["Оцени полноту: сохранились ли стек и дедлайн?"]
BRANCH_2_TAIL = [
    # тот же смысл, но с легкой вариацией инструкций в ветке 2
    "Какой дедлайн? (ветка 2)",
    "Переформулируй план в 5 пунктах (ветка 2).",
    "Добавь чеклист на запуск (ветка 2).",
    "Сделай итог: стек, дедлайн, план (ветка 2).",
    "Оцени полноту: сохранились ли стек и дедлайн? (ветка 2)",
]

def run_chat(context_strategy: str, stdin_payload: str) -> dict:
    env = BASE_ENV.copy()
    env["LLM_CHAT_HISTORY_PATH"] = f".llm_chat_history_{context_strategy}.db"

    cmd = ["python3", "llm_cli.py", "--chat", "--tokens", "-v", "--context-strategy", context_strategy]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    out, err = proc.communicate(stdin_payload, timeout=600)

    prompt_tokens, history_tokens, total_tokens, latency = [], [], [], []
    for line in err.splitlines():
        m = TOKEN_RE.match(line.strip())
        if m:
            key, raw_val = m.group(1), m.group(2).strip()
            if raw_val.isdigit():
                val = int(raw_val)
                if key == "prompt_tokens_total":
                    prompt_tokens.append(val)
                elif key == "dialog_history_tokens":
                    history_tokens.append(val)
                elif key == "total_tokens":
                    total_tokens.append(val)
        lm = LAT_RE.match(line.strip())
        if lm and lm.group(1).strip().replace(".", "").isdigit():
            raw_lat = lm.group(1).strip()
            if raw_lat.isdigit():
                latency.append(int(raw_lat))

    agent_lines = [ln for ln in out.splitlines() if "agent>" in ln]
    return {
        "prompt_tokens_avg": round(mean(prompt_tokens), 2) if prompt_tokens else None,
        "history_tokens_avg": round(mean(history_tokens), 2) if history_tokens else None,
        "total_tokens_avg": round(mean(total_tokens), 2) if total_tokens else None,
        "latency_ms_avg": round(mean(latency), 2) if latency else None,
        "agent_last_lines": "\n".join(agent_lines[-6:]),
    }

if __name__ == "__main__":
    # 1) sliding
    sliding_payload = "\n".join(SCENARIO + ["exit"]) + "\n"
    res_sliding = run_chat("sliding", sliding_payload)
    print("=== SLIDING ===")
    print(res_sliding)

    # 2) facts
    facts_payload = "\n".join(SCENARIO + ["exit"]) + "\n"
    res_facts = run_chat("facts", facts_payload)
    print("=== FACTS ===")
    print(res_facts)

    # 3) branching
    # Общий префикс -> checkpoint -> fork -> ветка 1 -> switch -> ветка 2 -> exit
    branching_payload = []
    branching_payload.extend(PREFIX)
    branching_payload.append("@checkpoint")
    branching_payload.append("@fork")
    branching_payload.extend(BRANCH_1_TAIL)
    branching_payload.append("@switch 2")
    branching_payload.extend(BRANCH_2_TAIL)
    branching_payload.append("exit")
    branching_payload = "\n".join(branching_payload) + "\n"
    res_branching = run_chat("branching", branching_payload)
    print("=== BRANCHING ===")
    print(res_branching)