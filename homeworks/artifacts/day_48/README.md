# Day 48 — LLM Gateway

## Commands

```bash
pip install -r homeworks/artifacts/day_48/requirements.txt

# start FastAPI gateway (mock LLM by default)
PYTHONPATH=. .venv/bin/python -m homeworks.src.day_48_llm_gateway.gateway_app
# http://127.0.0.1:8848/health
# POST http://127.0.0.1:8848/v1/chat

# offline case table
.venv/bin/python homeworks/src/day_48_llm_gateway/run_cases.py

# unit tests
.venv/bin/python -m pytest tests/test_day48_guards.py -q

# optional live upstream (OpenRouter/Groq via .env)
GATEWAY_LIVE=1 PYTHONPATH=. .venv/bin/python -m homeworks.src.day_48_llm_gateway.gateway_app
```

## Pricing (cost tracking)

Default ballpark USD / 1M tokens (override via env):

| | env | default |
|--|-----|---------|
| prompt | `GATEWAY_PRICE_PROMPT_PER_MTOK` | 0.10 |
| completion | `GATEWAY_PRICE_COMPLETION_PER_MTOK` | 0.40 |

## Rate limit

`GATEWAY_RATE_LIMIT_PER_MIN` (default 30) per client IP.

## Audit

JSONL at `artifacts/day_48/audit.jsonl` (hashes + redacted previews, no raw secrets).
