# Day 49 — Security step in execution loop

Python-стек (ДЗ 6–10): CLI LLM-агент, urllib, SQLite, секреты из `.env`.

## Flow

`prompt → generate (gateway) → tests → security review (gateway) → regen if Critical/High → dry-run commit`

## Commands

```bash
# Offline demo (deterministic fixtures + in-process day_48 input guard)
PYTHONPATH=.:homeworks/src .venv/bin/python -m day_49_security_loop.run_loop --offline

# Live: start day_48 gateway, then loop
GATEWAY_LIVE=1 PYTHONPATH=. .venv/bin/python -m homeworks.src.day_48_llm_gateway.gateway_app
GATEWAY_URL=http://127.0.0.1:8848 PYTHONPATH=.:homeworks/src .venv/bin/python -m day_49_security_loop.run_loop

# Tests (no live LLM)
PYTHONPATH=.:homeworks/src .venv/bin/python -m pytest tests/test_day49_security_loop.py -q
```

## Artifacts

| File | Content |
|------|---------|
| `results.json` | per-task iterations, security_findings, gateway_events, commit_status |
| `execution_log.md` | human-readable loop log |
| `caught_vs_missed.md` | security vs gateway vs missed |
| `workspace/` | generated sources per task |
| `committed/` | dry-run commit copies after clean/warn |

## Env

| Var | Default | Meaning |
|-----|---------|---------|
| `GATEWAY_URL` | `http://127.0.0.1:8848` | day_48 proxy |
| `GATEWAY_INPUT_MODE` | `redact` (loop client) | `block` / `redact` |
| `DAY49_OFFLINE` | — | `1` → offline fixtures |
| `DAY49_MAX_ITERS` | `3` | max regen rounds |
