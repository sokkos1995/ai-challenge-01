# Caught vs missed — day_49

| Task | Security step | Gateway | Missed by both | Commit |
|------|---------------|---------|----------------|--------|
| save_auth_token | hardcoded_api_key | api_key | — | committed |
| log_all_requests | pii_in_logs | clean | — | committed |
| api_request | hardcoded_api_key, http_not_https | api_key | — | committed |

Security step = heuristic+LLM code review (Critical/High → regen). Gateway = input/output secret/PII guards on all LLM calls.
