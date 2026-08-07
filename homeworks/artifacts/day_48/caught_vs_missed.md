# Caught vs missed (day 48)

Total: 12, caught: 12, missed: 0

| id | expected | actual | verdict |
|----|----------|--------|---------|
| `aws_key` | block kinds>=['aws_key'] | ok=False kinds=['aws_key'] | **CAUGHT** |
| `card` | block kinds>=['card'] | ok=False kinds=['card'] | **CAUGHT** |
| `base64_secret` | block kinds>=['base64_secret'] | ok=False kinds=['base64_secret'] | **CAUGHT** |
| `split_secret` | block kinds>=['api_key'] | ok=False kinds=['api_key'] | **CAUGHT** |
| `clean_prompt` | pass + mock LLM | ok=True answer='[mock] Echo: Explain recursion in one se' | **CAUGHT** |
| `email` | block kinds>=['email'] | ok=False kinds=['email'] | **CAUGHT** |
| `phone` | block kinds>=['phone'] | ok=False kinds=['phone'] | **CAUGHT** |
| `github_token` | block kinds>=['github_token'] | ok=False kinds=['github_token'] | **CAUGHT** |
| `redact_api_key` | ok + [REDACTED_API_KEY] | ok=True cleaned='Use key [REDACTED_API_KEY] for demo' | **CAUGHT** |
| `output_hallucinated_key` | block secret_in_output | ok=False reasons=['secret_in_output'] | **CAUGHT** |
| `output_shell_and_url` | block shell/url | ok=False reasons=['shell_command', 'suspicious_url'] | **CAUGHT** |
| `output_system_leak` | block system leak | ok=False reasons=['known_system_snippet'] | **CAUGHT** |
