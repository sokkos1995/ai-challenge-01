"""Unit tests for day_49 security loop (no live LLM)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "homeworks" / "src"))

from day_49_security_loop.gateway_client import GatewayClient  # noqa: E402
from day_49_security_loop.loop import (  # noqa: E402
    SecurityLoop,
    decision_for,
    extract_python,
    merge_findings,
    parse_security_json,
    run_offline,
    verify_task,
)
from day_49_security_loop.security_heuristics import (  # noqa: E402
    needs_regen,
    scan_code,
)
from day_49_security_loop.tasks import TASKS, get_task  # noqa: E402


def test_heuristic_catches_hardcoded_sk():
    code = 'API_KEY = "sk-proj-abc1234567890xyzDEMO"\n'
    findings = scan_code(code)
    assert any(f.kind == "hardcoded_api_key" for f in findings)
    assert needs_regen(findings)


def test_heuristic_catches_sql_fstring():
    code = 'cur.execute(f"SELECT * FROM t WHERE id={user_id}")\n'
    findings = scan_code(code)
    assert any(f.kind == "sql_injection_fstring" for f in findings)
    assert decision_for(findings) == "regen"


def test_heuristic_catches_http_not_https():
    code = 'URL = "http://api.example.com/v1"\n'
    findings = scan_code(code)
    assert any(f.kind == "http_not_https" for f in findings)
    assert decision_for(findings) == "warn"


def test_medium_is_warn_not_regen():
    findings = scan_code('URL = "http://api.example.com/v1"\n')
    assert decision_for(findings) == "warn"


def test_clean_code_commits():
    code = "def ok() -> int:\n    return 1\n"
    assert decision_for(scan_code(code)) == "commit"


def test_extract_python_from_fence():
    raw = "Here:\n```python\ndef f():\n    return 1\n```\n"
    assert "def f()" in extract_python(raw)


def test_parse_security_json():
    raw = '{"findings":[{"severity":"High","line":3,"detail":"token in logs"}]}'
    findings = parse_security_json(raw)
    assert len(findings) == 1
    assert findings[0].severity == "High"


def test_merge_prefers_heuristic_when_llm_empty():
    heur = scan_code('K = "sk-proj-abc1234567890xyzDEMO"\n')
    merged = merge_findings([], heur)
    assert merged == heur


def test_merge_upgrades_llm_low_severity_with_heuristic_critical():
    from day_49_security_loop.security_heuristics import SecurityFinding

    llm = [
        SecurityFinding(
            severity="Low",
            line=1,
            detail="test-only fixture, skip",
            kind="hardcoded_api_key",
            source="llm",
        )
    ]
    heur = scan_code('K = "sk-proj-abc1234567890xyzDEMO"\n')
    merged = merge_findings(llm, heur)
    assert any(f.kind == "hardcoded_api_key" and f.severity == "Critical" for f in merged)
    assert decision_for(merged) == "regen"


def test_scan_code_catches_comment_and_zw_split_keys():
    commented = 'K = "sk-" /* x */ + "proj-abc1234567890xyzDEMO"\n'
    zw = 'K = "sk-\u200bproj-abc1234567890xyzDEMO"\n'
    assert any(f.kind == "hardcoded_api_key" for f in scan_code(commented))
    assert any(f.kind == "hardcoded_api_key" for f in scan_code(zw))


def test_gateway_in_process_blocks_secret_in_block_mode():
    def boom(messages, model=None):
        raise AssertionError("completer must not run when blocked")

    client = GatewayClient(mode="block", in_process=boom, use_input_guard=True)
    result = client.chat(
        prompt="my key is sk-proj-abc1234567890xyzDEMO",
        stage="test",
    )
    assert result.event.blocked
    assert result.event.status == "blocked"
    assert "api_key" in result.event.findings


def test_gateway_in_process_redacts_and_passes():
    seen: list[str] = []

    def echo(messages, model=None):
        user = next(m["content"] for m in messages if m["role"] == "user")
        seen.append(user)
        return "ok"

    client = GatewayClient(mode="redact", in_process=echo, use_input_guard=True)
    result = client.chat(
        prompt="email me at alice@example.com please",
        stage="test",
    )
    assert not result.event.blocked
    assert result.event.status == "redacted"
    assert "email" in result.event.findings
    assert "[REDACTED_EMAIL]" in seen[0]


def test_verify_insecure_fixtures(tmp_path: Path):
    for task in TASKS:
        path = tmp_path / task.filename
        path.write_text(task.insecure_fixture, encoding="utf-8")
        ok, detail = verify_task(task, path)
        assert ok, f"{task.id}: {detail}"


def test_offline_loop_writes_results(tmp_path: Path):
    art = tmp_path / "day_49"
    results = run_offline(art_dir=art, max_iters=3)
    assert len(results) == 3
    assert all(r.commit_status == "committed" for r in results)
    payload = json.loads((art / "results.json").read_text(encoding="utf-8"))
    assert payload["summary"]["committed"] == 3
    for row in payload["tasks"]:
        assert "security_findings" in row
        assert "gateway_events" in row
        assert "commit_status" in row
        assert row["security_caught"] or row["gateway_caught"]
    assert (art / "execution_log.md").is_file()
    assert (art / "caught_vs_missed.md").is_file()
    for task in TASKS:
        assert (art / "committed" / f"{task.id}_{task.filename}").is_file()


def test_offline_loop_regens_on_critical(tmp_path: Path):
    art = tmp_path / "day_49"
    loop = SecurityLoop(art_dir=art, offline=True, max_iters=3)
    task = get_task("save_auth_token")
    result = loop.run_task(task)
    assert any(i.security_decision == "regen" for i in result.iterations)
    assert result.commit_status == "committed"
    final = Path(result.final_path).read_text(encoding="utf-8")
    assert "sk-proj-abc1234567890xyzDEMO" not in final
