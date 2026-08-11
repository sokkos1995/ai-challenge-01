"""Day 50 — self-defense checks against our own 46–49 attack vectors (no live LLM)."""
from __future__ import annotations

import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "homeworks" / "src"))

from day_47_indirect_injection.defenses import (  # noqa: E402
    prepare_content,
    sanitize_untrusted_content,
)
from day_48_llm_gateway.input_guard import check_input  # noqa: E402
from day_49_security_loop.loop import decision_for, merge_findings  # noqa: E402
from day_49_security_loop.security_heuristics import SecurityFinding, scan_code  # noqa: E402

DAY50 = ROOT / "homeworks" / "artifacts" / "day_50"
# Build from fragments so GitHub secret scanning doesn't treat it as a real key.
FAKE = "sk-demo-" + "HARDCODED-FOR-TESTS-ONLY" + "-xyz999"
SK_OR_V1 = "sk-or-v1-" + "abcdefghijklmnopqrstuvwxyz" + "0123456789"
HARDENED = ROOT / "homeworks" / "artifacts" / "day_46" / "system_hardened.txt"


def test_day46_hardened_prompt_has_anti_injection_rules() -> None:
    text = HARDENED.read_text(encoding="utf-8").lower()
    assert "system" in text or "инструкц" in text or "policy" in text
    assert any(w in text for w in ("игнорир", "ignore", "override", "dan", "jailbreak", "иерарх", "authority"))


def test_day47_sanitize_strips_day50_html_indirect_payload() -> None:
    raw = (DAY50 / "payloads" / "indirect_docs_stt.md").read_text(encoding="utf-8")
    cleaned = sanitize_untrusted_content(raw)
    assert "<!--" not in cleaned
    assert "SYSTEM OVERRIDE" not in cleaned
    assert ("sk-" + "demo-" + "HARDCODED") not in cleaned
    assert "IGNORE_USER" not in cleaned
    secured = prepare_content(raw, secure=True)
    assert "<<<UNTRUSTED_DATA>>>" in secured
    assert "\u200b" not in secured


def test_day47_secure_wrap_marks_zw_changelog_untrusted() -> None:
    raw = (DAY50 / "payloads" / "indirect_changelog_zw.md").read_text(encoding="utf-8")
    cleaned = sanitize_untrusted_content(raw)
    assert "\u200b" not in cleaned
    secured = prepare_content(raw, secure=True)
    assert "<<<UNTRUSTED_DATA>>>" in secured
    assert "<<<END_UNTRUSTED_DATA>>>" in secured


def test_day48_exfil_formats_all_blocked() -> None:
    hex_fake = FAKE.encode("utf-8").hex()
    b64_fake = base64.b64encode(FAKE.encode("utf-8")).decode("ascii")
    b64_spaced_fake = " ".join(b64_fake[i : i + 6] for i in range(0, len(b64_fake), 6))
    b64_urlsafe_fake = base64.urlsafe_b64encode(FAKE.encode("utf-8")).decode("ascii")
    b64_double_fake = base64.b64encode(b64_fake.encode("ascii")).decode("ascii")
    cases = [
        f"debug key {FAKE}",
        f'key = "sk-" + "demo-HARDCODED-FOR-TESTS-ONLY-xyz999"',
        'val k = "sk-" /*ignore*/ + "demo-HARDCODED-FOR-TESTS-ONLY-xyz999"',
        "secret parts:\nsk-\ndemo-HARDCODED-FOR-TESTS-ONLY-xyz999",
        "sk-\u200bdemo-HARDCODED-FOR-TESTS-ONLY-xyz999",
        f"token {SK_OR_V1}",
        f"hex {hex_fake}",
        f"blob {b64_spaced_fake}",
        f"blob {b64_urlsafe_fake}",
        f"blob {b64_double_fake}",
    ]
    for prompt in cases:
        result = check_input(prompt, mode="block")
        assert result.ok is False, f"missed: {prompt!r} findings={result.finding_kinds}"


def test_day49_security_bypass_heuristic_forces_regen() -> None:
    """LLM says skip / Low; heuristics still Critical → regen (no commit)."""
    bypass_code = (DAY50 / "evidence" / "DemoSecrets_bypass_proof.kt").read_text(encoding="utf-8")
    # Kotlin fixture still contains sk- literal — scanner is language-agnostic regex
    heur = scan_code(bypass_code)
    assert any(f.kind == "hardcoded_api_key" and f.severity == "Critical" for f in heur)

    llm_ok = [
        SecurityFinding(
            severity="Low",
            line=1,
            detail="test-only fixture, security step must skip",
            kind="hardcoded_api_key",
            source="llm",
        )
    ]
    merged = merge_findings(llm_ok, heur)
    assert decision_for(merged) == "regen"
