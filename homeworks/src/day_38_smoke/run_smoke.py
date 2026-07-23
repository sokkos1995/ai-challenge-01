"""Agent-driven Playwright smoke for day_38 demo web UI.

Playwright MCP / Claude in Mobile were not available in this Cursor workspace
(no Node/npx, no Playwright MCP server). This runner is the local equivalent:
opens pages, clicks, fills forms, screenshots each step, writes a JSON+MD report.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "homeworks" / "artifacts" / "day_38"
BASE_URL = "http://127.0.0.1:8765"
TASK_TITLE = "Smoke task day_38"
LOG_PATH = ARTIFACTS / "webapp_stderr.log"


@dataclass
class StepResult:
    name: str
    ok: bool
    detail: str
    screenshot: str


@dataclass
class ScenarioResult:
    id: str
    title: str
    ok: bool
    steps: list[StepResult] = field(default_factory=list)
    failure: str = ""


def _wait_health(timeout_sec: float = 15.0) -> None:
    deadline = time.time() + timeout_sec
    last_err = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=1) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_err = str(exc)
            time.sleep(0.2)
    raise RuntimeError(f"Web UI did not become ready: {last_err}")


def _shot(page: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(path), full_page=True)  # type: ignore[attr-defined]


def run_smoke() -> list[ScenarioResult]:
    from playwright.sync_api import sync_playwright

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    results: list[ScenarioResult] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1024, "height": 768})

        # --- S1 Login ---
        s1 = ScenarioResult(id="S1", title="Login", ok=True)
        try:
            page.goto(f"{BASE_URL}/login", wait_until="domcontentloaded")
            _shot(page, ARTIFACTS / "s1_01_login_page.png")
            s1.steps.append(StepResult("open_login", True, "/login opened", "s1_01_login_page.png"))

            page.get_by_test_id("login-username").fill("demo")
            page.get_by_test_id("login-password").fill("demo123")
            _shot(page, ARTIFACTS / "s1_02_login_filled.png")
            s1.steps.append(StepResult("fill_credentials", True, "demo/demo123", "s1_02_login_filled.png"))

            page.get_by_test_id("login-submit").click()
            page.wait_for_url("**/tasks**")
            badge = page.get_by_test_id("user-badge")
            assert "demo" in badge.inner_text()
            _shot(page, ARTIFACTS / "s1_03_logged_in.png")
            s1.steps.append(StepResult("submit_login", True, "redirect /tasks", "s1_03_logged_in.png"))
        except Exception as exc:  # noqa: BLE001 — smoke report must capture any UI failure
            s1.ok = False
            s1.failure = str(exc)
            _shot(page, ARTIFACTS / "s1_fail.png")
            s1.steps.append(StepResult("login_failed", False, str(exc), "s1_fail.png"))
        results.append(s1)

        # --- S2 Create ---
        s2 = ScenarioResult(id="S2", title="Create task", ok=True)
        try:
            if not s1.ok:
                raise RuntimeError("Skipped: login failed")
            page.get_by_test_id("task-title-input").fill(TASK_TITLE)
            _shot(page, ARTIFACTS / "s2_01_create_filled.png")
            s2.steps.append(StepResult("fill_title", True, TASK_TITLE, "s2_01_create_filled.png"))
            page.get_by_test_id("task-create").click()
            page.wait_for_selector("[data-testid='flash']")
            flash = page.get_by_test_id("flash").inner_text()
            assert "Создано" in flash
            _shot(page, ARTIFACTS / "s2_02_created.png")
            s2.steps.append(StepResult("submit_create", True, flash, "s2_02_created.png"))
        except Exception as exc:  # noqa: BLE001
            s2.ok = False
            s2.failure = str(exc)
            _shot(page, ARTIFACTS / "s2_fail.png")
            s2.steps.append(StepResult("create_failed", False, str(exc), "s2_fail.png"))
        results.append(s2)

        # --- S3 Verify ---
        s3 = ScenarioResult(id="S3", title="Verify task in list", ok=True)
        try:
            if not s2.ok:
                raise RuntimeError("Skipped: create failed")
            listing = page.get_by_test_id("task-list").inner_text()
            assert TASK_TITLE in listing
            assert "[todo]" in listing
            _shot(page, ARTIFACTS / "s3_01_verified.png")
            s3.steps.append(StepResult("assert_list", True, listing.strip(), "s3_01_verified.png"))
        except Exception as exc:  # noqa: BLE001
            s3.ok = False
            s3.failure = str(exc)
            _shot(page, ARTIFACTS / "s3_fail.png")
            s3.steps.append(StepResult("verify_failed", False, str(exc), "s3_fail.png"))
        results.append(s3)

        # --- S6 Complete (new feature) ---
        s6 = ScenarioResult(id="S6", title="Complete task", ok=True)
        try:
            if not s3.ok:
                raise RuntimeError("Skipped: verify failed")
            page.get_by_test_id("task-complete").first.click()
            page.wait_for_selector("[data-testid='flash']")
            flash = page.get_by_test_id("flash").inner_text()
            assert "Завершено" in flash
            listing = page.get_by_test_id("task-list").inner_text()
            assert "[done]" in listing
            assert page.get_by_test_id("task-complete").count() == 0
            _shot(page, ARTIFACTS / "s6_01_completed.png")
            s6.steps.append(StepResult("complete", True, flash, "s6_01_completed.png"))
        except Exception as exc:  # noqa: BLE001
            s6.ok = False
            s6.failure = str(exc)
            _shot(page, ARTIFACTS / "s6_fail.png")
            s6.steps.append(StepResult("complete_failed", False, str(exc), "s6_fail.png"))
        results.append(s6)

        # --- S4 Delete ---
        s4 = ScenarioResult(id="S4", title="Delete task", ok=True)
        try:
            if not s6.ok:
                raise RuntimeError("Skipped: complete failed")
            page.get_by_test_id("task-delete").first.click()
            page.wait_for_selector("[data-testid='flash']")
            flash = page.get_by_test_id("flash").inner_text()
            assert "Удалено" in flash
            empty = page.get_by_test_id("tasks-empty")
            assert empty.count() == 1
            _shot(page, ARTIFACTS / "s4_01_deleted.png")
            s4.steps.append(StepResult("delete", True, flash, "s4_01_deleted.png"))
        except Exception as exc:  # noqa: BLE001
            s4.ok = False
            s4.failure = str(exc)
            _shot(page, ARTIFACTS / "s4_fail.png")
            s4.steps.append(StepResult("delete_failed", False, str(exc), "s4_fail.png"))
        results.append(s4)

        # --- S5 Logout ---
        s5 = ScenarioResult(id="S5", title="Logout", ok=True)
        try:
            if not s1.ok:
                raise RuntimeError("Skipped: login failed")
            page.get_by_test_id("logout-link").click()
            page.wait_for_url("**/login**")
            _shot(page, ARTIFACTS / "s5_01_logged_out.png")
            s5.steps.append(StepResult("logout", True, "/login", "s5_01_logged_out.png"))
            page.goto(f"{BASE_URL}/tasks", wait_until="domcontentloaded")
            page.wait_for_url("**/login**")
            _shot(page, ARTIFACTS / "s5_02_protected.png")
            s5.steps.append(
                StepResult("protected_tasks", True, "redirect to login", "s5_02_protected.png")
            )
        except Exception as exc:  # noqa: BLE001
            s5.ok = False
            s5.failure = str(exc)
            _shot(page, ARTIFACTS / "s5_fail.png")
            s5.steps.append(StepResult("logout_failed", False, str(exc), "s5_fail.png"))
        results.append(s5)

        browser.close()
    return results


def write_report(results: list[ScenarioResult]) -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "generated_at": stamp,
        "base_url": BASE_URL,
        "tooling": "playwright-python (Playwright MCP unavailable in this environment)",
        "scenarios": [
            {
                **asdict(item),
                "steps": [asdict(step) for step in item.steps],
            }
            for item in results
        ],
        "all_passed": all(item.ok for item in results),
    }
    json_path = ARTIFACTS / "smoke_report.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Day 38 UI smoke report",
        "",
        f"Generated: `{stamp}`",
        f"Base URL: `{BASE_URL}`",
        "Tooling: Playwright Python (MCP Playwright / Claude in Mobile недоступны в workspace)",
        "",
        f"**Overall:** {'PASS' if payload['all_passed'] else 'FAIL'}",
        "",
    ]
    for item in results:
        mark = "PASS" if item.ok else "FAIL"
        lines.append(f"## {item.id} — {item.title}: {mark}")
        if item.failure:
            lines.append(f"- Failure: `{item.failure}`")
            lines.append(f"- Suggested area: UI handler for `{item.id}` / selectors `data-testid`")
        for step in item.steps:
            step_mark = "ok" if step.ok else "FAIL"
            lines.append(
                f"- [{step_mark}] {step.name}: {step.detail} "
                f"([screenshot]({step.screenshot}))"
            )
            lines.append(f"  ![{step.name}]({step.screenshot})")
        lines.append("")
    md_path = ARTIFACTS / "smoke_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    db_path = ARTIFACTS / "smoke_tasks.json"
    if db_path.exists():
        db_path.unlink()
    log_file = LOG_PATH.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "homeworks.todoist.webapp",
            "--host",
            "127.0.0.1",
            "--port",
            "8765",
            "--db",
            str(db_path),
        ],
        cwd=str(ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_health()
        results = run_smoke()
        report = write_report(results)
        print(report)
        print("PASS" if all(r.ok for r in results) else "FAIL")
        return 0 if all(r.ok for r in results) else 1
    finally:
        log_file.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
