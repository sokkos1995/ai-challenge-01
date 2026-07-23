"""Unit tests for PersonalizationService (offline, tmp users.db)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.personalization_service import PersonalizationService


def _answers() -> dict[str, str]:
    return {
        "role": "backend engineer",
        "stack": "Python",
        "answer_detail": "short",
        "answer_format": "bullets",
    }


def test_without_user_id_skips_interview(tmp_path: Path) -> None:
    service = PersonalizationService(users_base_path=str(tmp_path), user_id=None)
    assert service.has_user() is False
    assert service.needs_interview() is False
    assert service.snapshot() == {"user_id": None, "interview_completed": False, "profile": {}}
    assert service.system_message() is None
    with pytest.raises(RuntimeError, match="--user-id"):
        service.save_interview_answers(_answers())


def test_interview_flow_and_system_message(tmp_path: Path) -> None:
    service = PersonalizationService(users_base_path=str(tmp_path), user_id="u42")
    assert service.ensure_user_exists() is True
    assert service.needs_interview() is True

    service.save_interview_answers(_answers())
    assert service.needs_interview() is False

    snap = service.snapshot()
    assert snap["user_id"] == "u42"
    assert snap["interview_completed"] is True
    assert snap["profile"]["role"] == "backend engineer"

    msg = service.system_message()
    assert msg is not None
    assert msg["role"] == "system"
    assert "u42" in msg["content"] or "backend" in msg["content"].lower()


def test_partial_profile_still_needs_interview(tmp_path: Path) -> None:
    service = PersonalizationService(users_base_path=str(tmp_path), user_id="partial")
    service.ensure_user_exists()
    service.update_profile_entries({"role": "QA", "stack": "pytest"})
    assert service.needs_interview() is True

    service.update_profile_entries(
        {"answer_detail": "detailed", "answer_format": "paragraph"}
    )
    assert service.needs_interview() is False
