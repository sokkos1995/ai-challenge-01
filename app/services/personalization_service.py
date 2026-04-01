import os
from typing import Optional

from app.messages import personalization_system_message
from app.storage import (
    ensure_user_record,
    load_user_profile,
    set_user_interview_completed,
    upsert_user_profile_entries,
)


class PersonalizationService:
    REQUIRED_PROFILE_KEYS = ("role", "stack", "answer_detail", "answer_format")

    def __init__(self, users_base_path: str, user_id: Optional[str] = None) -> None:
        self._user_db_path = os.path.join(users_base_path, "users.db")
        self._user_id = user_id.strip() if user_id and user_id.strip() else None
        self._profile: dict[str, str] = {}
        self._interview_completed = False
        self._loaded = False

    @property
    def user_id(self) -> Optional[str]:
        return self._user_id

    def has_user(self) -> bool:
        return self._user_id is not None

    def _require_user_id(self) -> str:
        if not self._user_id:
            raise RuntimeError("Personalization requires --user-id.")
        return self._user_id

    def ensure_user_exists(self) -> bool:
        if not self._user_id:
            return False
        created = ensure_user_record(self._user_db_path, self._user_id)
        self._loaded = False
        return created

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

    def update_profile_entries(self, entries: dict[str, str]) -> None:
        user_id = self._require_user_id()
        self.ensure_user_exists()
        upsert_user_profile_entries(self._user_db_path, user_id, entries)
        self._refresh_completion_flag()

    def snapshot(self) -> dict:
        if not self._user_id:
            return {"user_id": None, "interview_completed": False, "profile": {}}
        self._ensure_loaded()
        return {
            "user_id": self._user_id,
            "interview_completed": self._interview_completed,
            "profile": dict(self._profile),
        }

    def system_message(self) -> Optional[dict[str, str]]:
        if not self._user_id:
            return None
        self._ensure_loaded()
        if not self._profile:
            return None
        return personalization_system_message(self._user_id, self._profile)
