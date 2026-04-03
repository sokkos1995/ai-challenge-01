import json
from typing import Optional

from app.models import AgentRequestOptions


def constraint_system_message(options: AgentRequestOptions) -> Optional[dict[str, str]]:
    constraints = []
    if options.response_format:
        constraints.append(f"Output format requirement: {options.response_format}")
    if options.max_output_tokens is not None:
        constraints.append(
            f"Length limit: keep the answer within approximately {options.max_output_tokens} output tokens."
        )
    if options.finish_instruction:
        constraints.append(f"Finish condition: {options.finish_instruction}")
    if not constraints:
        return None
    return {"role": "system", "content": "\n".join(constraints)}


def chat_session_system_message(options: AgentRequestOptions) -> dict[str, str]:
    parts = [
        "This request includes the full conversation so far with this user: every user and assistant "
        "message before the final user message is prior dialogue from the same session (restored across "
        "restarts). Use that history as ground truth. Do not claim you cannot see, remember, or access "
        "those earlier messages."
    ]
    extra = constraint_system_message(options)
    if extra:
        parts.append(extra["content"])
    return {"role": "system", "content": "\n\n".join(parts)}


def chat_summary_system_message(summary: str) -> Optional[dict[str, str]]:
    clean_summary = summary.strip()
    if not clean_summary:
        return None
    return {
        "role": "system",
        "content": (
            "Conversation memory summary for older turns (this replaces full old history):\n"
            f"{clean_summary}"
        ),
    }


def _chat_message_to_line(msg: dict[str, str]) -> str:
    role = str(msg.get("role", "user")).strip() or "user"
    content = str(msg.get("content", "")).strip().replace("\n", "\\n")
    return f"{role}: {content}"


def chat_summary_update_messages(previous_summary: str, chunk: list[dict[str, str]]) -> list[dict[str, str]]:
    chunk_text = "\n".join(_chat_message_to_line(msg) for msg in chunk)
    prior = previous_summary.strip() or "(empty)"
    prompt = (
        "Update the running conversation summary.\n"
        "Requirements:\n"
        "- Keep key user facts, constraints, preferences, and unresolved tasks.\n"
        "- Keep concrete decisions and important outputs.\n"
        "- Remove redundancy and small talk.\n"
        "- Keep it concise (5-12 bullet points max).\n"
        "- Output plain text only.\n\n"
        "Previous summary:\n"
        f"{prior}\n\n"
        "New messages to merge:\n"
        f"{chunk_text}"
    )
    return [
        {"role": "system", "content": "You are a chat memory compressor."},
        {"role": "user", "content": prompt},
    ]


def facts_system_message(facts: dict[str, str]) -> dict[str, str]:
    facts_json = json.dumps(facts, ensure_ascii=True)
    content = (
        "facts (key-value memory extracted from the user):\n"
        f"{facts_json}\n\n"
        "Use these facts as ground truth. If you need missing details, ask the user."
    )
    return {"role": "system", "content": content}


def invariants_system_message(invariants: list[str]) -> Optional[dict[str, str]]:
    clean_items = [item.strip() for item in invariants if item.strip()]
    if not clean_items:
        return None

    content = "\n".join(
        [
            "Non-negotiable invariants:",
            *[f"- {item}" for item in clean_items],
            "",
            "Treat these invariants as a source of truth separate from the dialog.",
            "Before answering, explicitly check whether the user's request conflicts with any invariant.",
            "Never propose, endorse, or elaborate a solution that would violate an invariant.",
            "If there is a conflict, refuse briefly, name the conflicting invariant, explain why, and offer a compliant alternative.",
        ]
    )
    return {"role": "system", "content": content}


def personalization_system_message(user_id: str, profile: dict[str, str]) -> dict[str, str]:
    profile_json = json.dumps(profile, ensure_ascii=True)
    content = (
        f"Personalization profile for user_id={user_id}:\n"
        f"{profile_json}\n\n"
        "Automatically adapt answer style, level of detail, output format, and constraints to this user. "
        "If the current user message explicitly asks for another style or format, follow the current message."
    )
    return {"role": "system", "content": content}


def merge_system_messages(*messages: Optional[dict[str, str]]) -> Optional[dict[str, str]]:
    parts = []
    for message in messages:
        if not message:
            continue
        content = str(message.get("content", "")).strip()
        if content:
            parts.append(content)
    if not parts:
        return None
    return {"role": "system", "content": "\n\n".join(parts)}
