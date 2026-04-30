from __future__ import annotations

from .context import TicketContext, UserContext


def build_support_prompt(
    question: str,
    retrieved_chunks: list[str],
    user: UserContext | None,
    ticket: TicketContext | None,
) -> str:
    kb_block = "\n".join(f"- {chunk}" for chunk in retrieved_chunks) or "- no kb data"
    user_block = (
        f"user_id={user.user_id}, name={user.name}, plan={user.plan}, locale={user.locale}"
        if user
        else "unknown user"
    )
    ticket_block = (
        f"ticket_id={ticket.ticket_id}, status={ticket.status}, topic={ticket.topic}, "
        f"last_error={ticket.last_error}"
        if ticket
        else "unknown ticket"
    )
    return (
        "You are a support assistant. Answer briefly and concretely.\n"
        "Use only provided context. If context is missing, say what is missing.\n\n"
        f"Question: {question}\n"
        f"User context: {user_block}\n"
        f"Ticket context: {ticket_block}\n"
        "Knowledge chunks:\n"
        f"{kb_block}\n"
    )
