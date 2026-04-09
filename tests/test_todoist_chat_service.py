from app.services.todoist_chat_service import (
    TodoistChatService,
    parse_todoist_create_intent,
)


def test_parses_russian_todoist_create_phrase() -> None:
    intent = parse_todoist_create_intent("сделай задачу в тудуисте сделать домашку сегодня в 12 17")

    assert intent is not None
    assert intent.content == "сделать домашку"
    assert intent.due_string == "today at 12:17"


def test_parses_explicit_create_task_with_due_inside_quotes() -> None:
    intent = parse_todoist_create_intent('create_task создай задачу "сделай дз в 12:18 сегодня"')

    assert intent is not None
    assert intent.content == "сделай дз"
    assert intent.due_string == "today at 12:18"


class _TodoistClientStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def create_task(self, content: str, due_string: str = "", project_id: str = "") -> dict:
        self.calls.append((content, due_string, project_id))
        return {
            "content": content,
            "due": {"string": due_string},
        }


def test_chat_service_returns_real_creation_result() -> None:
    client = _TodoistClientStub()
    service = TodoistChatService(client=client)

    reply = service.maybe_handle("create_task создай задачу сделать домашку сегодня в 12 17")

    assert reply == 'agent> Todoist: задача "сделать домашку" создана, срок: today at 12:17.'
    assert client.calls == [("сделать домашку", "today at 12:17", "")]
