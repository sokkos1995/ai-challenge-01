from app.mcp_servers import todoist_server


def test_list_tasks_unwraps_paginated_results(monkeypatch) -> None:
    monkeypatch.setattr(todoist_server, "_todoist_token", lambda: "token")
    monkeypatch.setattr(
        todoist_server,
        "_todoist_request",
        lambda **_: {
            "results": [
                {"id": "1", "content": "a", "project_id": "p1"},
                {"id": "2", "content": "b", "project_id": "p2"},
            ]
        },
    )

    result = todoist_server.list_tasks(project_id="p2", limit=20)

    assert result == {
        "count": 1,
        "tasks": [{"id": "2", "content": "b", "project_id": "p2"}],
    }


def test_list_tasks_follows_next_cursor_until_limit(monkeypatch) -> None:
    monkeypatch.setattr(todoist_server, "_todoist_token", lambda: "token")
    responses = [
        {
            "results": [{"id": "1", "content": "a", "project_id": "p1"}],
            "next_cursor": "cursor-1",
        },
        {
            "results": [{"id": "2", "content": "b", "project_id": "p1"}],
            "next_cursor": None,
        },
    ]
    calls: list[dict] = []

    def _fake_request(**kwargs):
        calls.append(kwargs)
        return responses[len(calls) - 1]

    monkeypatch.setattr(todoist_server, "_todoist_request", _fake_request)

    result = todoist_server.list_tasks(project_id="", limit=2)

    assert result == {
        "count": 2,
        "tasks": [
            {"id": "1", "content": "a", "project_id": "p1"},
            {"id": "2", "content": "b", "project_id": "p1"},
        ],
    }
    assert calls[0]["query"]["limit"] == "2"
    assert calls[1]["query"]["cursor"] == "cursor-1"


def test_list_tasks_limit_zero_returns_all_pages(monkeypatch) -> None:
    monkeypatch.setattr(todoist_server, "_todoist_token", lambda: "token")
    responses = [
        {"results": [{"id": "1"}], "next_cursor": "cursor-1"},
        {"results": [{"id": "2"}], "next_cursor": ""},
    ]

    monkeypatch.setattr(
        todoist_server,
        "_todoist_request",
        lambda **kwargs: responses.pop(0),
    )

    result = todoist_server.list_tasks(limit=0)

    assert result == {
        "count": 2,
        "tasks": [{"id": "1"}, {"id": "2"}],
    }


def test_list_tasks_passes_filter_query_to_api(monkeypatch) -> None:
    monkeypatch.setattr(todoist_server, "_todoist_token", lambda: "token")
    calls: list[dict] = []

    def _fake_request(**kwargs):
        calls.append(kwargs)
        return {"results": [], "next_cursor": ""}

    monkeypatch.setattr(todoist_server, "_todoist_request", _fake_request)

    todoist_server.list_tasks(limit=0, filter_query="today | overdue")

    assert calls[0]["query"]["filter"] == "today | overdue"
