from types import SimpleNamespace

from app.models import AgentRequestOptions
from app.services.chat_command_service import (
    handle_branching_command,
    handle_rag_command,
    handle_tokens_command,
)


class _AgentStub:
    provider = "stub-provider"
    model_candidates = ["stub-model"]


class _BranchingAgentStub:
    def __init__(self) -> None:
        self.context_strategy = "branching"
        self.calls: list[tuple[str, str]] = []

    def get_branch_info(self) -> str:
        return "branch-info"

    def branch_checkpoint(self) -> None:
        self.calls.append(("checkpoint", ""))

    def branch_fork(self) -> None:
        self.calls.append(("fork", ""))

    def branch_switch(self, branch_id: str) -> None:
        self.calls.append(("switch", branch_id))


def test_handle_tokens_command_toggles_state() -> None:
    options = AgentRequestOptions(
        temperature=0.2,
        top_p=None,
        top_k=None,
        response_format=None,
        max_output_tokens=None,
        stop_sequences=[],
        finish_instruction=None,
        count_tokens=False,
    )
    handled, enabled = handle_tokens_command(
        user_input="@tokens",
        options=options,
        tokens_enabled=False,
        last_token_stats=None,
        last_token_provider=None,
        last_token_model=None,
        agent=_AgentStub(),  # type: ignore[arg-type]
    )
    assert handled
    assert enabled
    assert options.count_tokens

    handled, enabled = handle_tokens_command(
        user_input="@tokens off",
        options=options,
        tokens_enabled=enabled,
        last_token_stats=None,
        last_token_provider=None,
        last_token_model=None,
        agent=_AgentStub(),  # type: ignore[arg-type]
    )
    assert handled
    assert not enabled
    assert not options.count_tokens


def test_handle_rag_command_switches_mode() -> None:
    handled, enabled = handle_rag_command("@rag", False, None)
    assert handled
    assert enabled

    handled, enabled = handle_rag_command("@rag off", enabled, None)
    assert handled
    assert not enabled

    fake_rag_chat = SimpleNamespace(memory_snapshot=lambda: {"goal": "x"})
    handled, enabled = handle_rag_command("@rag memory", enabled, fake_rag_chat)  # type: ignore[arg-type]
    assert handled


def test_handle_branching_command_dispatches_actions() -> None:
    agent = _BranchingAgentStub()
    assert handle_branching_command("@branches", agent)  # type: ignore[arg-type]
    assert handle_branching_command("@checkpoint", agent)  # type: ignore[arg-type]
    assert handle_branching_command("@fork", agent)  # type: ignore[arg-type]
    assert handle_branching_command("@switch 2", agent)  # type: ignore[arg-type]
    assert ("checkpoint", "") in agent.calls
    assert ("fork", "") in agent.calls
    assert ("switch", "2") in agent.calls
