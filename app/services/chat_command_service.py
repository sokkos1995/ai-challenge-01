from __future__ import annotations

import json

from app.agent import AgentRequestOptions, SimpleLLMAgent
from app.cli_utils import print_token_stats
from app.services.rag_chat_service import RagChatService

_TASK_COMMAND_USAGE = (
    "agent> Usage: @task show | pause | resume | reject | approve-plan | reject-plan | "
    "validate <pass|fail> | plan+ <text> | done+ <text> | expected <text> | "
    "state <PLANNING|EXECUTION|VALIDATION|DONE|REJECTED>"
)
_INVARIANT_COMMAND_USAGE = "agent> Usage: @invariant show | add <text> | clear"


def _task_command_argument(payload: str, prefix: str, usage: str) -> str:
    value = payload[len(prefix) :].strip()
    if not value:
        raise ValueError(usage)
    return value


def handle_tokens_command(
    user_input: str,
    options: AgentRequestOptions,
    tokens_enabled: bool,
    last_token_stats: dict | None,
    last_token_provider: str | None,
    last_token_model: str | None,
    agent: SimpleLLMAgent,
) -> tuple[bool, bool]:
    if not user_input.lower().startswith("@tokens"):
        return False, tokens_enabled
    cmd = user_input.lower()
    tokens_enabled = "off" not in cmd
    options.count_tokens = tokens_enabled
    if last_token_stats is not None:
        print_token_stats(
            last_token_stats,
            last_token_provider or agent.provider,
            last_token_model or agent.model_candidates[0],
        )
    return True, tokens_enabled


def handle_rag_command(
    user_input: str,
    rag_enabled: bool,
    rag_chat_service: RagChatService | None,
) -> tuple[bool, bool]:
    if user_input.lower() == "@rag":
        rag_enabled = not rag_enabled
        print(f"agent> RAG mode: {'ON' if rag_enabled else 'OFF'}")
        return True, rag_enabled
    if user_input.lower() in {"@rag on", "@rag off", "@rag status"}:
        cmd = user_input.lower().split()[-1]
        if cmd == "on":
            rag_enabled = True
        elif cmd == "off":
            rag_enabled = False
        print(f"agent> RAG mode: {'ON' if rag_enabled else 'OFF'}")
        return True, rag_enabled
    if user_input.lower() == "@rag memory":
        if rag_chat_service is None:
            print("agent> RAG mini-chat memory is unavailable.")
        else:
            print("agent> rag memory:\n" + json.dumps(rag_chat_service.memory_snapshot(), ensure_ascii=False, indent=2))
        return True, rag_enabled
    return False, rag_enabled


def handle_summary_command(user_input: str, agent: SimpleLLMAgent) -> bool:
    if user_input.lower() != "@summary":
        return False
    if not agent.chat_summary_enabled:
        print("agent> Summary mode is OFF. Run with --summary to enable.")
    else:
        summary = agent.get_chat_summary().strip()
        if summary:
            print(f"agent> Summary:\n{summary}")
        else:
            print("agent> Summary is empty yet.")
    return True


def handle_memory_command(user_input: str, agent: SimpleLLMAgent) -> bool:
    if not user_input.lower().startswith("@mem"):
        return False
    if user_input.lower() == "@mem show":
        snapshot = agent.memory_snapshot()
        print(f"agent> memory:\n{json.dumps(snapshot, ensure_ascii=False, indent=2)}")
        return True
    if user_input.lower().startswith("@mem clear "):
        layer = user_input[len("@mem clear ") :].strip()
        agent.clear_memory_layer(layer)
        print(f"agent> memory layer cleared: {layer}")
        return True
    if user_input.lower().startswith("@mem short note "):
        note = user_input[len("@mem short note ") :].strip()
        agent.add_short_term_note(note)
        print("agent> short-term note saved.")
        return True
    if user_input.lower().startswith("@mem work "):
        payload = user_input[len("@mem work ") :].strip()
        if "=" not in payload:
            print("agent> Usage: @mem work <field>=<value>")
            return True
        field_name, field_value = payload.split("=", 1)
        agent.update_working_task_field(field_name.strip(), field_value.strip())
        print(f"agent> working memory updated: {field_name.strip()}")
        return True
    if user_input.lower().startswith("@mem long decision "):
        decision = user_input[len("@mem long decision ") :].strip()
        agent.add_long_term_decision(decision)
        print("agent> long-term decision saved.")
        return True
    if user_input.lower().startswith("@mem long "):
        payload = user_input[len("@mem long ") :].strip()
        parts = payload.split(" ", 1)
        if len(parts) != 2 or "=" not in parts[1]:
            print("agent> Usage: @mem long <profile|knowledge> <key>=<value>")
            return True
        bucket = parts[0].strip().lower()
        key, value = parts[1].split("=", 1)
        agent.update_long_term_memory(bucket, key.strip(), value.strip())
        print(f"agent> long-term {bucket} updated: {key.strip()}")
        return True
    print("agent> Unknown @mem command. Use: show, clear, short note, work, long.")
    return True


def handle_personalization_command(user_input: str, agent: SimpleLLMAgent, interview_runner) -> bool:
    if not user_input.lower().startswith("@personalization"):
        return False
    if not agent.user_id:
        print("agent> Personalization is available only with --user-id.")
        return True

    payload = user_input[len("@personalization") :].strip()
    if not payload or payload.lower() == "show":
        snapshot = agent.personalization_snapshot()
        print(f"agent> personalization:\n{json.dumps(snapshot, ensure_ascii=False, indent=2)}")
        return True
    if payload.lower() == "interview":
        interview_runner(agent)
        return True
    if "=" not in payload:
        print("agent> Usage: @personalization show | interview | <key>=<value>")
        return True

    key, value = payload.split("=", 1)
    clean_key = key.strip()
    if not clean_key:
        print("agent> Personalization key must not be empty.")
        return True
    agent.update_personalization(clean_key, value.strip())
    print(f"agent> personalization updated: {clean_key}")
    return True


def handle_task_command(user_input: str, agent: SimpleLLMAgent) -> bool:
    if not user_input.lower().startswith("@task"):
        return False

    payload = user_input[len("@task") :].strip()
    if not payload or payload.lower() == "show":
        snapshot = agent.memory_snapshot().get("working", {})
        print(f"agent> task:\n{json.dumps(snapshot, ensure_ascii=False, indent=2)}")
        return True

    lower_payload = payload.lower()
    if lower_payload == "pause":
        agent.pause_current_task()
        print("agent> task paused.")
        return True
    if lower_payload == "resume":
        agent.resume_current_task()
        print("agent> task resumed.")
        return True
    if lower_payload == "reject":
        applied_state = agent.transition_task_state("REJECTED")
        print(f"agent> task state updated: {applied_state}")
        return True
    if lower_payload == "approve-plan":
        agent.update_working_task_field("plan_status", "APPROVED")
        print("agent> task plan approved.")
        return True
    if lower_payload == "reject-plan":
        agent.update_working_task_field("plan_status", "DRAFT")
        print("agent> task plan moved back to draft.")
        return True
    if lower_payload == "validate pass":
        agent.update_working_task_field("validation_status", "PASSED")
        print("agent> task validation marked as passed.")
        return True
    if lower_payload == "validate fail":
        agent.update_working_task_field("validation_status", "FAILED")
        print("agent> task validation marked as failed.")
        return True

    update_commands = {
        "plan+ ": ("plan+", "agent> Usage: @task plan+ <text>", "agent> task plan updated."),
        "done+ ": ("done+", "agent> Usage: @task done+ <text>", "agent> task done updated."),
        "expected ": ("expected_action", "agent> Usage: @task expected <text>", "agent> task expected action updated."),
    }
    for prefix, (field_name, usage, success_message) in update_commands.items():
        if lower_payload.startswith(prefix):
            field_value = _task_command_argument(payload, prefix, usage)
            agent.update_working_task_field(field_name, field_value)
            print(success_message)
            return True

    if lower_payload.startswith("state "):
        next_state = _task_command_argument(
            payload,
            "state ",
            "agent> Usage: @task state <PLANNING|EXECUTION|VALIDATION|DONE|REJECTED>",
        )
        applied_state = agent.transition_task_state(next_state)
        print(f"agent> task state updated: {applied_state}")
        return True

    print(_TASK_COMMAND_USAGE)
    return True


def handle_invariant_command(user_input: str, agent: SimpleLLMAgent) -> bool:
    if not user_input.lower().startswith("@invariant"):
        return False

    payload = user_input[len("@invariant") :].strip()
    if not payload or payload.lower() == "show":
        snapshot = agent.memory_snapshot()
        invariants = snapshot.get("long_term", {}).get("invariants", [])
        print(f"agent> invariants:\n{json.dumps(invariants, ensure_ascii=False, indent=2)}")
        return True

    lower_payload = payload.lower()
    if lower_payload == "clear":
        agent.clear_invariants()
        print("agent> invariants cleared.")
        return True
    if lower_payload.startswith("add "):
        invariant = payload[4:].strip()
        if not invariant:
            print(_INVARIANT_COMMAND_USAGE)
            return True
        agent.add_invariant(invariant)
        print("agent> invariant saved.")
        return True

    print(_INVARIANT_COMMAND_USAGE)
    return True


def handle_branching_command(user_input: str, agent: SimpleLLMAgent) -> bool:
    if agent.context_strategy != "branching":
        return False
    cmd = user_input.lower()
    if cmd in {"@branches", "@branch-info"}:
        print(f"agent> {agent.get_branch_info()}")
        return True
    if cmd == "@checkpoint":
        agent.branch_checkpoint()
        print("agent> checkpoint saved.")
        return True
    if cmd == "@fork":
        agent.branch_fork()
        print("agent> fork created (branch 1 active).")
        return True
    if cmd.startswith("@switch"):
        parts = user_input.split()
        branch_id = parts[1] if len(parts) >= 2 else ""
        if not branch_id:
            print("agent> Usage: @switch 1  (or  @switch 2).")
            return True
        agent.branch_switch(branch_id)
        print(f"agent> switched to branch {branch_id}.")
        return True
    return False
