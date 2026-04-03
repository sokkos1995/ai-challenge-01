import json
import os
import sys

from app.agent import AgentRequestOptions, SimpleLLMAgent, load_env_file
from app.cli_utils import parse_args, print_token_stats, print_verbose_stats, resolve_prompt


_TASK_COMMAND_USAGE = (
    "agent> Usage: @task show | pause | resume | plan+ <text> | done+ <text> | "
    "expected <text> | state <PLANNING|EXECUTION|VALIDATION|DONE>"
)
_INVARIANT_COMMAND_USAGE = "agent> Usage: @invariant show | add <text> | clear"


def _ask_interview_value(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    return input(f"{prompt}{suffix}: ").strip() or default


def _run_personalization_interview(agent: SimpleLLMAgent) -> None:
    snapshot = agent.personalization_snapshot()
    profile = snapshot.get("profile", {})
    print("agent> New user detected. Let's set up personalization.")
    answers = {
        "role": _ask_interview_value("Role / who are you", str(profile.get("role", ""))),
        "stack": _ask_interview_value("Main stack", str(profile.get("stack", ""))),
        "answer_detail": _ask_interview_value(
            "Answer style: brief or detailed",
            str(profile.get("answer_detail", "brief")),
        ),
        "answer_format": _ask_interview_value(
            "Preferred format (text, bullets, step-by-step, JSON, ...)",
            str(profile.get("answer_format", "bullets")),
        ),
        "constraints": _ask_interview_value(
            "Constraints (optional)",
            str(profile.get("constraints", "")),
        ),
    }
    agent.save_user_profile_interview(answers)
    print("agent> personalization saved.")


def _ensure_user_personalization(agent: SimpleLLMAgent) -> None:
    if not agent.user_id:
        return
    created = agent.ensure_user_profile()
    if created:
        print(f"User profile created for user_id={agent.user_id}")
    if agent.user_profile_needs_interview():
        _run_personalization_interview(agent)


def _handle_personalization_command(user_input: str, agent: SimpleLLMAgent) -> bool:
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
        _run_personalization_interview(agent)
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


def _task_command_argument(payload: str, prefix: str, usage: str) -> str:
    value = payload[len(prefix) :].strip()
    if not value:
        raise ValueError(usage)
    return value


def _handle_task_command(user_input: str, agent: SimpleLLMAgent) -> bool:
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

    update_commands = {
        "plan+ ": ("plan+", "agent> Usage: @task plan+ <text>", "agent> task plan updated."),
        "done+ ": ("done+", "agent> Usage: @task done+ <text>", "agent> task done updated."),
        "expected ": (
            "expected_action",
            "agent> Usage: @task expected <text>",
            "agent> task expected action updated.",
        ),
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
            "agent> Usage: @task state <PLANNING|EXECUTION|VALIDATION|DONE>",
        )
        applied_state = agent.transition_task_state(next_state)
        print(f"agent> task state updated: {applied_state}")
        return True

    print(_TASK_COMMAND_USAGE)
    return True


def _handle_invariant_command(user_input: str, agent: SimpleLLMAgent) -> bool:
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


def main() -> None:
    load_env_file()
    args = parse_args()
    agent = SimpleLLMAgent.from_env(user_id=getattr(args, "user_id", None))
    if getattr(args, "context_strategy", None):
        if bool(getattr(args, "summary", False)):
            print("Warning: --summary is ignored when --context-strategy is set.", file=sys.stderr)
        agent.set_context_strategy(str(args.context_strategy))
    else:
        agent.set_context_strategy("summary" if bool(getattr(args, "summary", False)) else "full")
    tokens_enabled = bool(getattr(args, "tokens", False))
    last_token_stats = None
    last_token_provider = None
    last_token_model = None

    options = AgentRequestOptions(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        response_format=args.response_format,
        max_output_tokens=args.max_output_tokens,
        stop_sequences=args.stop_sequences,
        finish_instruction=args.finish_instruction,
        count_tokens=tokens_enabled,
    )

    try:
        _ensure_user_personalization(agent)
        if args.chat:
            print("Interactive mode started. Type your message and press Enter. Type 'exit' to quit.")
            print(f"Context strategy: {agent.context_strategy}")
            if agent.user_id:
                print(f"User ID: {agent.user_id}")
            if agent.chat_history_path and agent.context_strategy in {"full", "summary"}:
                print(f"Chat history SQLite (restored on restart): {os.path.abspath(agent.chat_history_path)}")
            while True:
                user_input = input("you> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in {"exit", "quit", "q"}:
                    print("Bye!")
                    break

                if user_input.lower().startswith("@tokens"):
                    cmd = user_input.lower()
                    if "off" in cmd:
                        tokens_enabled = False
                    else:
                        tokens_enabled = True
                    options.count_tokens = tokens_enabled
                    if last_token_stats is not None:
                        print_token_stats(
                            last_token_stats,
                            last_token_provider or agent.provider,
                            last_token_model or agent.model_candidates[0],
                        )
                    continue

                if _handle_personalization_command(user_input, agent):
                    continue

                try:
                    if _handle_invariant_command(user_input, agent):
                        continue
                except Exception as exc:
                    print(f"agent> invariant command error: {exc}")
                    continue

                try:
                    if _handle_task_command(user_input, agent):
                        continue
                except Exception as exc:
                    print(f"agent> task command error: {exc}")
                    continue

                if user_input.lower() == "@summary":
                    if not agent.chat_summary_enabled:
                        print("agent> Summary mode is OFF. Run with --summary to enable.")
                    else:
                        summary = agent.get_chat_summary().strip()
                        if summary:
                            print(f"agent> Summary:\n{summary}")
                        else:
                            print("agent> Summary is empty yet.")
                    continue

                if user_input.lower().startswith("@mem"):
                    try:
                        if user_input.lower() == "@mem show":
                            snapshot = agent.memory_snapshot()
                            print(f"agent> memory:\n{json.dumps(snapshot, ensure_ascii=False, indent=2)}")
                            continue
                        if user_input.lower().startswith("@mem clear "):
                            layer = user_input[len("@mem clear ") :].strip()
                            agent.clear_memory_layer(layer)
                            print(f"agent> memory layer cleared: {layer}")
                            continue
                        if user_input.lower().startswith("@mem short note "):
                            note = user_input[len("@mem short note ") :].strip()
                            agent.add_short_term_note(note)
                            print("agent> short-term note saved.")
                            continue
                        if user_input.lower().startswith("@mem work "):
                            payload = user_input[len("@mem work ") :].strip()
                            if "=" not in payload:
                                print("agent> Usage: @mem work <field>=<value>")
                                continue
                            field_name, field_value = payload.split("=", 1)
                            agent.update_working_task_field(field_name.strip(), field_value.strip())
                            print(f"agent> working memory updated: {field_name.strip()}")
                            continue
                        if user_input.lower().startswith("@mem long decision "):
                            decision = user_input[len("@mem long decision ") :].strip()
                            agent.add_long_term_decision(decision)
                            print("agent> long-term decision saved.")
                            continue
                        if user_input.lower().startswith("@mem long "):
                            payload = user_input[len("@mem long ") :].strip()
                            parts = payload.split(" ", 1)
                            if len(parts) != 2 or "=" not in parts[1]:
                                print("agent> Usage: @mem long <profile|knowledge> <key>=<value>")
                                continue
                            bucket = parts[0].strip().lower()
                            key, value = parts[1].split("=", 1)
                            agent.update_long_term_memory(bucket, key.strip(), value.strip())
                            print(f"agent> long-term {bucket} updated: {key.strip()}")
                            continue
                        print("agent> Unknown @mem command. Use: show, clear, short note, work, long.")
                        continue
                    except Exception as exc:
                        print(f"agent> memory command error: {exc}")
                        continue

                if agent.context_strategy == "branching":
                    cmd = user_input.lower()
                    if cmd == "@branches" or cmd == "@branch-info":
                        print(f"agent> {agent.get_branch_info()}")
                        continue
                    if cmd == "@checkpoint":
                        agent.branch_checkpoint()
                        print("agent> checkpoint saved.")
                        continue
                    if cmd == "@fork":
                        agent.branch_fork()
                        print("agent> fork created (branch 1 active).")
                        continue
                    if cmd.startswith("@switch"):
                        parts = user_input.split()
                        branch_id = parts[1] if len(parts) >= 2 else ""
                        if not branch_id:
                            print("agent> Usage: @switch 1  (or  @switch 2).")
                            continue
                        agent.branch_switch(branch_id)
                        print(f"agent> switched to branch {branch_id}.")
                        continue

                response = agent.ask_chat(user_input, options)
                print(f"agent> {response.answer}")
                if args.verbose:
                    print_verbose_stats(response.raw_data, response.provider, response.model, response.latency_sec)
                if options.count_tokens and getattr(response, "token_stats", None):
                    last_token_stats = response.token_stats
                    last_token_provider = response.provider
                    last_token_model = response.model
                    print_token_stats(last_token_stats, response.provider, response.model)
            return

        prompt = resolve_prompt(args)
        response = agent.ask(prompt, options)
        print(response.answer)
        if args.verbose:
            print_verbose_stats(response.raw_data, response.provider, response.model, response.latency_sec)
        if options.count_tokens and getattr(response, "token_stats", None):
            print_token_stats(response.token_stats, response.provider, response.model)
            last_token_stats = response.token_stats
            last_token_provider = response.provider
            last_token_model = response.model
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except EOFError:
        print("\nInput stream closed.")
        sys.exit(1)
    except Exception as exc:
        print(str(exc))
        sys.exit(1)
