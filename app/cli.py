import os
import select
import sys

from app.agent import AgentRequestOptions, SimpleLLMAgent, load_env_file
from app.cli_utils import print_token_stats, print_verbose_stats, resolve_prompt, parse_args
from app.rag_output import format_rag_answer
from app.services.chat_command_service import (
    handle_branching_command,
    handle_invariant_command,
    handle_memory_command,
    handle_personalization_command,
    handle_rag_command,
    handle_summary_command,
    handle_task_command,
    handle_tokens_command,
)
from app.services.rag_chat_service import RagChatService
from app.services.rag_service import RagAnswer, RagService
from app.services.todoist_chat_service import TodoistChatService
from app.services.todoist_reminder_service import TodoistReminderService

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


def _read_chat_input(reminder_service: TodoistReminderService | None) -> str:
    if reminder_service is None:
        return input("you> ").strip()

    sys.stdout.write("you> ")
    sys.stdout.flush()
    while True:
        for message in reminder_service.drain_messages():
            sys.stdout.write(f"\n{message}\n")
            sys.stdout.write("you> ")
            sys.stdout.flush()

        ready, _, _ = select.select([sys.stdin], [], [], 1.0)
        if not ready:
            continue

        line = sys.stdin.readline()
        if line == "":
            raise EOFError
        return line.strip()


def _rag_answer_for_prompt(
    rag_service: RagService,
    rag_chat_service: RagChatService | None,
    prompt: str,
    agent: SimpleLLMAgent,
    options: AgentRequestOptions,
    chat_mode: bool,
) -> RagAnswer:
    if chat_mode and rag_chat_service is not None:
        return rag_chat_service.ask(
            prompt,
            ask_llm=lambda rag_prompt: agent.ask_chat(rag_prompt, options),
        )
    rag_prompt, contexts, low_relevance = rag_service.build_prompt(prompt)
    if low_relevance:
        return rag_service.low_relevance_answer(contexts)
    response = agent.ask_chat(rag_prompt, options) if chat_mode else agent.ask(rag_prompt, options)
    return rag_service.parse_answer(response.answer, contexts)


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
    reminder_service: TodoistReminderService | None = None
    todoist_chat_service = TodoistChatService()
    rag_service: RagService | None = None
    rag_chat_service: RagChatService | None = None
    rag_enabled = bool(getattr(args, "rag", False))

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
        if rag_enabled:
            rag_service = RagService.from_json_index(
                index_path=str(args.rag_index_path),
                top_k_before=int(args.rag_top_k_before),
                top_k_after=int(args.rag_top_k_after),
                similarity_threshold=float(args.rag_similarity_threshold),
                min_context_score=float(args.rag_min_context_score),
            )
            rag_chat_service = RagChatService(rag_service)
        _ensure_user_personalization(agent)
        if args.chat:
            print("Interactive mode started. Type your message and press Enter. Type 'exit' to quit.")
            print(f"Context strategy: {agent.context_strategy}")
            print(f"RAG mode: {'ON' if rag_enabled else 'OFF'}")
            if agent.user_id:
                print(f"User ID: {agent.user_id}")
            if agent.chat_history_path and agent.context_strategy in {"full", "summary"}:
                print(f"Chat history SQLite (restored on restart): {os.path.abspath(agent.chat_history_path)}")
            reminder_service = TodoistReminderService.from_env()
            if reminder_service is not None:
                reminder_service.start()
            while True:
                user_input = _read_chat_input(reminder_service)
                if not user_input:
                    continue
                if user_input.lower() in {"exit", "quit", "q"}:
                    print("Bye!")
                    break

                handled, tokens_enabled = handle_tokens_command(
                    user_input=user_input,
                    options=options,
                    tokens_enabled=tokens_enabled,
                    last_token_stats=last_token_stats,
                    last_token_provider=last_token_provider,
                    last_token_model=last_token_model,
                    agent=agent,
                )
                if handled:
                    continue

                handled, rag_enabled = handle_rag_command(user_input, rag_enabled, rag_chat_service)
                if handled:
                    continue

                if handle_personalization_command(user_input, agent, _run_personalization_interview):
                    continue

                try:
                    if handle_invariant_command(user_input, agent):
                        continue
                except Exception as exc:
                    print(f"agent> invariant command error: {exc}")
                    continue

                try:
                    if handle_task_command(user_input, agent):
                        continue
                except Exception as exc:
                    print(f"agent> task command error: {exc}")
                    continue

                try:
                    todoist_reply = todoist_chat_service.maybe_handle(user_input)
                    if todoist_reply is not None:
                        print(todoist_reply)
                        continue
                except Exception as exc:
                    print(f"agent> Todoist command error: {exc}")
                    continue

                if handle_summary_command(user_input, agent):
                    continue

                try:
                    if handle_memory_command(user_input, agent):
                        continue
                except Exception as exc:
                    print(f"agent> memory command error: {exc}")
                    continue

                if handle_branching_command(user_input, agent):
                    continue

                if rag_enabled:
                    if rag_service is None:
                        print("agent> RAG service is not initialized.")
                        continue
                    rag_answer = _rag_answer_for_prompt(
                        rag_service,
                        rag_chat_service,
                        user_input,
                        agent,
                        options,
                        chat_mode=True,
                    )
                    print(format_rag_answer(rag_answer))
                else:
                    response = agent.ask_chat(user_input, options)
                    print(f"agent> {response.answer}")
                if args.verbose:
                    if not rag_enabled:
                        print_verbose_stats(response.raw_data, response.provider, response.model, response.latency_sec)
                if not rag_enabled and options.count_tokens and getattr(response, "token_stats", None):
                    last_token_stats = response.token_stats
                    last_token_provider = response.provider
                    last_token_model = response.model
                    print_token_stats(last_token_stats, response.provider, response.model)
            return

        prompt = resolve_prompt(args)
        if rag_enabled:
            if rag_service is None:
                raise RuntimeError("RAG service is not initialized.")
            rag_answer = _rag_answer_for_prompt(
                rag_service,
                rag_chat_service,
                prompt,
                agent,
                options,
                chat_mode=False,
            )
            print(format_rag_answer(rag_answer, prefix=""))
        else:
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
    finally:
        if reminder_service is not None:
            reminder_service.stop()
