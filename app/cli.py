import os
import sys

from app.agent import AgentRequestOptions, SimpleLLMAgent, load_env_file
from app.cli_utils import parse_args, print_token_stats, print_verbose_stats, resolve_prompt


def main() -> None:
    load_env_file()
    args = parse_args()
    agent = SimpleLLMAgent.from_env()
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
        if args.chat:
            print("Interactive mode started. Type your message and press Enter. Type 'exit' to quit.")
            print(f"Context strategy: {agent.context_strategy}")
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
