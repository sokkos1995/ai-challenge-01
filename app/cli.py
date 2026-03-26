import os
import sys

from app.agent import AgentRequestOptions, SimpleLLMAgent, load_env_file
from app.cli_utils import parse_args, print_token_stats, print_verbose_stats, resolve_prompt


def main() -> None:
    load_env_file()
    args = parse_args()
    agent = SimpleLLMAgent.from_env()
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
            if agent.chat_history_path:
                print(
                    f"Chat history SQLite (restored on restart): {os.path.abspath(agent.chat_history_path)}"
                )
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
