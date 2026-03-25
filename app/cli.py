import os
import sys

from app.agent import AgentRequestOptions, SimpleLLMAgent, load_env_file
from app.cli_utils import parse_args, print_verbose_stats, resolve_prompt


def main() -> None:
    load_env_file()
    args = parse_args()
    agent = SimpleLLMAgent.from_env()
    options = AgentRequestOptions(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        response_format=args.response_format,
        max_output_tokens=args.max_output_tokens,
        stop_sequences=args.stop_sequences,
        finish_instruction=args.finish_instruction,
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
                response = agent.ask_chat(user_input, options)
                print(f"agent> {response.answer}")
                if args.verbose:
                    print_verbose_stats(response.raw_data, response.provider, response.model, response.latency_sec)
            return

        prompt = resolve_prompt(args)
        response = agent.ask(prompt, options)
        print(response.answer)
        if args.verbose:
            print_verbose_stats(response.raw_data, response.provider, response.model, response.latency_sec)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except EOFError:
        print("\nInput stream closed.")
        sys.exit(1)
    except Exception as exc:
        print(str(exc))
        sys.exit(1)
