import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    default_stop_sequences = []
    stop_sequences_env = os.getenv("LLM_STOP_SEQUENCES")
    if stop_sequences_env:
        default_stop_sequences = [s.strip() for s in stop_sequences_env.split(",") if s.strip()]

    parser = argparse.ArgumentParser(description="Send prompt to LLM API and print answer.")
    parser.add_argument("prompt", nargs="*", help="Prompt text (ignored if --prompt-file is set)")
    prompt_file_default = os.getenv("LLM_PROMPT_FILE")
    if prompt_file_default is not None:
        prompt_file_default = prompt_file_default.strip() or None
    parser.add_argument(
        "-f",
        "--prompt-file",
        dest="prompt_file",
        metavar="PATH",
        default=prompt_file_default,
        help="Read prompt from file (UTF-8). Overrides positional prompt. Default: LLM_PROMPT_FILE.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        help="Sampling temperature (0..2). Default: 0.7",
    )
    parser.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=float(os.getenv("LLM_TOP_P")) if os.getenv("LLM_TOP_P") else None,
        help="Nucleus sampling top_p (0..1). Optional.",
    )
    parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=int(os.getenv("LLM_TOP_K")) if os.getenv("LLM_TOP_K") else None,
        help="Top-k sampling (>=1). Optional.",
    )
    parser.add_argument(
        "--response-format",
        dest="response_format",
        default=os.getenv("LLM_RESPONSE_FORMAT"),
        help="Explicit output format requirements (e.g. JSON schema or bullet structure).",
    )
    parser.add_argument(
        "--max-output-tokens",
        dest="max_output_tokens",
        type=int,
        default=int(os.getenv("LLM_MAX_OUTPUT_TOKENS")) if os.getenv("LLM_MAX_OUTPUT_TOKENS") else None,
        help="Maximum response length in tokens (>=1). Optional.",
    )
    parser.add_argument(
        "--stop-sequence",
        dest="stop_sequences",
        action="append",
        default=default_stop_sequences.copy(),
        help="Stop generation when this sequence is produced. Can be passed multiple times.",
    )
    parser.add_argument(
        "--finish-instruction",
        dest="finish_instruction",
        default=os.getenv("LLM_FINISH_INSTRUCTION"),
        help="Explicit condition/instruction for when and how to finish the answer.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print minimal response stats (latency, tokens, model, finish_reason).",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Run interactive chat mode (type 'exit' to quit).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Enable chat summary mode (use summary + last N messages instead of full history).",
    )
    parser.add_argument(
        "--context-strategy",
        dest="context_strategy",
        choices=["full", "summary", "sliding", "facts", "branching", "memory"],
        default=None,
        help=(
            "Context strategy for chat mode (--chat). If not set: uses 'summary' when --summary is provided, else 'full'. "
            "Choices: full, summary, sliding, facts, branching, memory."
        ),
    )
    parser.add_argument(
        "--tokens",
        action="store_true",
        help=(
            "Print token breakdown: current request tokens, dialog history tokens, and model response tokens. "
            "For chat mode this may add extra API calls to estimate tokens for history/current parts."
        ),
    )
    args = parser.parse_args()

    if not 0 <= args.temperature <= 2:
        parser.error("--temperature must be in range [0, 2]")
    if args.top_p is not None and not 0 <= args.top_p <= 1:
        parser.error("--top-p must be in range [0, 1]")
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if args.max_output_tokens is not None and args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be >= 1")
    return args


def resolve_prompt(args: argparse.Namespace) -> str:
    path = (args.prompt_file or "").strip()
    if path:
        if not os.path.isfile(path):
            print(f"Error: prompt file not found: {path}", file=sys.stderr)
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as file_handle:
            text = file_handle.read()
        text = text.strip()
        if not text:
            print("Error: prompt file is empty.", file=sys.stderr)
            sys.exit(1)
        return text
    inline = " ".join(args.prompt).strip()
    return inline or "Привет! Скажи коротко, что такое LLM?"


def print_verbose_stats(data: dict, provider: str, model: str, elapsed_sec: float) -> None:
    usage = data.get("usage", {})
    choice0 = data.get("choices", [{}])[0]
    finish_reason = choice0.get("finish_reason")

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    cost = usage.get("cost")

    stats_lines = [
        f"[stats] provider={provider}",
        f"model={model}",
        f"latency_ms={elapsed_sec * 1000:.0f}",
        f"finish_reason={finish_reason}",
        f"prompt_tokens={prompt_tokens}",
        f"completion_tokens={completion_tokens}",
        f"total_tokens={total_tokens}",
    ]
    if cost is not None:
        stats_lines.append(f"cost={cost}")

    print("\n".join(stats_lines), file=sys.stderr)


def print_token_stats(token_stats: dict, provider: str, model: str) -> None:
    """
    token_stats expected keys:
      - current_request_tokens
      - dialog_history_tokens
      - response_model_tokens
      - prompt_tokens_total
      - completion_tokens
      - total_tokens
    """

    stats_lines = [
        f"[tokens] provider={provider}",
        f"model={model}",
        f"current_request_tokens={token_stats.get('current_request_tokens')}",
        f"dialog_history_tokens={token_stats.get('dialog_history_tokens')}",
        f"response_model_tokens={token_stats.get('response_model_tokens')}",
        f"prompt_tokens_total={token_stats.get('prompt_tokens_total')}",
        f"completion_tokens={token_stats.get('completion_tokens')}",
        f"total_tokens={token_stats.get('total_tokens')}",
    ]
    print("\n".join(stats_lines), file=sys.stderr)
