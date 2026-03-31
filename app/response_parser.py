from app.models import AgentResponse


def parse_agent_response(
    data: dict, tried_model: str, response_elapsed_sec: float, provider: str
) -> AgentResponse:
    choice0 = data.get("choices", [{}])[0]
    message = choice0.get("message", {})
    answer = message.get("content")
    if isinstance(answer, str) and answer.strip():
        return AgentResponse(
            answer=answer,
            raw_data=data,
            model=tried_model,
            latency_sec=response_elapsed_sec,
            provider=provider,
        )

    finish_reason = choice0.get("finish_reason")
    usage = data.get("usage", {})
    completion_details = usage.get("completion_tokens_details", {})
    reasoning_tokens = completion_details.get("reasoning_tokens")
    raise RuntimeError(
        "Received empty assistant content from provider (message.content is null/empty).\n"
        f"finish_reason={finish_reason}, reasoning_tokens={reasoning_tokens}.\n"
        "This often happens when strict limits are used and the model spends output tokens on reasoning.\n"
        "Try one of:\n"
        "1) increase --max-output-tokens (or LLM_MAX_OUTPUT_TOKENS),\n"
        "2) remove/relax stop sequences and finish instruction,\n"
        "3) try another model/provider."
    )

