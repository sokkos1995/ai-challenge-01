from __future__ import annotations

from pathlib import Path

from app.config import build_ssl_context, get_provider_config, load_env_file
from app.models import AgentRequestOptions
from app.response_parser import parse_agent_response
from app.services.provider_service import ProviderService

from .context import LocalJsonContextProvider
from .prompting import build_support_prompt
from .rag import SimpleFaqRetriever


def answer_question(question: str, user_id: str, ticket_id: str) -> str:
    base_dir = Path(__file__).resolve().parent.parent
    retriever = SimpleFaqRetriever(base_dir / "knowledge_base" / "faq.md")
    context_provider = LocalJsonContextProvider(base_dir / "data")

    chunks = retriever.retrieve(question)
    user = context_provider.get_user(user_id)
    ticket = context_provider.get_ticket(ticket_id)
    prompt = build_support_prompt(question, chunks, user, ticket)

    provider, api_url, api_key, model_candidates = get_provider_config()
    service = ProviderService(
        provider=provider,
        api_url=api_url,
        api_key=api_key,
        model_candidates=model_candidates,
        ssl_context=build_ssl_context(),
    )
    options = AgentRequestOptions(
        temperature=0.2,
        top_p=None,
        top_k=None,
        response_format=None,
        max_output_tokens=500,
        stop_sequences=[],
        finish_instruction=None,
        count_tokens=False,
    )
    messages = [
        {
            "role": "system",
            "content": "You are a support assistant. Answer in user's language.",
        },
        {"role": "user", "content": prompt},
    ]
    raw_data, model, latency_sec = service.complete(messages, options)
    parsed = parse_agent_response(
        data=raw_data,
        tried_model=model,
        response_elapsed_sec=latency_sec,
        provider=provider,
    )
    return parsed.answer


def main() -> None:
    load_env_file()
    print("Support assistant demo")
    question = input("Вопрос [Почему не работает авторизация?]: ").strip() or "Почему не работает авторизация?"
    user_id = input("User ID [u_001]: ").strip() or "u_001"
    ticket_id = input("Ticket ID [t_1001]: ").strip() or "t_1001"
    print(answer_question(question=question, user_id=user_id, ticket_id=ticket_id))


if __name__ == "__main__":
    main()
