"""Token cost estimation for gateway audit."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenPrices:
    """USD per 1M tokens (OpenRouter-style ballpark for a cheap chat model)."""

    prompt_per_mtok: float = 0.10
    completion_per_mtok: float = 0.40


def prices_from_env() -> TokenPrices:
    prompt = float(os.getenv("GATEWAY_PRICE_PROMPT_PER_MTOK", "0.10"))
    completion = float(os.getenv("GATEWAY_PRICE_COMPLETION_PER_MTOK", "0.40"))
    return TokenPrices(prompt_per_mtok=prompt, completion_per_mtok=completion)


def estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    prices: TokenPrices | None = None,
) -> float:
    p = prices or prices_from_env()
    cost = (
        prompt_tokens * p.prompt_per_mtok + completion_tokens * p.completion_per_mtok
    ) / 1_000_000.0
    return round(cost, 8)


def rough_token_count(text: str) -> int:
    """Heuristic ~4 chars/token when provider usage is missing."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)
