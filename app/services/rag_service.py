from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RagContext:
    chunk_id: str
    source: str
    section: str
    score: float
    text: str
    quote: str


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: list[dict[str, str]]
    quotes: list[str]
    policy: str
    contexts: list[RagContext]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-zА-Яа-я0-9_]+", text.lower())


def _vectorize(text: str, dim: int = 128) -> list[float]:
    vec = [0.0] * dim
    for token in _tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weight = 1.0 + (digest[5] / 255.0)
        vec[idx] += sign * weight
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vector dimensions do not match.")
    return sum(x * y for x, y in zip(a, b))


def _rewrite_query(question: str) -> str:
    replacements = {
        "оркестрация": "orchestration маршрутизация tool registry",
        "индекс": "index embedding retrieval chunk",
        "провайдер": "provider model fallback api key",
        "тесты": "tests pytest assertions",
    }
    rewritten = question
    lowered = question.lower()
    for key, expansion in replacements.items():
        if key in lowered and expansion not in lowered:
            rewritten = f"{rewritten}. {expansion}"
    return rewritten


def _extract_quote(text: str, query: str, max_chars: int = 220) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    query_tokens = [token for token in _tokenize(query) if len(token) >= 4]
    lowered = cleaned.lower()
    hit_index = -1
    for token in query_tokens:
        idx = lowered.find(token)
        if idx >= 0:
            hit_index = idx
            break
    if hit_index < 0:
        return cleaned[:max_chars].strip()
    start = max(0, hit_index - max_chars // 3)
    return cleaned[start : start + max_chars].strip()


def _heuristic_rerank_score(question: str, record: dict[str, Any], dense_score: float) -> float:
    text = str(record.get("text", ""))
    metadata = record.get("metadata", {})
    source = str(metadata.get("source", "")) if isinstance(metadata, dict) else ""
    section = str(metadata.get("section", "")) if isinstance(metadata, dict) else ""

    q_tokens = set(_tokenize(question))
    d_tokens = set(_tokenize(f"{text} {source} {section}"))
    overlap = 0.0
    if q_tokens and d_tokens:
        overlap = len(q_tokens & d_tokens) / len(q_tokens)

    source_bonus = 0.08 if source.endswith(".py") or source.endswith(".md") else 0.0
    return dense_score + 0.45 * overlap + source_bonus


def _retrieve_contexts(
    records: list[dict[str, Any]],
    question: str,
    top_k_before: int,
    top_k_after: int,
    similarity_threshold: float,
) -> list[RagContext]:
    if not records:
        return []
    first_embedding = records[0].get("embedding")
    dim = len(first_embedding) if isinstance(first_embedding, list) and first_embedding else 128
    query_vector = _vectorize(_rewrite_query(question), dim=dim)

    scored: list[tuple[float, dict[str, Any]]] = []
    for record in records:
        embedding = record.get("embedding")
        if not isinstance(embedding, list):
            continue
        dense_score = _cosine_similarity(query_vector, [float(x) for x in embedding])
        rerank_score = _heuristic_rerank_score(question, record, dense_score)
        if rerank_score < similarity_threshold:
            continue
        scored.append((rerank_score, record))
    scored.sort(key=lambda item: item[0], reverse=True)

    selected: list[RagContext] = []
    for score, record in scored[: max(1, min(top_k_before, top_k_after))]:
        metadata = record.get("metadata", {})
        source = str(metadata.get("source", "unknown")) if isinstance(metadata, dict) else "unknown"
        section = str(metadata.get("section", "unknown")) if isinstance(metadata, dict) else "unknown"
        text = str(record.get("text", ""))
        selected.append(
            RagContext(
                chunk_id=str(record.get("chunk_id", "")),
                source=source,
                section=section,
                score=round(score, 4),
                text=text,
                quote=_extract_quote(text, question),
            )
        )
    return selected


def _build_grounded_prompt(question: str, contexts: list[RagContext]) -> str:
    context_lines: list[str] = []
    for idx, item in enumerate(contexts, start=1):
        context_lines.extend(
            [
                f"[Context {idx}]",
                f"source: {item.source}",
                f"section: {item.section}",
                f"chunk_id: {item.chunk_id}",
                f"score: {item.score}",
                f"text: {item.text}",
                "",
            ]
        )
    return (
        "Ты отвечаешь только по контексту ниже.\n"
        "Верни строго JSON с ключами answer, sources, quotes.\n"
        "sources: массив объектов {source, section, chunk_id}.\n"
        "quotes: массив коротких цитат из контекста.\n"
        "Если данных недостаточно, в answer напиши: "
        '"Не знаю по текущему контексту. Уточните вопрос.".\n\n'
        f"{chr(10).join(context_lines).strip()}\n\n"
        f"Вопрос: {question}"
    )


def _fallback_payload(contexts: list[RagContext], answer: str) -> dict[str, Any]:
    return {
        "answer": answer,
        "sources": [
            {"source": ctx.source, "section": ctx.section, "chunk_id": ctx.chunk_id} for ctx in contexts
        ],
        "quotes": [ctx.quote for ctx in contexts if ctx.quote],
    }


def _extract_first_json_object(raw_answer: str) -> str | None:
    text = raw_answer.strip()
    if not text:
        return None

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1).strip()

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1].strip()
    return None


def _load_grounded_payload(raw_answer: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw_answer)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass

    extracted = _extract_first_json_object(raw_answer)
    if not extracted:
        return None
    try:
        payload = json.loads(extracted)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_grounded_json(raw_answer: str, contexts: list[RagContext]) -> dict[str, Any]:
    fallback = _fallback_payload(contexts, "Не удалось распарсить ответ модели.")
    payload = _load_grounded_payload(raw_answer)
    if payload is None:
        return fallback
    answer = str(payload.get("answer", "")).strip() or fallback["answer"]

    sources: list[dict[str, str]] = []
    raw_sources = payload.get("sources", [])
    if isinstance(raw_sources, list):
        for item in raw_sources:
            if not isinstance(item, dict):
                continue
            sources.append(
                {
                    "source": str(item.get("source", "")),
                    "section": str(item.get("section", "")),
                    "chunk_id": str(item.get("chunk_id", "")),
                }
            )
    quotes = payload.get("quotes", [])
    parsed_quotes = [str(item).strip() for item in quotes] if isinstance(quotes, list) else []
    parsed_quotes = [item for item in parsed_quotes if item]

    if not sources:
        sources = fallback["sources"]
    if not parsed_quotes:
        parsed_quotes = fallback["quotes"]
    return {"answer": answer, "sources": sources, "quotes": parsed_quotes}


class RagService:
    def __init__(
        self,
        records: list[dict[str, Any]],
        top_k_before: int = 8,
        top_k_after: int = 4,
        similarity_threshold: float = 0.2,
        min_context_score: float = 0.24,
    ) -> None:
        self._records = records
        self._top_k_before = max(1, top_k_before)
        self._top_k_after = max(1, top_k_after)
        self._similarity_threshold = similarity_threshold
        self._min_context_score = min_context_score

    @classmethod
    def from_json_index(
        cls,
        index_path: str,
        top_k_before: int = 8,
        top_k_after: int = 4,
        similarity_threshold: float = 0.2,
        min_context_score: float = 0.24,
    ) -> "RagService":
        path = Path(index_path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        if not isinstance(records, list):
            raise RuntimeError("Index payload does not contain a valid records list.")
        valid_records = [item for item in records if isinstance(item, dict)]
        return cls(
            records=valid_records,
            top_k_before=top_k_before,
            top_k_after=top_k_after,
            similarity_threshold=similarity_threshold,
            min_context_score=min_context_score,
        )

    def is_low_relevance(self, contexts: list[RagContext]) -> bool:
        return not contexts or max(ctx.score for ctx in contexts) < self._min_context_score

    def build_prompt(self, question: str) -> tuple[str, list[RagContext], bool]:
        contexts = _retrieve_contexts(
            records=self._records,
            question=question,
            top_k_before=self._top_k_before,
            top_k_after=self._top_k_after,
            similarity_threshold=self._similarity_threshold,
        )
        return _build_grounded_prompt(question, contexts), contexts, self.is_low_relevance(contexts)

    def low_relevance_answer(self, contexts: list[RagContext]) -> RagAnswer:
        payload = _fallback_payload(contexts, "Не знаю по текущему контексту. Уточните вопрос.")
        return RagAnswer(
            answer=str(payload["answer"]),
            sources=payload["sources"],
            quotes=payload["quotes"],
            policy="low_relevance_fallback",
            contexts=contexts,
        )

    def parse_answer(self, raw_answer: str, contexts: list[RagContext]) -> RagAnswer:
        payload = _parse_grounded_json(raw_answer, contexts)
        return RagAnswer(
            answer=str(payload["answer"]),
            sources=payload["sources"],
            quotes=payload["quotes"],
            policy="grounded_answer",
            contexts=contexts,
        )
