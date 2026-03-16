#!/usr/bin/env bash

set -euo pipefail

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [[ $# -eq 0 ]]; then
  PROMPT="Привет! Коротко объясни, что такое LLM."
else
  PROMPT="$*"
fi

PROVIDER="${LLM_PROVIDER:-auto}"
if [[ "${PROVIDER}" == "auto" ]]; then
  if [[ -n "${LLM_API_KEY:-}" ]]; then
    PROVIDER="openrouter"
  elif [[ -n "${GROQ_API_KEY:-}" ]]; then
    PROVIDER="groq"
  elif [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
    PROVIDER="openrouter"
  else
    echo "Error: set LLM_API_KEY, GROQ_API_KEY or OPENROUTER_API_KEY." >&2
    exit 1
  fi
fi

if [[ "${PROVIDER}" == "groq" ]]; then
  API_URL="${LLM_API_URL:-https://api.groq.com/openai/v1/chat/completions}"
  MODEL="${LLM_MODEL:-llama-3.1-8b-instant}"
  FALLBACK_MODELS="${LLM_FALLBACK_MODELS:-gemma2-9b-it,llama-3.3-70b-versatile}"
  API_KEY="${LLM_API_KEY:-${GROQ_API_KEY:-}}"
elif [[ "${PROVIDER}" == "openrouter" ]]; then
  API_URL="${LLM_API_URL:-https://openrouter.ai/api/v1/chat/completions}"
  MODEL="${LLM_MODEL:-openrouter/auto}"
  FALLBACK_MODELS="${LLM_FALLBACK_MODELS:-qwen/qwen-2.5-7b-instruct:free,google/gemma-2-9b-it:free,mistralai/mistral-7b-instruct:free}"
  API_KEY="${LLM_API_KEY:-${OPENROUTER_API_KEY:-}}"
else
  echo "Error: LLM_PROVIDER must be auto, openrouter, or groq." >&2
  exit 1
fi

if [[ -z "${API_KEY}" ]]; then
  echo "Error: API key is not set for provider ${PROVIDER}." >&2
  exit 1
fi

IFS=',' read -r -a EXTRA_MODELS <<< "${FALLBACK_MODELS}"
MODELS=("${MODEL}")
for M in "${EXTRA_MODELS[@]}"; do
  M_TRIMMED="$(echo "${M}" | xargs)"
  if [[ -n "${M_TRIMMED}" && "${M_TRIMMED}" != "${MODEL}" ]]; then
    MODELS+=("${M_TRIMMED}")
  fi
done

for CURRENT_MODEL in "${MODELS[@]}"; do
  RESPONSE="$(curl -sS "${API_URL}" \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${CURRENT_MODEL}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"${PROMPT}\"}],
      \"temperature\": 0.7
    }")"

  if [[ "${RESPONSE}" == *"No endpoints found for"* || "${RESPONSE}" == *"model_not_found"* || "${RESPONSE}" == *"does not exist"* ]]; then
    continue
  fi

  if [[ "${CURRENT_MODEL}" != "${MODEL}" ]]; then
    echo "Info: primary model unavailable, used fallback: ${CURRENT_MODEL}" >&2
  fi
  echo "${RESPONSE}"
  exit 0
done

if [[ "${PROVIDER}" == "openrouter" ]]; then
  echo "Error: no available endpoints for selected OpenRouter models." >&2
  echo "Tip: switch to free Groq provider: export LLM_PROVIDER=groq && export GROQ_API_KEY=your_key" >&2
else
  echo "Error: no available endpoints for selected models." >&2
fi
exit 1
