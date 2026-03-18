#!/usr/bin/env bash

set -euo pipefail

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

TEMPERATURE="${LLM_TEMPERATURE:-0.7}"
TOP_P="${LLM_TOP_P:-}"
TOP_K="${LLM_TOP_K:-}"
RESPONSE_FORMAT="${LLM_RESPONSE_FORMAT:-}"
MAX_OUTPUT_TOKENS="${LLM_MAX_OUTPUT_TOKENS:-}"
FINISH_INSTRUCTION="${LLM_FINISH_INSTRUCTION:-}"
VERBOSE=0
STOP_SEQUENCES=()
if [[ -n "${LLM_STOP_SEQUENCES:-}" ]]; then
  IFS=',' read -r -a STOP_SEQUENCES <<< "${LLM_STOP_SEQUENCES}"
fi
PROMPT_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --temperature)
      [[ $# -ge 2 ]] || { echo "Error: --temperature requires value." >&2; exit 1; }
      TEMPERATURE="$2"
      shift 2
      ;;
    -v|--verbose)
      VERBOSE=1
      shift
      ;;
    --top-p)
      [[ $# -ge 2 ]] || { echo "Error: --top-p requires value." >&2; exit 1; }
      TOP_P="$2"
      shift 2
      ;;
    --top-k)
      [[ $# -ge 2 ]] || { echo "Error: --top-k requires value." >&2; exit 1; }
      TOP_K="$2"
      shift 2
      ;;
    --response-format)
      [[ $# -ge 2 ]] || { echo "Error: --response-format requires value." >&2; exit 1; }
      RESPONSE_FORMAT="$2"
      shift 2
      ;;
    --max-output-tokens)
      [[ $# -ge 2 ]] || { echo "Error: --max-output-tokens requires value." >&2; exit 1; }
      MAX_OUTPUT_TOKENS="$2"
      shift 2
      ;;
    --stop-sequence)
      [[ $# -ge 2 ]] || { echo "Error: --stop-sequence requires value." >&2; exit 1; }
      STOP_SEQUENCES+=("$2")
      shift 2
      ;;
    --finish-instruction)
      [[ $# -ge 2 ]] || { echo "Error: --finish-instruction requires value." >&2; exit 1; }
      FINISH_INSTRUCTION="$2"
      shift 2
      ;;
    --)
      shift
      PROMPT_ARGS+=("$@")
      break
      ;;
    *)
      PROMPT_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#PROMPT_ARGS[@]} -eq 0 ]]; then
  PROMPT="Привет! Коротко объясни, что такое LLM."
else
  PROMPT="${PROMPT_ARGS[*]}"
fi

is_number() {
  [[ "$1" =~ ^-?[0-9]+([.][0-9]+)?$ ]]
}

if ! is_number "${TEMPERATURE}" || ! awk "BEGIN { exit !(${TEMPERATURE} >= 0 && ${TEMPERATURE} <= 2) }"; then
  echo "Error: --temperature must be number in range [0,2]." >&2
  exit 1
fi
if [[ -n "${TOP_P}" ]] && { ! is_number "${TOP_P}" || ! awk "BEGIN { exit !(${TOP_P} >= 0 && ${TOP_P} <= 1) }"; }; then
  echo "Error: --top-p must be number in range [0,1]." >&2
  exit 1
fi
if [[ -n "${TOP_K}" ]] && [[ ! "${TOP_K}" =~ ^[0-9]+$ ]]; then
  echo "Error: --top-k must be integer >= 1." >&2
  exit 1
fi
if [[ -n "${TOP_K}" ]] && [[ "${TOP_K}" -lt 1 ]]; then
  echo "Error: --top-k must be integer >= 1." >&2
  exit 1
fi
if [[ -n "${MAX_OUTPUT_TOKENS}" ]] && [[ ! "${MAX_OUTPUT_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "Error: --max-output-tokens must be integer >= 1." >&2
  exit 1
fi
if [[ -n "${MAX_OUTPUT_TOKENS}" ]] && [[ "${MAX_OUTPUT_TOKENS}" -lt 1 ]]; then
  echo "Error: --max-output-tokens must be integer >= 1." >&2
  exit 1
fi

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"
}

ESCAPED_PROMPT="$(json_escape "${PROMPT}")"
CONSTRAINT_LINES=()
if [[ -n "${RESPONSE_FORMAT}" ]]; then
  CONSTRAINT_LINES+=("Output format requirement: ${RESPONSE_FORMAT}")
fi
if [[ -n "${MAX_OUTPUT_TOKENS}" ]]; then
  CONSTRAINT_LINES+=("Length limit: keep the answer within approximately ${MAX_OUTPUT_TOKENS} output tokens.")
fi
if [[ -n "${FINISH_INSTRUCTION}" ]]; then
  CONSTRAINT_LINES+=("Finish condition: ${FINISH_INSTRUCTION}")
fi

if [[ ${#CONSTRAINT_LINES[@]} -gt 0 ]]; then
  SYSTEM_PROMPT="$(printf '%s\n' "${CONSTRAINT_LINES[@]}")"
  SYSTEM_PROMPT="${SYSTEM_PROMPT%$'\n'}"
  ESCAPED_SYSTEM_PROMPT="$(json_escape "${SYSTEM_PROMPT}")"
  MESSAGES_JSON="[{\"role\":\"system\",\"content\":${ESCAPED_SYSTEM_PROMPT}},{\"role\":\"user\",\"content\":${ESCAPED_PROMPT}}]"
else
  MESSAGES_JSON="[{\"role\":\"user\",\"content\":${ESCAPED_PROMPT}}]"
fi

SAMPLING_FIELDS="\"temperature\": ${TEMPERATURE}"
if [[ -n "${TOP_P}" ]]; then
  SAMPLING_FIELDS+=", \"top_p\": ${TOP_P}"
fi
if [[ -n "${TOP_K}" ]]; then
  SAMPLING_FIELDS+=", \"top_k\": ${TOP_K}"
fi
if [[ -n "${MAX_OUTPUT_TOKENS}" ]]; then
  SAMPLING_FIELDS+=", \"max_tokens\": ${MAX_OUTPUT_TOKENS}"
fi
if [[ ${#STOP_SEQUENCES[@]} -gt 0 ]]; then
  STOP_ITEMS=()
  for STOP_SEQ in "${STOP_SEQUENCES[@]}"; do
    if [[ -n "${STOP_SEQ}" ]]; then
      STOP_ITEMS+=("$(json_escape "${STOP_SEQ}")")
    fi
  done
  if [[ ${#STOP_ITEMS[@]} -gt 0 ]]; then
    STOP_JSON="["
    for i in "${!STOP_ITEMS[@]}"; do
      if [[ "${i}" -gt 0 ]]; then
        STOP_JSON+=","
      fi
      STOP_JSON+="${STOP_ITEMS[$i]}"
    done
    STOP_JSON+="]"
    SAMPLING_FIELDS+=", \"stop\": ${STOP_JSON}"
  fi
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
  REQUEST_START_MS="$(python3 -c 'import time; print(int(time.time() * 1000))')"
  RESPONSE="$(curl -sS "${API_URL}" \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${CURRENT_MODEL}\",
      \"messages\": ${MESSAGES_JSON},
      ${SAMPLING_FIELDS}
    }")"

  if [[ "${RESPONSE}" == *"No endpoints found for"* || "${RESPONSE}" == *"model_not_found"* || "${RESPONSE}" == *"does not exist"* ]]; then
    continue
  fi

  if [[ "${CURRENT_MODEL}" != "${MODEL}" ]]; then
    echo "Info: primary model unavailable, used fallback: ${CURRENT_MODEL}" >&2
  fi
  REQUEST_END_MS="$(python3 -c 'import time; print(int(time.time() * 1000))')"
  ELAPSED_MS=$((REQUEST_END_MS - REQUEST_START_MS))
  if ! printf '%s' "${RESPONSE}" | python3 - "${VERBOSE}" "${PROVIDER}" "${CURRENT_MODEL}" "${ELAPSED_MS}" <<'PY'
import json
import sys

verbose = sys.argv[1] == "1"
provider = sys.argv[2]
model = sys.argv[3]
elapsed_ms = sys.argv[4]

try:
    data = json.load(sys.stdin)
except Exception as exc:
    print(f"Error: failed to parse provider response as JSON: {exc}", file=sys.stderr)
    sys.exit(1)

choice0 = (data.get("choices") or [{}])[0]
message = choice0.get("message") or {}
content = message.get("content")

if isinstance(content, str) and content.strip():
    print(content)
    sys.stdout.flush()
else:
    usage = data.get("usage", {})
    completion_details = usage.get("completion_tokens_details", {})
    reasoning_tokens = completion_details.get("reasoning_tokens")
    finish_reason = choice0.get("finish_reason")
    print(
        "Received empty assistant content from provider (message.content is null/empty).\n"
        f"finish_reason={finish_reason}, reasoning_tokens={reasoning_tokens}.\n"
        "Try increasing --max-output-tokens, relaxing output constraints, or changing model/provider.",
        file=sys.stderr,
    )
    sys.exit(2)

if verbose:
    usage = data.get("usage", {})
    finish_reason = choice0.get("finish_reason")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    cost = usage.get("cost")

    line = (
        f"[stats] provider={provider} model={model} latency_ms={elapsed_ms} "
        f"finish_reason={finish_reason} prompt_tokens={prompt_tokens} "
        f"completion_tokens={completion_tokens} total_tokens={total_tokens}"
    )
    if cost is not None:
        line += f" cost={cost}"
    print(line, file=sys.stderr)
PY
  then
    exit 1
  fi
  exit 0
done

if [[ "${PROVIDER}" == "openrouter" ]]; then
  echo "Error: no available endpoints for selected OpenRouter models." >&2
  echo "Tip: switch to free Groq provider: export LLM_PROVIDER=groq && export GROQ_API_KEY=your_key" >&2
else
  echo "Error: no available endpoints for selected models." >&2
fi
exit 1
