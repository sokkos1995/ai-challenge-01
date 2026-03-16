# Минимальный запрос к LLM через API (без OpenAI)

Этот проект показывает минимальный пример, который:
- отправляет запрос в LLM через API,
- получает ответ,
- выводит ответ в консоль (CLI).

## Файл проекта

- `llm_cli.py` — скрипт для отправки запроса и вывода ответа.
- `llm_request.sh` — bash/curl скрипт для отправки запроса.

## Подготовка

Нужен Python 3.8+ и API-ключ от любого провайдера:
- `OPENROUTER_API_KEY` (OpenRouter),
- или `GROQ_API_KEY` (Groq, бесплатный tier).

```bash
export OPENROUTER_API_KEY="ваш_api_ключ"
```

или:

```bash
export LLM_PROVIDER="groq"
export GROQ_API_KEY="ваш_api_ключ"
```

Где взять ключ:
- зарегистрироваться на [OpenRouter](https://openrouter.ai/),
- открыть [Keys](https://openrouter.ai/keys),
- создать API key.

## Запуск

```bash
python3 llm_cli.py "Объясни, что такое LLM в 2 предложениях"
```

Если аргумент не передан, скрипт использует промпт по умолчанию.

По умолчанию:
- если есть `GROQ_API_KEY`, используется Groq (`llama-3.1-8b-instant`);
- иначе используется OpenRouter (`openrouter/auto`).

Если модель недоступна, скрипт автоматически пробует фолбэки:
- `qwen/qwen-2.5-7b-instruct:free`
- `google/gemma-2-9b-it:free`
- `mistralai/mistral-7b-instruct:free`

Можно переопределить модель и URL:

```bash
export LLM_MODEL="mistralai/mistral-7b-instruct:free"
export LLM_API_URL="https://openrouter.ai/api/v1/chat/completions"
python3 llm_cli.py "Напиши краткий план обучения ML"
```

Можно задать свой список фолбэков:

```bash
export LLM_FALLBACK_MODELS="google/gemma-2-9b-it:free,qwen/qwen-2.5-7b-instruct:free"
```

## Запуск через bash + curl

```bash
chmod +x llm_request.sh
./llm_request.sh "Привет! Ответь в 1 предложении."
```

Если нужен другой endpoint/model:

```bash
export LLM_API_URL="https://openrouter.ai/api/v1/chat/completions"
export LLM_MODEL="mistralai/mistral-7b-instruct:free"
./llm_request.sh "Дай 3 идеи pet-проекта"
```

## Разбор ошибок

- `SSL: CERTIFICATE_VERIFY_FAILED`  
  На системе не хватает корректного набора CA-сертификатов.  
  Решение:
  1. `pip install certifi`
  2. `export SSL_CERT_FILE=$(python3 -c 'import certifi; print(certifi.where())')`

- `HTTP 403: error code 1010`  
  Это блокировка со стороны Cloudflare/OpenRouter (часто регион/политика).  
  Решение:
  - попробовать другой IP/VPN,
  - или использовать другой совместимый endpoint через `LLM_API_URL`.

- `HTTP 404: No endpoints found for ...`  
  У модели сейчас нет доступных endpoint.  
  Решение:
  - скрипт уже делает авто-фолбэк,
  - или переключиться на Groq:
    - `export LLM_PROVIDER=groq`
    - `export GROQ_API_KEY=...`

## Пример кода

```python
import json
import os
import urllib.request

api_key = os.getenv("OPENROUTER_API_KEY")
payload = {
    "model": "meta-llama/llama-3.1-8b-instruct:free",
    "messages": [{"role": "user", "content": "Привет! Что такое LLM?"}]
}

req = urllib.request.Request(
    "https://openrouter.ai/api/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    method="POST",
)

with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    print(data["choices"][0]["message"]["content"])
```

Пример запроса
```bash
/Users/konstantinsokolov/dev/projects/pet_projects/ai_challenge/hw/hw01/llm_request.sh | jq
{
  "id": "gen-1773673608-Vej6Hs4Ut05zcymGFZXJ",
  "object": "chat.completion",
  "created": 1773673608,
  "model": "openai/gpt-5-nano-2025-08-07",
  "provider": "OpenAI",
  "system_fingerprint": null,
  "choices": [
    {
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop",
      "native_finish_reason": "completed",
      "message": {
        "role": "assistant",
        "content": "LLM — большая языковая модель. Нейросеть, обученная на огромном объёме текста, чтобы предсказывать следующий токен и на его основе генерировать связанный текст.\n\nКак работает: используется архитектура Transformer; обучение идёт на больших данных (самообучение), затем модель адаптируют под конкретные задачи (ответы, перевод, резюме и т.д.).\n\nЧто умеет и какие есть нюансы: может писать тексты, отвечать на вопросы, переводить, суммировать. Но может ошибаться и повторять предвзятости данных.",
        "refusal": null,
        "reasoning": "**Summarizing LLM characteristics**\n\nI’m crafting a concise explanation since the user asked for a short summary. I should highlight key points about LLMs:\n\n- LLMs are large language models, AI trained on vast text data to predict the next token and generate text.\n- They operate on the Transformer architecture and can perform tasks like answering questions, writing, translating, and summarizing.\n- Advantages include versatility, while limitations involve potential errors, fact hallucinations, and biases from training data. \n\nThis structure seems clear and concise for the user’s request!",
        "reasoning_details": [
          {
            "type": "reasoning.summary",
            "summary": "**Summarizing LLM characteristics**\n\nI’m crafting a concise explanation since the user asked for a short summary. I should highlight key points about LLMs:\n\n- LLMs are large language models, AI trained on vast text data to predict the next token and generate text.\n- They operate on the Transformer architecture and can perform tasks like answering questions, writing, translating, and summarizing.\n- Advantages include versatility, while limitations involve potential errors, fact hallucinations, and biases from training data. \n\nThis structure seems clear and concise for the user’s request!",
            "format": "openai-responses-v1",
            "index": 0
          },
          {
            "type": "reasoning.encrypted",
            "data": "gAAAAABpuByUqemnz0G4wfNDPdTyqL2sf51HqRUk7BxPJmGcem1H0LSAwth_O46R36zQ9G8uxv7n2RAWoiSL7Iy8OGe2Tt3Kszp4vHlwTOaqGRUczk7pXjMknPjTNeBjBlwmxH2U6uvlTjmG36TbZ-KITVGwG6gscxPGkvUGiiknkgnevLnj55akcXhh6d98RUGJyCYyZN3lwwJLRfd5RdfrqqXMkb8qoG5Fm8YA93Ms4Cwn06IBpwR2z7CmCbgP-Ab5ciRDrU5hvsrq4rLtOLl-5RFz-nXVqPbhmj2s5TuuH2w0t1iovyGtLtpqJljaaa7DZ_AlB9LPNXScnkE7BkS77XMkQ0cD52t72xBXwxB2q_BBFyOuB-GHAf9J6RTqcNAAf85_Pvce_4fq0X10r2JBysSz6a3zxp-cZxhTFqMYVGWXIbKAngBzy27yiF9-tLe0RuwE9EZRFaZJwUYwyol73wdUs9tdIpHpfItRLz0_XCYCDtQshYperHP2cVHQTKK5EH4_6MGUg7tr2AK2C2JS9AvKue9V4V-dLQ86lixwu7071wYX6n-X0H9ws0Aclj7Tlw9cgvLpnxXibkK_Dp-P-HJbAaujlQTACexDc3mboLKLIynx3t2k9_z1EtjSOyMEY4JT2LUuUz5wqd76QV67rZtw_vS0jf__nlnwFqlUrb9D_pWrxJY4S27VQRqyT0qnW-m9of3yBrVXL_CuMI0YlgXB-9-h2O2ItOYQGwQ4flglpEk4OdLx6BRvR0_qcF5fJ2nZ_NyKEHCP2tePysvwWFXMCWJfW24fMarljBFL15qP1Os6AX7H6mQnQzBmaG15dXmLvVYVwlQCoR4Nbj_uVHU6T0khzM_aWef8Wav-q9yg5Psl6hZsPP46R0uMigV-GEAeOFxk5Zd5Wg4yjqacwirdC3Ib5Upz2szeW1gnqacB1KMUqk0h6S4O5vHXJcMKxwXsWHkTBkRy-mlY0lAEcfKDHQOVxXUU3TETxyZGHU1rOfP2mwJpjo-LecmhKSRmHalgNWMySZL3CZPV7azH1rGhE9-w32G1iBfqlQgYpUXiaw99jWFVI3M5KIBGOHUh_-7yP89OQmaO_zqTSbLJQzVsVbS3v4d2uX0T38cTWEZrBm-8rh4mTtb2JpMysRdC3v9YvOSovsWo5iiAyCkt2yegGlrlTOfewYmCqJWCJOUSCL8qwzQff4k_eBimC8MZrsEoIth6FSjIjC8dTmjQyEyCBv-NxpYjqqzn897c0OWhK7b_WOvW0fUKWJlSgQ7XiprGl0tHZZuU4eGztNpw_-FYkC7Aumxytk_Sp-1W64GHCsez99UozwKeNxcPZxEnCQDBqFY_SUtPX1SPoNpOnArj2bg5eVpW8uqx7wlFhS00WxbmR7ykzRlPVfQ04oFGhCpZFuPtdGLQrPkpGmf0NGIaBYweWQAGiQycnfbIbKD8CwP4VcRZaTrjzpYPnSRjUkhbS4C7GWmqpCpRGw94lBcsWVp8k9bVTRIlZ8DmaAlkynfV7CtxfMH8ILof-2GXIj26NtGrJg_es8NnqRxoW__dsH-np1nhi3XicNcJ-MltW1tCvpPH1nPEb2hHydExdZQHqUE1pLqgt6z2YrgAsaGEwn1x40IXNdgOhilBe5xI1G5OWa1eQP5-CC_6rM1U5jlQeF_UH2aODbkfol2YiUtIg8sOJ-6slXmqSGqGcIUAXO_A6y5iob-0yKC_mJ4LTXcF2mBGTg-1oSMI6DBGduMa9_-PQhpzmBhQ2tPO-9UCsLqYCqoE7sP3-yfPuHTWA0A8jhWrOkB8ii3MACvDVvXAAAXpuCKNzUOiSvQWikRiq6WQyNrZ1c0ueSijyLHt5tZ0rKHnr-Ii_3tKWW5_uxf_1-6RnhewgY2yHT5eIPX1OcqPk5puWUIIZKgPl3Ga2WCjceIFybGcG54m5oaCHAWi_BMHgUorvjWaeBvhlshk64ZQi8d-7ZgopaE2-4Z_mFzeGIfhmsEt_mvrEyoHoxzKMzcH2fJnnmKgSOA8F77hv3StWRobMuStfXFtMNnEGMcdrRoQeKUlkt8yev9iZeWABOsLg8WHz6Rquw2AJZdt-Mw0s_-99fJy_JiIrcieHyJhOf-tzvDWhxcAlz2TjQbIFIGGY0AEM6nc1hFJPBAP4Wc7fsrtLuO1E6tG6cxc1BQb6M1grdn3iec4sCMA8--133NQXBXuZnmeW9OZ8sGJOJ9rnYCV_wJfTWuVA131eo0F2RnFhgqP0P0Jn-80HybSdn-TTOIZYs5kehlTBYCo5i_PGyhdmXK6LhB7uwnF8WhUgcyddgvbbJ5d_ldLvOFQTC4O9Tz1g1AW3W-SLs23vVpkP3dfa3jhpFBCdL9v5qrw2geTvU_dt9ap23ZAo_nKEa7ynttm92JDtXBd8unuprCeBewYeY2bA3F6-K78vgjIi8KiuHkqydqTmQaA-bYZ-52Wt_kPWeuX826RISTDkvhRyIzMthOLmMQlhtdzKPDS6YZthk9wG2mWfidNo-pBjmxX9qzL1fTFal8kvm6ukwvxDw6dlkD0pBIu3B7V_xUgewL9ZD7VatZ7ZMSYKIAxmRmoNm6bXMEdobqAF_8fFzKSJBssTetLbRSG34NtFz5ZZCBE5lkH8bviVxFUZugN5kzBABP-fGhUNoY8Q29ARSDy3d5s3A-VOvnqZ25j7SBzlj7A7_BR7IBSA8Nx6KU8Ej4ck_M7kCuGdKe9fqiO84lu52a9HY6V5Y7St8S6Fh_jDpyFfOjGzoCMF8o9JC5PAiFCDY55bj--OnYcLVEVPbq3JsD-It5p5WX5FfjKzCjuu8DYmtS8MAElb20ycFzohQ1sjjQU8y4K2_x3yLireQd3KnIL6SENUuvdOewoSPUXM0tZfPqhpAGTgsfkF4MPm8vcLSlRtfCwP6YlPUCR5EX9UE2YCbNhNs-ZtfjzZujaqbYf81zI_5Br-gK2UrdXZfi94pIN7-bIycL9IcI1T2eeIRxE4x_6Z_mIbdAtFnW11HAgvd7zGHnfKmgLQiQKCKkyBfeiuNsbbRJCrrWcLAKh99ey3CjEN3YD4dx9xX3ri0gIoafAZttO956l-PTNU1t5Ya8KjFuaC_gf3ZwIlMGnKO5FlhqskmrjxjhBsZbIiwxIoVsTe3EQArdhNt7DSQaHVvzikUopl94Hrn3uCBIOATZO-Jvt88U2-0ZY6smQm5XyhGyuBEFu90Q58xMpz6T291cL4mUGrGUqXOhgJD7I6RKX4tPy1EdL2_z6YS-4dqM4HFL9-WBcnX1-VBdX0mZ-czfpQJOYtdasAGAe1n9eAX9x16A7zlIxtjf-L_3M2SmEOsurPGbUlWr1Rym7c4vkbgwflGhq08UOLz8di1Bsnn2tv1p6Y4UNd2rdmnZgm-3VhznOAYrPxymzjg0PaXT-0rhN2mcwbU9FVzgkafgqze0gRzp_H2COEW6QJ7RHxpp4Lgmnhdk1qYBp0iGXMTqAqdArNCmX2ZYe6yXVhoyqA778Jh3Hc6mlSv2-tz9ShqfXp80J3FhVU2tfwDWfGzmqZI7-qwDZmyGvjYZlGUpJwjVcf2tBgMpUWPnFgDlsO7yiENDPrdNf-h7_qLE_-VvlUKYG1uJ3ERjMsYEkg8GiLxHPAr4sabuajSjE633FNnQUbmZWbrh0l8jS_eKOzIHfuQE3IWHohSdzUDXTLadvEzmB4B4kjzRSODYhG3v3pESXiJP-WIslRGbPeBiTC0QQnReNgVp31R-2LWv1jIZD9eWHksMqEsBFMtwwEzQwHAyl4cXcn0uNg5MyqcZIIfZKpzu1iIbb5gT3AyHWg7WyEsm9XIWaS-h0DJJq7Cim4vVRnFSWQ_wA4CVFP2nzOd76NKoY8I-rlWsGdbRAWfCR7SnsIfTcbqLb4UyKQJRrwqSW3Nvex0-Jidn55oe6NP6ykWHQhezCbQL5MG5zZpxSbG8jCvnoTFMuzopzzRoCQx50e4gT50AZvFagMfmk7cYyzRQqghxPOAcywgxMzjiA9pMMBEfrUlT94ZKC8A-fQbkbHtQS1GeRCzkOZFOHf047aWCuVOmdnhMw6cMChoGvItD6lQW-9BapYzsRo9iTVMeDLAf1cVW_5rcFzjI5juJJvqwt31gGFIKynA1vkl3ijLeM9OVjzlGHlB5M90IDszKd9m0QJRhCYw79_NthKi1Sbmj6DyfzNMVH3gIjj2M9adoyxJII96jYXVQxqElV5AaXnNAQ8WdyoBkz743PSbwCz2ZhK6DWQ6h59MQSsNhTfj004oVP8MzcB0nK9IQen5tjGtBlWdODqKF0GNn75C_pElsDJi2wmWADeuu0-eL6-QUdaUX7Ww2Uu2Rb_lCkyLh0Oc8NJRpqOpaGbjSqbxaHAaoGETsmw-BN8qAwKjLQXcJRh8DD6IbHXJwBRrJKjgqFRwOyeeW5XGRlNaP59D4ecKkSfVjtzuTZSTCtfsnmYKUwjmH4nx_eJ8Rtkg35o2iVXp6SNJUaxLf5QjXo-Qo4INUT_iDydrbJnBLVe5RNeRA89TEWtTp0TKxsdwSynUvJkg22YaCZrCJUUih7YAu96GnZ-ncFQ9w4FyhBAcWFJV4uV7twx1Wn-P4ZgOtPpueVb_NCDA3IdMTRh85CRv57tTwmc83DTm7X3lHHRsrYZam92kGeqL_J4P6uCppp2tdixxZX-Hqdgf8eq6wm2FmZd9hzw26FLkHtqBi7GdHGgM8DF-MFnnmw7YZM4MjkHIS1lAr0ChBT-YeKgf28AcmziHR19rqf6Wyigz5QR38ybGMtw4inU9doXbpaUzNt2ByhPU8P1oywXQRUbZyd4jrrVqqRSYipG4DjHhscELW-RSpj9pn_xEQ530QXTeKUIf42SxDDTfraqImRNs9Ej36Z8ZYMHBozVlR3yxcMseRkhEGAcr9oXRjjxnxLqZVZc_y1AMZxbBfnNOkei54IHDo7_De-8i7ALLBxqtzuZqOWyk57Tw9baOkt6cAEEbX39Ky_WogRodKJSumo9WWfDHrmCaB32LUPv7vKyd-YQ4OFNe4E5JS6UrBNKJmC-5LEdidt8U59TN7L6j6K5Ww1SMmHG3ahrJ19olVsrH9LkOhlHPkiGg7OvFXi2HLlAByrxJhG6uFLVi5-nKF9AMGMTwavaL8I6DWtzeucegmoEh_zlEYPnb1Wlse15TvPFPI4oC3fn5-FgGAVxXhGwi4MB3VLEPIqH4p2bicDp4-Cxc1YtTnJt2V29KZKGfaWYu3bFiYAkAaH_kK8ydoO9w5bl6fAfeNi25KzsS0e8RguzNfKCH-FJoI_AmIyrZV2l-S-gTz5pQZamoMP6Q40cA_H7ppNKtb2VmSnUAvA16EdoAP2h6-ePkdcofJ65SNqL-e_c8OGE9fUhPd9OOnqRfZBK8avAD5gvgunBpItCPhF8kT1dS2cjLWqxI5C4dVPsOpfdZBvQZSP-XcxCW65z9rC89ThsJZMnWGxFClTTFo2vHC8esW2ulodmakNauOzRFYpBbK3wwEL43-6Gb14E3NWAXYiPT6n5c4W5s7oLsslChc8CqCvIdBPpLmxwvbLJ17-rYfL5ccd4fA4TbDQiQyxhyZavWIsmgfI6JeLPIvfwNB-R6e47PUxaIcd-ZsH28lkDfj6Dx96MMxnlfgYr8GU-Fd3mc76o7Xev3MM0ByHGU1tH5SKj1YX3jeoljmzukAtkDFzZNwcClDYgLcVWpVBOc4OqeOrFK2o63wGTbRVQbNlGkAJRjvQg6wXBI0G5vE-UbYyerkyglfmFGbWjsK6-4aJAky8pkS-Y_SHh_r6WXhJuAP6lU97_szBpdOD5MKE0_fCwbGGsGW1_CTe_rYW54ZEnZb2HpvQXuJ9Euth93YJGaWVfbc-EJN_4UYVI18wVHhb9lUaAMaQJvx82jFNvVG65Rw--MirRTMiq3NEYFfnTBQAPdt5zH5YtNUqnIhjrSjbJbARFfnFszrxNV9HWyYdQAQEmXcduX9DjSe2zKI0JXYQFtIUgCjgJGC6RGAF0ZpKHlIPybIJ-vaGBoDsoJb3fLZCMQq3fLNZXWWOa2hdpvXkO7AzeuCVT4TkcJzOkHhcljICF1Mkys71-hBp7bCMEr9pCaOySWlCAESbDL5WZCyY38ZwkWwsfCRqyuFvo1pPOGkHRO6cHzh_qg-lo4cY7NuB_9SE_c8RwHAEOGmW_xI-8b9PcsbryKy3-zYJ_7XUG0hBDtucJusIWxiZwE8HO7qLOSAQk83CIbZkgaHwiZv6cOmEUv3DCosHKqTAsFTLuvPtPDBYhRKGchQLcAIdGeTOF8QzCW_gc8sllrY4rr3jHd7cPIk-03zZuFM-3I7so3yGYlhXMWOrt7FodkKWPX7y9cUKcPQ_3mrUBYe4n37416O0Id4XmthTx3hxGvdK6Jc0oF--FAryZYJ7mMT3QX0EI63cObarwvEDVCsif3KkXjEvuPYRFwEt121ufneXT645QdWlhRsnVVYv9Ut6THER_V6WD0JyC7t6W8taosuCr2pnjxfB5PeYk-RDjkTiQqRkY7GJ1JZ6laaz9uBD4heCQFWG4qbo2hASsNdK1dhm9uXkBfp3_10is5CwOGXGY7xMS4mIKajXMpqm87cEsKG5SStySVc11uMD50gTkLztKB-qiCbgsfYVqg57Gdy20fK9oZt-QQve9n0g2sMfeGet1wr6IpCeziGd3k_5idq9IY34pFrVEDq4BTfyFSqVZ2oBqPoH1h-iMyDA8Kd5Esc1Z7vswHx2bOUP0QDiwI7l-3CbNEo_azjZdUMGmMQ7k1kmzYXcCLh5910N1YsQyOq4I9xKUVCLyFWPNQpMsyoWV_u0OFawzVGgJ67k-inXJuFLaqZ1JbugQHPAkGCK6frXRkMxYg1EjNXzNxytqXlwpFNxKYPvuX3ja-zr6r05baEVy_75SlQ5QKxVD1qfAW6A02CX8L09JySRYgQsfCq2XaIZlWpe9eUL0iP6-D78T6OPOXdC_2wxhvv0UJBYmuBA_jSod9AT9g3fZVODYHQuG4dtko9zNTDPJN-8rT-JTqdPKhFkoi2vVbAIYCafe5j285q9kE6Dq1W9OK5r7_gyw5Ah8xBxopWFjcxMi4thiPS_4UnBbLgdgOZA_0A88LD14UKvJ3CyukB_9o6YknLmNsMw5sCXWkOEUJzfZnpeEv4RVpUQlvUi27pMBiCr3VCq2xMUg7DZm0VKKmiz1ZWsv6oBIjkp9YmB9II2fdEiX0EyU3JQlHkTS7ZAP1-CtCk8clkrpiDXhtBWSh5th_MrZuPi-1P1GXYmeXZ9RBeYnS0RVpdIBp8a_pmF-_w8G-dqOlWQle5GJRozp60T0SxI3wG91-UUD8OiXKPAYsajkEpdM3Og0VOd5UcfwiKhFadAkOfRnJgBQKu7Hj1Kldx5ip2_ieUgnOwA0Jw1V3yyxvNxJKlFVkkUXE1FimCKpV5IwlfDeo_25w5fLlHpKinhmvdwt08zgPVvQi4WyPgqvRxg7F0o10AWHU8bdvQN0MsJd4ugVnLsAONYiQ9FFoQ3_Q4KEmGIpDm2SLmPw1q53HNJzf-GYI2M9ErF3qI83Q1O4fHEFXDrs8PIt6oPqMD3S3mxW-nBYobFH_OWBwa5rgg5XFZ-uT19SZ5lY9y5vgDdiTab8VepwydlvO9bLCq7vV8YcAGj9p8AiZUqoAlGZiWkuO_BaWBmIQPjDEjniMNqJbMJkjsK8XHO1vii5vuwRSlOiADYkZcLyVjYHgGNVnY6Ek9z8-CwAqZ4sbagS9Xpx-yIuD4zKI_rmq3rKLvnM8PMf4k_lAYoJP32xu-nsqzPwLaKIH9gv5iBJLdXlamuZp1OFL2dUj0fpHOzSdkjEJrqhkFIjIKqUJ4ZqkN4KenCs64sLCHWBjMPdtXz_w9uGDxIYl79NZ6V9JVyZQOgDuPv57l6XfgeaK62JQsmPspc_B-q48zUFzS2EInS7oV70a9GW0-6HuK7ZOSV5pflHlSv183OiSQwJ8kgWs6YcmA7aJWCmoKNGUcPyzw70sP9Ab3TSmu840iGUTBprsm0oCGDJUexgmv_351AMNcnk22WDvIqgSSW3nCMPUUPSu4eUFNh_zjd-2-b5KVE2IHJRxIdVnahLrB0TzjCPJ-QkmSG3M8_XgJ3tRIhJ_hRNA4Nl35h-jsXrAAR39VfAaSw0bzN_Q7HNV5k4HQg7MVhviTve3BxtpYfKT2misRaAYa5lfoOOLQqTDy_jZ1XKjKrNKsMawUXQrq5sPlPPHNrA-ahLJb8heYecS0uYBL_ZDlhw2l_7JsyhJF1V7yHwGNc6F0R1KmiE8Gr-MdhR3b0t7ZZipUn6A2koUCFD63sInsXTmbVaulJc2Ue-jtdQAs5_wayW9IdKMRCdJUDWf1YrqlvKKBs_b0yv9u6b1Ma2GtTeQnnQmwNY-Nbt5vhugdDqkFvO8vs612UpoBISSN7sk5nqmrdJWeuaBDjVUBxU4qOe7DMywVTVRszGvOfRECJA3xty2M7SMTOGHEv5noOudYop0Mp4nseTVDZA5y3-g==",
            "format": "openai-responses-v1",
            "id": "rs_08ae9e8c9b8d660f0169b81c88d72c8195ae20f0a143f7ef0a",
            "index": 1
          }
        ]
      }
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 1105,
    "total_tokens": 1125,
    "cost": 0.000443,
    "is_byok": false,
    "prompt_tokens_details": {
      "cached_tokens": 0,
      "cache_write_tokens": 0,
      "audio_tokens": 0,
      "video_tokens": 0
    },
    "cost_details": {
      "upstream_inference_cost": 0.000443,
      "upstream_inference_prompt_cost": 0.000001,
      "upstream_inference_completions_cost": 0.000442
    },
    "completion_tokens_details": {
      "reasoning_tokens": 960,
      "image_tokens": 0,
      "audio_tokens": 0
    }
  }
}
```

Пример запроса на питоне
```bash
python3 llm_cli.py "Привет"                                                                
Привет! Рад помочь. Что хочешь сделать: ответить на вопрос, написать текст, перевести, разобрать код, или что-то ещё? Сформулируй запрос — и я постараюсь помочь.
```