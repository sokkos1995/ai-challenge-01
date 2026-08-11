🔥 День 50. Self-Defense — фиксы по векторам из ATTACK_REPORT

Этот файл — итог “защитного hardening” пайплайна после анализа отчёта соседа. Основной фокус: закрыть обходы детекторов секретов (hex/base64/split), утечку системного промпта и усилить offline security review, чтобы “не LLM режим” не был единственной слабой точкой.

## Что именно было закрыто

### 1) Входной guard (exfil/secret обходы)
Закрыты обходы, где секрет:
- кодируется в `hex` (раньше не распознавалось),
- прячется в `base64` с пробелами/переводами строк (раньше требовался непрерывный блок),
- прячется в url-safe `base64` (`-` / `_`),
- встречается в виде “double base64” (base64(base64(...))),
- собирается “в обход” split-нормализации: ` "s" + "k-..." ` и ` "".join([...]) `,
- использует Stripe-подобный формат `sk_live_... / sk_test_...` (underscore вместо дефиса).

Код: `app/services/llm_input_guard.py`.

### 2) Выходной guard (утечка system prompt)
Усилен детект утечки системных инструкций для варианта ответа вида:
`You are GatewayAssistant` (без “a/an”).

Дополнительно сделан case-insensitive поиск по известным сигнатурам.

Код: `app/services/llm_output_guard.py`.

### 3) Offline security review (AST-эвристика)
Усилен offline-ревью модуль `day_49_security_loop`, чтобы он ловил те же техники обхода, что были в отчёте соседа:
- `getattr(__builtins__, "ev" + "al")(...)` / `getattr(__builtins__, "ex" + "ec")(...)` → теперь детектится как Critical,
- `getattr(os, "sys" + "tem")(...)` → теперь детектится как High,
- `shell=` через переменную (не только `ast.Constant True`) → теперь флагируется, если это не константное `False`,
- SQL-инъекция через `.format(...)` и через `"..." + user_input` → детектится по AST,
- логирование/печать секретов через нейтральную переменную (`token = "..."; print(token)`) — добавлен простой taint.

Код: `homeworks/src/day_49_security_loop/security_heuristics.py`.

### 4) Прозрачность offline-режима
В `offline=True` добавлена явная warning-пометка: что security review в этом режиме работает детерминированными эвристиками, а не через LLM.

Код: `homeworks/src/day_49_security_loop/loop.py`.

## Регрессионные тесты (что добавлено/расширено)

Обновлены/добавлены проверки в:
- `tests/test_day48_guards.py`
  - `test_hex_encoded_secret_blocked`
  - `test_base64_spaced_secret_blocked`
  - `test_base64_urlsafe_secret_blocked`
  - `test_base64_double_secret_blocked`
  - `test_underscore_stripe_sk_live_blocked`
  - `test_split_prefix_s_plus_k_blocked`
  - `test_join_array_secret_blocked`
  - `test_output_blocks_system_prompt_leak_without_article`
- `tests/test_day49_security_loop.py`
  - AST кейсы: obfuscated `eval/exec`, obfuscated `os.system`, `shell` через переменную, SQL `.format()` и `"..." + ...`., taint для `print(token)`
- `tests/test_day50_self_defense.py`
  - расширен набор `cases` для `check_input(..., mode="block")` (hex/base64-spaces/url-safe/double).

## Как проверить

Запустить точечно:
```bash
pytest tests/test_day48_guards.py tests/test_day49_security_loop.py tests/test_day50_self_defense.py -q
```

## Residual / ограничения детектов (честно)
- Base64/hex detection валидирует находку только через “декод → plain secret scan”, поэтому если злоумышленник выберет нестандартное кодирование/алфавит так, что декод “не попадает” в UTF-строку, сигнатура может не сработать.
- Нормализация split/join восстанавливает самые типовые “2-фрагментные” схемы (`"s"+"k-..."`, `join(["sk-","proj-..."])` и `join(["s","k-..."])`).

## Векторы, не закрытые в этой итерации
Этот раунд hardening сделан строго под residual’ы из присланного соседом `ATTACK_REPORT.md` (hex/base64 со спец-обходами + security review обходы).
Остальные family-обходы, перечисленные в `homeworks/day_50_attack.md` (например `gzip+base64`, `rot13`, `base32`, HTML entities `&#...;`, `\\uXXXX` как литералы, `chr()/codepoints` сборка), в текущих правках не покрывались отдельной логикой и не входят в регрессионные тесты, указанные выше.

