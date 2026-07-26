# Execution log day_40 — прогон 3 (local `qwen2.5-coder:7b`)

Без пауз: **5 мин** (288 с) · streak=3/18 · ok=9/18 · first_try=50.0%

| # | key | type | result | gen_sec | notes |
|---|-----|------|--------|---------|-------|
| 1 | broken_avg | bug | ok | 17.91 |  |
| 2 | broken_slice | bug | ok | 4.87 |  |
| 3 | broken_append | bug | ok | 3.4 |  |
| 4 | fizzbuzz | feature | fail | 14.67 | Traceback (most recent call last):
  File "<string>", line 6, in <module>
    as |
| 5 | wordcount | feature | ok | 13.71 |  |
| 6 | json_pretty | feature | ok | 8.48 |  |
| 7 | temp_convert | feature | ok | 15.47 |  |
| 8 | csv_to_md | feature | fail | 13.62 | Traceback (most recent call last):
  File "/Users/konstantinsokolov/dev/projects |
| 9 | password_gen | feature | ok | 11.47 |  |
| 10 | anagrams | feature | fail | 17.63 |  |
| 11 | monolith_stats | refactor | fail | 25.43 | Traceback (most recent call last):
  File "<string>", line 3, in <module>
    m= |
| 12 | validate_email | refactor | ok | 12.49 |  |
| 13 | student_grade | refactor | fail | 12.25 | Traceback (most recent call last):
  File "<string>", line 3, in <module>
    m= |
| 14 | test_fizzbuzz | test | fail | 10.85 | 
==================================== ERRORS =================================== |
| 15 | test_wordcount | test | fail | 20.99 | F                                                                        [100%]
 |
| 16 | test_bugfixes | test | fail | 20.65 | 
==================================== ERRORS =================================== |
| 17 | readme | docs | fail | 15.72 | readme len=226 |
| 18 | typing_cheatsheet | docs | ok | 46.33 |  |

Сломался на: `fizzbuzz` — Traceback (most recent call last):
  File "<string>", line 6, in <module>
    assert m.main(['5'])==0
           ^^^^^^^^^^^^^^^^
AssertionError

