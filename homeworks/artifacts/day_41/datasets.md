# Датасеты для fine-tuning: классификация / генерация / extraction

Критерий отбора — популярность (загрузки/цитирования) и наличие реальных (не синтетических) данных, пригодных для покрытия требования "минимум 20% реальных примеров".

## Классификация (тональность / категории тикетов / spam)

1. **SST2 (Stanford Sentiment Treebank)** — 67k предложений, бинарная тональность, реальные movie reviews.
   https://huggingface.co/datasets/stanfordnlp/sst2
2. **IMDB (aclImdb)** — 50k отзывов (25k train / 25k test), pos/neg, реальные.
   https://huggingface.co/datasets/stanfordnlp/imdb
3. **Yelp Polarity** — 560k train / 38k test отзывов Yelp, бинарная тональность, реальные.
   https://huggingface.co/datasets/fancyzhx/yelp_polarity
4. **Amazon Reviews (amazon_polarity)** — 3.6M train / 400k test отзывов, реальные.
   https://huggingface.co/datasets/fancyzhx/amazon_polarity
5. **AG News** — 120k новостных заголовков, 4 категории (топик-классификация), реальные.
   https://huggingface.co/datasets/fancyzhx/ag_news
6. **Banking77** — 13,083 реальных клиентских запроса в банковской сфере, 77 fine-grained intents. Прямой аналог "категорий тикетов".
   https://huggingface.co/datasets/PolyAI/banking77
7. **Bitext Customer Support LLM Training Dataset** — 27k примеров, intent + category (11 категорий, 27 intents), тикеты поддержки.
   https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
8. **SMS Spam Collection (UCI)** — 5,574 реальных SMS, spam/ham.
   https://huggingface.co/datasets/ucirvine/sms_spam
9. **Enron Spam** — реальные email-письма Enron corpus, spam/ham классификация.
   https://huggingface.co/datasets/SetFit/enron_spam
10. **Emotion (dair-ai/emotion)** — 20k твитов, 6 классов эмоций, реальные тексты.
    https://huggingface.co/datasets/dair-ai/emotion

## Генерация (код в стеке / ответы в стиле компании / саммари)

1. **CodeSearchNet** — 2M пар (docstring, код) из реальных open-source репозиториев GitHub, несколько языков.
   https://huggingface.co/datasets/code-search-net/code_search_net
2. **The Stack** — датасет исходного кода из GitHub, 358 языков программирования, реальный код.
   https://huggingface.co/datasets/bigcode/the-stack
3. **CodeAlpaca-20k** — 20k пар instruction→code (Python), сгенерировано LLM, широко используется как baseline для code instruction tuning.
   https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K
4. **python_code_instructions_18k_alpaca** — 18.6k instruction→code, Python, Alpaca-формат.
   https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca
5. **HumanEval** — 164 реальных задач программирования с тестами, эталон для оценки code generation (годится и под few-shot / eval).
   https://huggingface.co/datasets/openai/openai_humaneval
6. **MBPP (Mostly Basic Python Problems)** — ~1000 задач Python с решениями и тестами.
   https://huggingface.co/datasets/google-research-datasets/mbpp
7. **CNN/DailyMail** — 300k пар (статья, саммари), реальные новостные статьи с человеческими highlights.
   https://huggingface.co/datasets/abisee/cnn_dailymail
8. **XSum** — 226k пар (статья BBC, однострочное саммари), реальные, экстремальная суммаризация.
   https://huggingface.co/datasets/EdinburghNLP/xsum
9. **SAMSum** — 16k диалогов (чаты) с человеческими саммари, реальные переписки.
   https://huggingface.co/datasets/Samsung/samsum
10. **Databricks Dolly 15k** — 15k инструкций/ответов, написанных людьми (не сгенерированы LLM), включает саммаризацию, генерацию, QA — подходит как реальный слой под "ответы в стиле компании".
    https://huggingface.co/datasets/databricks/databricks-dolly-15k

## Extraction (сущности из текста / парсинг документов)

1. **CoNLL-2003** — 22k предложений, NER (PER/ORG/LOC/MISC), реальные новостные тексты Reuters. Наиболее цитируемый NER-бенчмарк.
   https://huggingface.co/datasets/eriktks/conll2003
2. **OntoNotes 5.0** — ~1.7M токенов, 18 типов сущностей, реальные тексты (новости, разговоры, веб).
   https://huggingface.co/datasets/tner/ontonotes5
3. **WikiANN (PAN-X)** — NER на 282 языках, реальные тексты Wikipedia с автоматической разметкой.
   https://huggingface.co/datasets/unimelb-nlp/wikiann
4. **SQuAD 2.0** — 150k пар вопрос-ответ, извлечение спана из реального текста Wikipedia.
   https://huggingface.co/datasets/rajpurkar/squad_v2
5. **FUNSD** — 199 реальных отсканированных форм, извлечение key-value полей (document parsing).
   https://huggingface.co/datasets/nielsr/funsd
6. **CORD (Consolidated Receipt Dataset)** — 1000 реальных чеков с полной разметкой полей (парсинг документов).
   https://huggingface.co/datasets/naver-clova-ix/cord-v2
7. **DocVQA** — реальные отсканированные документы (счета, отчёты) с вопросами и ответами-извлечениями.
   https://huggingface.co/datasets/lmms-lab/DocVQA
8. **TACRED** — 106k предложений, relation extraction (извлечение отношений между сущностями), реальные новостные и веб-тексты.
   https://huggingface.co/datasets/DFKI-SLT/tacred
9. **ADE Corpus V2** — реальные медицинские тексты, извлечение сущностей "препарат — побочный эффект".
   https://huggingface.co/datasets/ade-benchmark-corpus/ade_corpus_v2
10. **NYT29 (relation extraction)** — реальные статьи New York Times с разметкой сущностей и отношений.
    https://huggingface.co/datasets/DFKI-SLT/nyt29

## Как использовать

Для выполнения условия "минимум 20% реальных данных, остальное можно сгенерировать через ИИ" — берите нужный объём примеров из соответствующего датасета выше как реальный слой, конвертируйте в формат `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`, остальные 80% догенерируйте через API, используя реальные примеры как few-shot затравку для сохранения стиля, домена и распределения меток/сущностей.
