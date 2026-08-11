# JarvisChat — CLAUDE.md

Чат с AI-ассистентом (DeepSeek) на Kotlin Multiplatform + Compose Multiplatform, Android и iOS.
Text-to-speech из коробки — ответы ассистента можно озвучить.

## Стек
- Kotlin Multiplatform + Compose Multiplatform (Material3)
- Ktor client + kotlinx.serialization — сеть
- Koin — DI
- TextToSpeechKt (`nl.marc-apps:tts`) — озвучка

## Сборка и запуск
```bash
export JAVA_HOME=/Users/Victor/Library/Java/JavaVirtualMachines/corretto-21.0.9/Contents/Home
./gradlew :androidApp:assembleDevDebug
```
Ключ DeepSeek — в `local.properties` (`DEEPSEEK_API_KEY=...`), файл в `.gitignore`. Не коммить.

## Структура
- `composeApp/` — общий код (`commonMain`) + платформенное (`androidMain`, `iosMain`)
- `androidApp/` — Android-приложение

Фичу (экран чата) размести в `composeApp` в отдельном пакете, раздели логику и UI.

## Правила
- Пиши на Kotlin, следуй чистому и читаемому коду.
- Ключи и токены не хардкодить.
- Перед сдачей — собери проект.
