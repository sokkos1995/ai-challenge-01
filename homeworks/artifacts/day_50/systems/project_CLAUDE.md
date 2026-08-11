# JarvisChat — CLAUDE.md (project rules)

Чат с AI-ассистентом (DeepSeek) на **Kotlin Multiplatform + Compose Multiplatform** (Android + iOS),
с text-to-speech из коробки. Этот файл — проектные правила поверх глобального `~/.claude/CLAUDE.md`
(Open-Closed: здесь — уточнения и project-specific, база — в глобале).

## Стек (пиннится, latest-stable)
| Область | Технология |
|---|---|
| Multiplatform / UI | Kotlin Multiplatform · Compose Multiplatform (Material3) |
| Сеть | Ktor client + kotlinx.serialization |
| DI | Koin (+ koin-compose-viewmodel) |
| Async | kotlinx.coroutines (StateFlow) |
| TTS | TextToSpeechKt (`nl.marc-apps:tts`) |
| Архитектура | Clean Architecture + UDF (`core/viewmodel`) |

## Команды
```bash
export JAVA_HOME=/Users/Victor/Library/Java/JavaVirtualMachines/corretto-21.0.9/Contents/Home
./gradlew :androidApp:assembleDevDebug              # сборка Android
./gradlew ktlintCheck detekt -PchangedFiles="<files>"   # линт по изменённым файлам
./gradlew :composeApp:compileKotlinIosSimulatorArm64    # iOS compile (best-effort)
```
Ключ DeepSeek — `local.properties` → `BuildConfig.DEEPSEEK_API_KEY` → `AppConfig.deepSeekApiKey` (DI). **Не хардкодить, не коммитить.**

## Архитектура — Clean + UDF
`UdfBaseViewModel<Action, UiState, State, Event>` (генерик-порядок именно такой: State 3-й, Event 4-й).
Поток строго однонаправленный: `UI → Action → ViewModel → updateState → State → mapper → UiModel → UI`;
one-off — через `Event` (`postEvent` → `collectEvent`).

### Каталогизация фичи (`feature/<name>/`, пакет `com.jarvis.chat.feature.<name>`)
```
data/
  model/        *RequestModel / *ResponseModel / *DataModel   (@Serializable)
  datasource/   *RemoteDataSource (+Impl)
  mapper/       toXxxModel() extension-функции
  repository/   *RepositoryImpl
domain/
  model/        *Model
  repository/   *Repository (интерфейс)
  usecase/      *UseCase
presentation/
  model/        *Action / *State / *Event / *UiModel
  mapper/       *UiMapper : UiMapper<State, UiModel>
  *ViewModel.kt, *Screen.kt
di/             *Module.kt (Koin)
```
Одна сущность — один файл в своём под-каталоге.

## STRICT-конвенции (обязательны)
- **Именование моделей:** `data` — `*RequestModel` / `*ResponseModel` / `*DataModel`; `domain` — `*Model`;
  `presentation` — `*UiModel`. **Слово «DTO» / `Dto` запрещено** (классы, файлы, пакеты, комментарии).
  Голые `*Request` / `*Response` без `Model` — запрещены. **Класс с именем `*UiState` запрещён** — UI-модель это `*UiModel`.
- **State:** внутренний `*State` — `data class`, `internal`, весь экранный стейт внутри; обновление только
  `updateState { copy(...) }` — **никогда `_state.value = …`**.
- **Мапперы:** `data→domain` — top-level extension `fun XxxResponseModel.toXxxModel()` в `data/mapper/`
  (один файл на исходную модель). `State→UiModel` — класс `*UiMapper : UiMapper<State, UiModel>` в
  `presentation/mapper/`. **Никакого инлайн-маппинга** в репозитории или ViewModel.
- **Action:** `sealed interface` с вложенными `Ui` / `Internal`.
- **Видимость:** `internal` по умолчанию; `public` — только для реального cross-module API.
- **Лямбды:** именованные параметры, даже одиночные (`messages.map { message -> … }`), никогда `it`.
- **Секреты:** только через `AppConfig` из DI; не хардкодить.
- **Комментарии/KDoc:** не писать без явной просьбы.
- **Kotlin:** без `!!`, без `Any` (дженерики), без magic numbers (константы).

## Хорошие примеры (из этого проекта — как ХОЧУ)
1. `data/model/ChatCompletionRequestModel.kt` — `@Serializable internal data class …RequestModel` (не `Dto`, не голый `Request`).
2. `data/mapper/ChatCompletionResponseMapper.kt` — `internal fun ChatCompletionResponseModel.toChatMessageModel(): ChatMessageModel` (extension, не класс-маппер).
3. `presentation/mapper/CompanionUiMapper.kt` — `internal class CompanionUiMapper : UiMapper<CompanionState, CompanionUiModel>`.
4. `domain/usecase/SendMessageUseCase.kt` — тонкий use-case поверх `CompanionRepository`.
5. `presentation/CompanionViewModel.kt` — `UdfBaseViewModel<CompanionAction, CompanionUiModel, CompanionState, CompanionEvent>`, стейт через `updateState`, ошибки без краша.

## Антипаттерны (ЗАПРЕЩЕНО)
- ❌ `DeepSeekMessageDto`, пакет `data/remote/dto/`, голые `XxxRequest`/`XxxResponse` — нарушают нейминг.
- ❌ класс `XxxUiState` вместо `XxxUiModel`.
- ❌ `_state.value = …` вместо `updateState { copy(...) }`.
- ❌ инлайн-маппинг DTO↔domain прямо в репозитории/ViewModel вместо `mapper/`.
- ❌ хардкод API-ключа / `!!` / `Any` / незапрошенные комментарии / неявный `it`.

## Шаблон типичного файла (feature-класс)
```kotlin
package com.jarvis.chat.feature.companion.data.mapper

import com.jarvis.chat.feature.companion.data.model.ChatCompletionResponseModel
import com.jarvis.chat.feature.companion.domain.model.ChatMessageModel

internal fun ChatCompletionResponseModel.toChatMessageModel(): ChatMessageModel =
    ChatMessageModel(
        author = MessageAuthor.ASSISTANT,
        text = choices.firstOrNull()?.message?.content.orEmpty(),
    )
```
Порядок: `package` → импорты (полные, без `*`) → одна сущность на файл → `internal` по умолчанию.
