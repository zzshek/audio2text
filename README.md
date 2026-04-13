# audio2text

Запись, транскрибация, диаризация и суммаризация встреч на macOS (Apple Silicon optimized).

## Требования

- macOS с Apple Silicon (M1–M4) — или Linux/Windows с NVIDIA GPU
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — менеджер пакетов
- ffmpeg и portaudio

## Установка

### 1. Системные зависимости

```bash
brew install ffmpeg portaudio blackhole-2ch
```

> `blackhole-2ch` — виртуальное аудиоустройство для захвата системного звука (Telegram, Zoom, Teams, браузер).

### 2. Клонирование и установка

```bash
git clone <repo-url>
cd audio2text

# Установка uv (если ещё нет)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Полная установка для Mac (MLX + диаризация + суммаризация + LLM)
uv sync --extra mac

# Или минимальная (только транскрибация через MLX)
uv sync --extra mlx

# Или для NVIDIA GPU
uv sync --extra nvidia
```

### 3. Токен HuggingFace (для диаризации)

Диаризация использует gated-модель `pyannote/speaker-diarization-3.1`. Нужен токен:

1. Зарегистрируйтесь на [huggingface.co](https://huggingface.co)
2. Примите лицензию модели: [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Создайте токен: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Укажите в `config.yaml`:

```yaml
diarization:
  hf_token: "hf_ваш_токен"
```

### 4. Настройка записи системного звука (BlackHole)

Чтобы записывать звук собеседника из Telegram, Zoom, Teams, браузера — нужно настроить виртуальное аудиоустройство:

**Шаг 1.** Откройте **Audio MIDI Setup** (Finder → Приложения → Утилиты → Настройка Audio MIDI)

**Шаг 2.** Нажмите `+` внизу слева → **Создать устройство с несколькими выходами** (Multi-Output Device)

**Шаг 3.** Отметьте галочками:
- ✅ Динамики MacBook Pro (или ваши наушники)
- ✅ BlackHole 2ch

> Это позволяет слышать звук И записывать его одновременно.

**Шаг 4.** В **Системные настройки → Звук → Выход** — выберите созданный "Multi-Output Device"

**Шаг 5.** В audio2text (GUI):
- **Микрофон:** Микрофон MacBook Pro (или AirPods) — пишет ваш голос
- **Системный звук:** BlackHole 2ch — пишет звук собеседника

Оба канала микшируются в одну запись автоматически.

> **Совет:** Если подключаете наушники, устройство автоматически переключится (при выборе "По умолчанию"). Также можно переключить вручную во время записи — через комбобокс.

## GUI (графический интерфейс)

```bash
uv run python gui.py
```

| Вкладка | Описание |
|---------|----------|
| **Запись** | Запись с микрофона/системного звука. VU-метр, Mute, горячая смена устройств |
| **Live** | Запись + real-time транскрибация (текст появляется во время записи) |
| **Транскрибация** | Транскрибация файла или папки |
| **Диаризация** | Транскрибация + определение спикеров |
| **Суммаризация** | Суммаризация с редактируемым промптом. Экспорт в Obsidian |
| **Pipeline** | Полный цикл: транскрибация → диаризация → суммаризация → Obsidian |
| **Спикеры** | Редактор базы голосов — вписать ФИО для автоопределения |
| **Монитор** | CPU, RAM, потоки процесса в реальном времени |
| **Настройки** | Backend, модели, язык, HF-токен, Obsidian vault |

## CLI-команды

```bash
# Запись
uv run audio2text record
uv run audio2text record --device 2

# Live запись + транскрибация
uv run audio2text live

# Транскрибация
uv run audio2text transcribe path/to/audio.opus

# Диаризация
uv run audio2text diarize path/to/audio.opus --min-speakers 2 --max-speakers 4

# Полный pipeline
uv run audio2text process path/to/audio.opus

# Регистрация спикеров в базе голосов
uv run audio2text speakers-register path/to/audio.opus
uv run audio2text speakers-list

# Устройства и информация
uv run audio2text devices
uv run audio2text info
```

## Структура файлов

### Записи

```
recordings/
  2026-04-10_Friday/
    2026-04-10 16-04 Название встречи/
      2026-04-10 16-04 Название встречи.opus      # аудио
      2026-04-10 16-04 Название встречи.txt        # текст
      2026-04-10 16-04 Название встречи_timed.txt  # с таймкодами
      2026-04-10 16-04 Название встречи_diar.txt   # со спикерами
      2026-04-10 16-04 Название встречи_summary.txt # резюме
```

### Obsidian

```
obsidian_vault/Meetings/
  2026-04-10 Название встречи/
    2026-04-10 Название встречи.md     # Markdown с frontmatter
```

### База спикеров

```
speakers_db/
  speakers.yaml       # маппинг голос → ФИО (редактировать в GUI или вручную)
  embeddings/          # голосовые отпечатки (.npy)
```

## Pipeline: как работает

```
АУДИО (.opus) → Транскрибация (Whisper) → Диаризация (pyannote)
  → Идентификация спикеров (по базе голосов)
  → Суммаризация (Qwen2.5-14B / API)
  → Obsidian export (.md)
```

Подробная документация: [PIPELINE.md](PIPELINE.md)

## Конфигурация

Все настройки в `config.yaml`:

```yaml
recording:
  format: opus          # opus | wav | mp3
  bitrate: "48k"        # 48k mono ≈ 21 МБ/час
  sample_rate: 16000

transcription:
  backend: auto         # auto | mlx | faster-whisper
  language: ru          # ru | en | auto

diarization:
  enabled: true
  hf_token: ""          # обязателен для диаризации

llm:
  enabled: true
  provider: local       # local | openai | anthropic
  local_model: mlx-community/Qwen2.5-14B-Instruct-4bit

obsidian:
  enabled: true
  vault_path: ""        # путь к Obsidian vault
  folder: "Meetings"
```

## Модели и память

| Модель | Размер | Назначение |
|--------|--------|-----------|
| whisper-large-v3-turbo (MLX) | ~2.3 GB | Транскрибация |
| pyannote/speaker-diarization-3.1 | ~3-4 GB | Диаризация |
| pyannote/embedding | ~0.1 GB | Идентификация спикеров |
| Qwen2.5-14B-Instruct-4bit | ~8.5 GB | Суммаризация |

Peak memory: ~8.5 GB (модели выгружаются между шагами pipeline).
