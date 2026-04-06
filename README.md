# audio2text

Запись, транскрибация и диаризация встреч на macOS (Apple Silicon optimized).

## Требования

- macOS с Apple Silicon (M1–M4) — или Linux/Windows с NVIDIA GPU
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — менеджер пакетов
- ffmpeg и portaudio

## Установка

### 1. Системные зависимости

```bash
brew install ffmpeg portaudio
```

### 2. Клонирование и установка

```bash
git clone <repo-url>
cd audio2text

# Установка uv (если ещё нет)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Полная установка для Mac (MLX + диаризация + суммаризация)
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

## CLI-команды

Все команды запускаются через `uv run audio2text` или, если пакет установлен, просто `audio2text`.

### Запись звонка с микрофона

```bash
# Записать аудио (Ctrl+C для остановки)
uv run audio2text record

# Указать конкретное устройство ввода
uv run audio2text record --device 2
```

Файл сохраняется в `recordings/YYYY-MM-DD_Day/meeting_HH-MM.opus`.

Посмотреть доступные устройства:

```bash
uv run audio2text devices
```

### Транскрибация файла в текст

```bash
# Транскрибировать один файл
uv run audio2text transcribe path/to/audio.opus

# Транскрибировать все файлы в папке
uv run audio2text transcribe path/to/folder/

# Указать язык и бэкенд
uv run audio2text transcribe file.mp3 --language en --backend mlx
```

Поддерживаемые форматы: `.opus`, `.ogg`, `.wav`, `.mp3`, `.m4a`, `.flac`, `.webm` — ffmpeg автоматически обрабатывает конвертацию.

Результат:
- `file.txt` — полный текст
- `file_timed.txt` — текст с таймкодами

### Транскрибация с диаризацией (определение спикеров)

```bash
# Диаризация одного файла
uv run audio2text diarize path/to/audio.opus

# С указанием количества спикеров
uv run audio2text diarize audio.opus --min-speakers 2 --max-speakers 4
```

Результат:
- `file.txt` — полный текст
- `file_timed.txt` — текст с таймкодами
- `file_diar.txt` — текст с разметкой спикеров (`[MM:SS-MM:SS] Speaker_N: текст`)

### Полный pipeline (транскрибация + диаризация + суммаризация)

```bash
# Обработать один файл
uv run audio2text process path/to/audio.opus

# Обработать все файлы в папке
uv run audio2text process path/to/folder/
```

Выполняет последовательно: транскрибация → диаризация → суммаризация (если включена в конфиге).

### Запись + полная обработка

Записать звонок, затем сразу обработать:

```bash
# 1. Записать (Ctrl+C для остановки)
uv run audio2text record

# 2. Обработать записанный файл
uv run audio2text process recordings/2026-04-04_Friday/meeting_14-30.opus
```

Или одной командой через shell:

```bash
FILE=$(uv run audio2text record 2>/dev/null | grep "Файл:" | awk '{print $2}') && \
uv run audio2text process "$FILE"
```

### Информация о системе

```bash
uv run audio2text info
```

Покажет: ОС, чип, доступные бэкенды (MLX/CUDA), установленные библиотеки, выбранную модель.

## GUI (графический интерфейс)

Помимо CLI доступен графический интерфейс на базе tkinter.

### Запуск

```bash
uv run python gui.py
```

Откроется окно с пятью вкладками:

| Вкладка | Описание |
|---------|----------|
| **Запись** | Запись аудио с микрофона. Выбор устройства ввода, индикатор уровня (VU-метр). Кнопка «Начать запись» / «Остановить» |
| **Транскрибация** | Транскрибация одного файла или всех файлов в папке. Выбор языка и backend (auto / mlx / faster-whisper) |
| **Диаризация** | Транскрибация + определение спикеров. Можно задать мин./макс. количество спикеров |
| **Полный pipeline** | Запуск полного цикла: транскрибация → диаризация → суммаризация. Настройки берутся из вкладки «Настройки» и `config.yaml` |
| **Настройки** | Backend, модели (MLX/faster-whisper), язык, beam size, вкл/выкл диаризации, HF-токен. Кнопки «Информация о системе» и «Аудиоустройства» |

В нижней части окна расположена панель лога, куда выводятся все события и ошибки в реальном времени.

## Сводная таблица команд

| Команда | Что делает |
|---------|-----------|
| `audio2text record` | Записать аудио с микрофона |
| `audio2text transcribe <path>` | Транскрибировать в текст |
| `audio2text diarize <path>` | Транскрибировать + определить спикеров |
| `audio2text process <path>` | Полный pipeline (transcribe + diarize + summarize) |
| `audio2text devices` | Показать аудиоустройства |
| `audio2text info` | Информация о системе и бэкендах |

## Конфигурация

Все настройки в файле `config.yaml`. Основные параметры:

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
  hf_token: ""          # обязателен для работы

summarization:
  enabled: false        # включить при необходимости
```

Переопределить конфиг:

```bash
uv run audio2text --config my_config.yaml transcribe file.opus
```

## Память и модели

При первом запуске модели скачиваются автоматически:

| Модель | Размер | Назначение |
|--------|--------|-----------|
| whisper-large-v3-turbo (MLX) | ~1.6 GB | Транскрибация |
| pyannote/speaker-diarization-3.1 | ~0.5 GB | Диаризация |
| mbart_ruDialogSum | ~2.5 GB | Суммаризация (опц.) |

Требования к памяти (unified memory): ~5 GB без LLM, ~9.5 GB с LLM.
