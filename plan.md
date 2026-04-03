# Plan: Meeting Recorder & Transcriber for macOS

## Цель

Приложение для macOS (Apple Silicon M1-M4), которое:
1. Записывает аудио встреч с микрофона
2. Сохраняет записи в структурированные папки по дате
3. Транскрибирует речь в текст с GPU-ускорением (Metal)
4. Выполняет диаризацию (определение кто говорит)
5. Позволяет указать LLM-модель для обработки

---

## Apple Silicon: GPU-ускорение

### Проблема с текущим стеком

`faster-whisper` использует CTranslate2, который **НЕ поддерживает Metal/MPS**.
На Apple Silicon он работает только на CPU. Нужна замена.

### Решение: MLX (Apple's ML framework)

MLX — фреймворк от Apple, оптимизированный под unified memory M-серии.
Использует Metal GPU на полную мощность без копирования данных CPU↔GPU.

| Задача | Текущий стек | Новый стек (Apple Silicon) |
|--------|-------------|---------------------------|
| Транскрибация | faster-whisper (CPU only) | **mlx-whisper** (Metal GPU) |
| Диаризация | pyannote.audio (MPS частично) | **pyannote.audio** (CPU) + MPS где поддерживается |
| Суммаризация | T-pro-it-1.0 (неправильная загрузка) | **mlx-lm** или seq2seq модели |

### Автоопределение платформы

```python
import platform, subprocess

def detect_device():
    """Определяет лучший доступный backend"""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"       # Apple Silicon → MLX (Metal GPU)
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"  # NVIDIA GPU
        if torch.backends.mps.is_available():
            return "mps"   # Fallback MPS
    except ImportError:
        pass
    return "cpu"
```

---

## Модели HuggingFace (все работают локально)

### Транскрибация (Whisper)

| Модель | Размер | Скорость | Качество | Backend |
|--------|--------|----------|----------|---------|
| `mlx-community/whisper-large-v3-turbo` | ~1.6 GB | Быстрая | Отличное | MLX (Apple Silicon) |
| `mlx-community/whisper-large-v3-turbo-q4` | ~0.6 GB | Очень быстрая | Хорошее | MLX (4-bit квант.) |
| `mlx-community/whisper-large-v3-mlx` | ~3 GB | Средняя | Лучшее | MLX (full precision) |
| `openai/whisper-large-v3-turbo` | ~1.6 GB | Средняя | Отличное | CUDA/CPU (faster-whisper) |
| `openai/whisper-medium` | ~1.5 GB | Быстрая | Хорошее | CUDA/CPU (faster-whisper) |

**Рекомендация:** `mlx-community/whisper-large-v3-turbo` — на M-серии в 3-10x быстрее faster-whisper.

### Диаризация (определение спикеров)

| Модель | Размер | Примечание |
|--------|--------|------------|
| `pyannote/speaker-diarization-3.1` | ~18 MB | SOTA качество, требует HF token (gated model) |
| `pyannote/segmentation-3.0` | ~5 MB | Только сегментация (компонент diarization) |
| `speechbrain/spkrec-ecapa-voxceleb` | ~80 MB | Speaker embeddings, можно использовать для кластеризации |

**Примечание:** pyannote.audio на Apple Silicon работает на CPU. MPS-backend нестабилен —
многие операторы PyTorch не реализованы для MPS в контексте pyannote.
Для часовой записи CPU-обработка занимает ~2-5 минут на M1/M2 — приемлемо.

### Суммаризация (русский язык)

| Модель | Размер | Тип | Назначение |
|--------|--------|-----|------------|
| `Kirili4ik/mbart_ruDialogSum` | ~2.4 GB | Seq2Seq | **Суммаризация диалогов** — идеально для встреч |
| `RussianNLP/FRED-T5-Summarizer` | ~3.4 GB | Seq2Seq | Лучший общий русский суммаризатор |
| `IlyaGusev/mbart_ru_sum_gazeta` | ~2.4 GB | Seq2Seq | Суммаризация новостей/текстов |
| `cointegrated/rut5-base-absum` | ~0.9 GB | Seq2Seq | Самый легковесный |

**Рекомендация:** `Kirili4ik/mbart_ruDialogSum` — обучена именно на диалогах.

### LLM для пост-обработки (опционально, через MLX)

| Модель | Размер | Примечание |
|--------|--------|------------|
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | ~4 GB | Хороший русский, быстрый на M-серии |
| `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | ~4 GB | Универсальный |
| `mlx-community/gemma-2-9b-it-4bit` | ~5 GB | Высокое качество |

---

## Архитектура

```
audio2text/
├── config.yaml                 # Конфигурация (модели, пути, формат)
├── recorder.py                 # Запись аудио с микрофона
├── transcriber.py              # Транскрибация (mlx-whisper / faster-whisper)
├── diarizer.py                 # Диаризация (pyannote.audio)
├── processor.py                # Оркестратор: запись -> транскрибация -> диаризация
├── main.py                     # CLI точка входа
├── logger.py                   # Логирование (существует)
├── converter.py                # Конвертация форматов (существует)
├── summariser.py               # Суммаризация (существует, нужен рефакторинг)
└── recordings/                 # Папка с записями
    └── 2026-04-03_Thursday/
        ├── meeting_14-30.opus       # Аудиозапись
        ├── meeting_14-30.txt        # Полный текст
        ├── meeting_14-30_timed.txt  # Текст с таймкодами
        └── meeting_14-30_diar.txt   # Текст с диаризацией (кто говорит)
```

---

## Формат аудио: Opus (OGG-контейнер)

**Почему Opus:**
- Лучший кодек для речи — разработан специально для голоса
- 48 kbps mono = ~21 МБ/час (в 5 раз меньше лимита 100 МБ)
- Даже при 96 kbps mono = ~43 МБ/час — отличное качество с запасом
- Поддерживается mlx-whisper и faster-whisper напрямую
- Open-source, без лицензионных ограничений

**Выбранные параметры:** 48 kbps, mono, 16kHz sample rate — оптимально для речи.

---

## Этапы разработки

### Этап 1: Конфигурация (config.yaml)

```yaml
# Запись
recording:
  format: opus           # opus | wav | mp3
  bitrate: 48k           # для opus: 24k-96k
  sample_rate: 16000     # 16kHz достаточно для речи
  channels: 1            # mono
  output_dir: recordings # корневая папка записей

# Транскрибация
transcription:
  # Backend: mlx (Apple Silicon GPU) | faster-whisper (CUDA/CPU)
  backend: auto          # auto | mlx | faster-whisper

  # MLX модели (Apple Silicon)
  mlx_model: mlx-community/whisper-large-v3-turbo

  # faster-whisper модели (CUDA/CPU)
  fw_model: large-v3-turbo  # tiny, base, small, medium, large-v3, large-v3-turbo
  fw_device: auto            # auto | cpu | cuda
  fw_compute_type: int8      # int8, float16, float32

  language: ru           # код языка или auto
  beam_size: 5

# Диаризация
diarization:
  enabled: true
  model: pyannote/speaker-diarization-3.1
  hf_token: ""           # HuggingFace token (нужен для pyannote)
  min_speakers: null     # мин. кол-во спикеров (null = авто)
  max_speakers: null     # макс. кол-во спикеров (null = авто)
  device: cpu            # cpu | mps (mps нестабилен для pyannote)

# Суммаризация
summarization:
  enabled: false
  model: Kirili4ik/mbart_ruDialogSum   # или RussianNLP/FRED-T5-Summarizer
  device: auto                          # auto | cpu | cuda | mps
  max_length: 250
  min_length: 50

# LLM для пост-обработки (опционально)
llm:
  enabled: false
  provider: local        # local | openai | anthropic
  # Локальная модель (через mlx-lm)
  local_model: mlx-community/Qwen2.5-7B-Instruct-4bit
  # Или API
  api_model: ""          # gpt-4o, claude-sonnet-4-20250514
  api_key: ""
  task: summarize        # summarize | format | extract_actions
```

### Этап 2: Запись аудио (recorder.py)

- Использовать `sounddevice` + `soundfile` для записи с микрофона
- Кодирование в Opus через `ffmpeg` (subprocess)
- Автоматическое создание папки: `recordings/2026-04-03_Thursday/`
- Имя файла: `meeting_HH-MM.opus`
- Запись по нажатию Enter (старт/стоп) или Ctrl+C для остановки
- Показывать уровень громкости в реальном времени

**Зависимости:** `sounddevice`, `soundfile`, `numpy`
**Системные:** `brew install ffmpeg portaudio`

### Этап 3: Транскрибация (transcriber.py)

Два backend с единым интерфейсом:

```python
class TranscriberBase(ABC):
    def transcribe(self, audio_path: str) -> dict: ...

class MLXTranscriber(TranscriberBase):
    """Apple Silicon — использует Metal GPU через mlx-whisper"""
    def __init__(self, model="mlx-community/whisper-large-v3-turbo", language="ru"):
        import mlx_whisper
        self.engine = mlx_whisper
        self.model = model
        self.language = language

    def transcribe(self, audio_path):
        result = self.engine.transcribe(
            audio_path,
            path_or_hf_repo=self.model,
            language=self.language,
        )
        return result

class FasterWhisperTranscriber(TranscriberBase):
    """CUDA/CPU — использует CTranslate2"""
    # ... существующий код из main.py

def create_transcriber(config) -> TranscriberBase:
    """Фабрика: автоматически выбирает лучший backend"""
    backend = config.get("backend", "auto")
    if backend == "auto":
        backend = "mlx" if detect_device() == "mlx" else "faster-whisper"
    if backend == "mlx":
        return MLXTranscriber(config["mlx_model"], config["language"])
    return FasterWhisperTranscriber(config["fw_model"], config["language"])
```

### Этап 4: Диаризация (diarizer.py)

- `pyannote.audio` для определения спикеров
- На Apple Silicon работает на CPU (MPS нестабилен для pyannote)
- Объединение с транскрибацией: сопоставление таймкодов whisper + спикеров pyannote
- Формат вывода:
  ```
  [00:00-00:15] Спикер 1: Добрый день, начнём встречу...
  [00:15-00:32] Спикер 2: Да, у меня есть обновления по проекту...
  ```
- Требует HuggingFace token
- Fallback: если pyannote недоступен — пропустить диаризацию

**Зависимости:** `pyannote.audio`, `torch`

### Этап 5: Оркестратор (processor.py)

Единый pipeline:
```
запись → транскрибация (MLX GPU) → диаризация (CPU) → суммаризация/LLM
```

CLI-команды:
- `audio2text record` — начать запись
- `audio2text transcribe [path]` — транскрибировать файл/папку
- `audio2text diarize [path]` — диаризация файла/папки
- `audio2text process [path]` — полный pipeline
- `audio2text info` — показать обнаруженное оборудование и доступные backend'ы

### Этап 6: CLI (main.py)

- `click` для CLI
- Подкоманды: `record`, `transcribe`, `diarize`, `process`, `info`
- Флаги: `--model`, `--language`, `--device`, `--backend`, `--config`

### Этап 7: Рефакторинг summariser.py

**Текущая проблема:** `t-tech/T-pro-it-1.0` загружается как Seq2Seq, но это decoder-only модель.
**Исправление:** заменить на `Kirili4ik/mbart_ruDialogSum` (настоящая seq2seq для диалогов).

### Этап 8: LLM интеграция (опционально)

- Локальные модели через `mlx-lm` (Metal GPU на Apple Silicon)
- API: OpenAI, Anthropic
- Задачи: суммаризация, форматирование, action items

---

## Зависимости

### pyproject.toml

```toml
[project]
name = "audio2text"
version = "0.2.0"
description = "Meeting recorder, transcriber & diarizer for macOS (Apple Silicon optimized)"
requires-python = ">=3.12"
dependencies = [
    "pyyaml>=6.0",
    "click>=8.1",
    "sounddevice>=0.5",
    "soundfile>=0.13",
    "numpy>=1.26",
]

[project.optional-dependencies]
# Apple Silicon (MLX) — рекомендуется для Mac
mlx = [
    "mlx>=0.22",
    "mlx-whisper>=0.4",
]
# CUDA/CPU (faster-whisper)
cuda = [
    "faster-whisper>=1.2.0",
]
# Диаризация
diarization = [
    "pyannote.audio>=3.3",
    "torch>=2.0",
]
# Суммаризация
summarization = [
    "transformers>=4.40",
    "torch>=2.0",
]
# LLM (локальные модели через MLX)
llm = [
    "mlx-lm>=0.20",
]
# Всё вместе для Mac
mac = [
    "audio2text[mlx,diarization,summarization]",
]
# Всё вместе для CUDA
nvidia = [
    "audio2text[cuda,diarization,summarization]",
]

[project.scripts]
audio2text = "main:cli"
```

### Системные зависимости (macOS)

```bash
brew install ffmpeg portaudio
```

### Установка

```bash
# Apple Silicon Mac (полная установка)
uv pip install -e ".[mac]"

# Только транскрибация на Mac
uv pip install -e ".[mlx]"

# NVIDIA GPU
uv pip install -e ".[nvidia]"
```

---

## Порядок реализации

| #  | Задача | Приоритет |
|----|--------|-----------|
| 1  | Создать `config.yaml` с дефолтными значениями | Высокий |
| 2  | Реализовать `recorder.py` (запись в opus) | Высокий |
| 3  | Реализовать `transcriber.py` (MLX + faster-whisper backend) | Высокий |
| 4  | Реализовать `diarizer.py` (pyannote) | Высокий |
| 5  | Реализовать `processor.py` (pipeline) | Высокий |
| 6  | Обновить `main.py` с CLI через click | Высокий |
| 7  | Обновить `pyproject.toml` с зависимостями | Высокий |
| 8  | Рефакторинг `summariser.py` (исправить модель) | Средний |
| 9  | LLM интеграция через `mlx-lm` | Средний |
| 10 | Команда `info` — диагностика оборудования | Средний |
| 11 | Тесты | Средний |

---

## Требования к памяти (Apple Silicon)

| Задача | Модель | RAM (unified memory) |
|--------|--------|---------------------|
| Транскрибация | whisper-large-v3-turbo (MLX) | ~2 GB |
| Транскрибация | whisper-large-v3-turbo-q4 (MLX) | ~0.8 GB |
| Диаризация | pyannote/speaker-diarization-3.1 | ~0.5 GB |
| Суммаризация | mbart_ruDialogSum | ~2.5 GB |
| LLM | Qwen2.5-7B-Instruct-4bit (MLX) | ~4.5 GB |
| **Итого (полный pipeline без LLM)** | | **~5 GB** |
| **Итого (полный pipeline с LLM)** | | **~9.5 GB** |

Всё укладывается в 16 GB unified memory с запасом.
