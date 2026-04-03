"""Оркестратор: pipeline запись → транскрибация → диаризация → суммаризация."""

from __future__ import annotations

import time
from pathlib import Path

import yaml

from logger import logger

SUPPORTED_AUDIO = {".opus", ".ogg", ".wav", ".mp3", ".m4a", ".flac", ".webm"}


def load_config(config_path: str = "config.yaml") -> dict:
    """Загружает конфигурацию из YAML-файла."""
    p = Path(config_path)
    if not p.exists():
        logger.warning(f"Конфиг {config_path} не найден, используются значения по умолчанию")
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    logger.info(f"Сохранено: {path}")


def transcribe_file(audio_path: str, config: dict) -> dict:
    """Транскрибирует один аудиофайл. Сохраняет .txt и _timed.txt рядом."""
    from transcriber import create_transcriber

    audio = Path(audio_path)
    transcriber = create_transcriber(config)
    result = transcriber.transcribe(str(audio))
    segments = result["segments"]

    # Полный текст
    full_text = "\n".join(seg["text"] for seg in segments)
    _save_text(audio.with_suffix(".txt"), full_text)

    # Текст с таймкодами
    timed_lines = []
    for seg in segments:
        s = _fmt(seg["start"])
        e = _fmt(seg["end"])
        timed_lines.append(f"[{s}-{e}] {seg['text']}")
    _save_text(
        audio.parent / f"{audio.stem}_timed.txt",
        "\n".join(timed_lines),
    )

    return result


def diarize_file(audio_path: str, config: dict, transcription: dict | None = None) -> list[dict]:
    """Диаризация аудиофайла. Если есть транскрибация — объединяет."""
    from diarizer import Diarizer

    audio = Path(audio_path)
    diar = Diarizer(config)
    turns = diar.diarize(str(audio))

    if transcription and transcription.get("segments"):
        merged = Diarizer.merge_transcription_and_diarization(
            transcription["segments"], turns
        )
        text = Diarizer.format_diarized_text(merged)
        _save_text(audio.parent / f"{audio.stem}_diar.txt", text)
        return merged

    return turns


def summarize_file(text_path: str, config: dict) -> str:
    """Суммаризирует текстовый файл."""
    cfg = config.get("summarization", {})
    llm_cfg = config.get("llm", {})

    if llm_cfg.get("enabled"):
        from summariser import LLMSummarizer
        s = LLMSummarizer(config)
    elif cfg.get("enabled"):
        from summariser import Summarizer
        s = Summarizer(config)
    else:
        logger.info("Суммаризация отключена в конфиге")
        return ""

    text = Path(text_path).read_text(encoding="utf-8")
    summary = s.summarize(text)

    out_path = Path(text_path).parent / f"{Path(text_path).stem}_summary.txt"
    _save_text(out_path, summary)
    return summary


def process_file(audio_path: str, config: dict) -> None:
    """Полный pipeline для одного файла: транскрибация → диаризация → суммаризация."""
    audio = Path(audio_path)
    logger.info(f"═══ Обработка: {audio.name} ═══")
    total_start = time.time()

    # 1. Транскрибация
    result = transcribe_file(str(audio), config)

    # 2. Диаризация
    diar_cfg = config.get("diarization", {})
    if diar_cfg.get("enabled", True) and diar_cfg.get("hf_token"):
        try:
            diarize_file(str(audio), config, transcription=result)
        except Exception as e:
            logger.error(f"Ошибка диаризации: {e}")
    elif diar_cfg.get("enabled", True):
        logger.warning("Диаризация включена, но hf_token не указан — пропускаем")

    # 3. Суммаризация
    sum_cfg = config.get("summarization", {})
    llm_cfg = config.get("llm", {})
    if sum_cfg.get("enabled") or llm_cfg.get("enabled"):
        txt_path = audio.with_suffix(".txt")
        if txt_path.exists():
            try:
                summarize_file(str(txt_path), config)
            except Exception as e:
                logger.error(f"Ошибка суммаризации: {e}")

    elapsed = time.time() - total_start
    logger.info(f"═══ Готово: {audio.name} ({elapsed:.1f} сек) ═══")


def process_directory(dir_path: str, config: dict) -> None:
    """Обрабатывает все аудиофайлы в директории."""
    d = Path(dir_path)
    audio_files = sorted(
        f for f in d.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO
    )

    if not audio_files:
        logger.warning(f"Нет аудиофайлов в {dir_path}")
        return

    logger.info(f"Найдено {len(audio_files)} аудиофайлов в {dir_path}")
    for f in audio_files:
        process_file(str(f), config)


def show_info(config: dict) -> str:
    """Показывает информацию о системе и доступных backend'ах."""
    import platform
    lines = [
        "══════ audio2text: информация о системе ══════",
        f"OS:       {platform.system()} {platform.release()}",
        f"Machine:  {platform.machine()}",
        f"Python:   {platform.python_version()}",
    ]

    # Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        lines.append("Chip:     Apple Silicon (arm64)")
        try:
            import mlx.core as mx
            lines.append(f"MLX:      ✓ (версия {mx.__version__})")
        except ImportError:
            lines.append("MLX:      ✗ (pip install mlx mlx-whisper)")
    else:
        lines.append("Chip:     x86_64")

    # CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
            lines.append(f"CUDA:     ✓ ({gpu}, {vram:.1f} GB VRAM)")
        else:
            lines.append("CUDA:     ✗")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            lines.append("MPS:      ✓")
    except ImportError:
        lines.append("PyTorch:  ✗ (pip install torch)")

    # faster-whisper
    try:
        import faster_whisper
        lines.append(f"faster-whisper: ✓")
    except ImportError:
        lines.append("faster-whisper: ✗ (pip install faster-whisper)")

    # pyannote
    try:
        import pyannote.audio
        lines.append(f"pyannote.audio: ✓")
    except ImportError:
        lines.append("pyannote.audio: ✗ (pip install pyannote.audio)")

    # ffmpeg
    import shutil
    if shutil.which("ffmpeg"):
        lines.append("ffmpeg:   ✓")
    else:
        lines.append("ffmpeg:   ✗ (brew install ffmpeg)")

    # Выбранный backend
    from transcriber import detect_device
    device = detect_device()
    backend = "mlx" if device == "mlx" else "faster-whisper"
    lines.append(f"\nBackend (auto): {backend} (device={device})")

    # Конфиг
    cfg_t = config.get("transcription", {})
    if backend == "mlx":
        lines.append(f"Модель:   {cfg_t.get('mlx_model', 'mlx-community/whisper-large-v3-turbo')}")
    else:
        lines.append(f"Модель:   {cfg_t.get('fw_model', 'large-v3-turbo')}")

    lines.append("══════════════════════════════════════════════")
    return "\n".join(lines)


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
