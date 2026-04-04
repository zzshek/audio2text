"""Транскрибация аудио: MLX (Apple Silicon GPU) и faster-whisper (CUDA/CPU)."""

from __future__ import annotations

import platform
import time
from abc import ABC, abstractmethod
from pathlib import Path

from logger import logger


def detect_device() -> str:
    """Определяет лучший доступный backend."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx.core  # noqa: F401
            return "mlx"
        except ImportError:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class TranscriberBase(ABC):
    """Базовый интерфейс транскрибатора."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> dict:
        """Транскрибирует аудиофайл.

        Returns:
            {"segments": [{"start": float, "end": float, "text": str}, ...]}
        """


class MLXTranscriber(TranscriberBase):
    """Apple Silicon — Metal GPU через mlx-whisper."""

    def __init__(self, model: str = "mlx-community/whisper-large-v3-turbo",
                 language: str = "ru", beam_size: int = 5):
        self.model = model
        self.language = language if language != "auto" else None
        self.beam_size = beam_size

        logger.info(f"MLX Whisper: загрузка модели {model}...")
        start = time.time()
        import mlx_whisper  # noqa: F401
        self._engine = mlx_whisper
        logger.info(f"MLX Whisper готов ({time.time() - start:.1f} сек)")

    def transcribe(self, audio_path: str) -> dict:
        logger.info(f"[MLX] Транскрибация: {Path(audio_path).name}")
        start = time.time()

        result = self._engine.transcribe(
            audio_path,
            path_or_hf_repo=self.model,
            language=self.language,
            beam_size=self.beam_size,
            word_timestamps=True,
        )

        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            })

        elapsed = time.time() - start
        logger.info(
            f"[MLX] Транскрибация завершена за {elapsed:.1f} сек "
            f"({len(segments)} сегментов)"
        )
        return {"segments": segments, "language": result.get("language", self.language)}


class FasterWhisperTranscriber(TranscriberBase):
    """CUDA / CPU через faster-whisper (CTranslate2)."""

    def __init__(self, model_size: str = "large-v3-turbo", language: str = "ru",
                 device: str = "auto", compute_type: str = "int8",
                 beam_size: int = 5):
        self.language = language if language != "auto" else None
        self.beam_size = beam_size

        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        # float16 на CUDA, int8 на CPU
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        logger.info(f"faster-whisper: загрузка {model_size} (device={device}, compute={compute_type})...")
        start = time.time()

        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info(f"faster-whisper готов ({time.time() - start:.1f} сек)")

    def transcribe(self, audio_path: str) -> dict:
        logger.info(f"[faster-whisper] Транскрибация: {Path(audio_path).name}")
        start = time.time()

        segments_iter, info = self.model.transcribe(
            audio_path,
            language=self.language,
            beam_size=self.beam_size,
        )

        segments = []
        for seg in segments_iter:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            })

        elapsed = time.time() - start
        logger.info(
            f"[faster-whisper] Транскрибация завершена за {elapsed:.1f} сек "
            f"(язык: {info.language}, вероятность: {info.language_probability:.2f}, "
            f"{len(segments)} сегментов)"
        )
        return {"segments": segments, "language": info.language}


def create_transcriber(config: dict) -> TranscriberBase:
    """Фабрика: создаёт транскрибатор на основе конфигурации."""
    cfg = config.get("transcription", {})
    backend = cfg.get("backend", "auto")
    language = cfg.get("language", "ru")
    beam_size = cfg.get("beam_size", 5)

    if backend == "auto":
        device = detect_device()
        backend = "mlx" if device == "mlx" else "faster-whisper"
        logger.info(f"Автоопределение backend: {backend} (device={device})")

    if backend == "mlx":
        return MLXTranscriber(
            model=cfg.get("mlx_model", "mlx-community/whisper-large-v3-turbo"),
            language=language,
            beam_size=1,  # mlx-whisper поддерживает только greedy decoding
        )
    else:
        return FasterWhisperTranscriber(
            model_size=cfg.get("fw_model", "large-v3-turbo"),
            language=language,
            device=cfg.get("fw_device", "auto"),
            compute_type=cfg.get("fw_compute_type", "int8"),
            beam_size=beam_size,
        )
