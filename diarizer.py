"""Диаризация: определение спикеров в аудиозаписи."""

from __future__ import annotations

import time
from pathlib import Path

from logger import logger


class Diarizer:
    """Speaker diarization через pyannote.audio.

    Определяет кто и когда говорит, затем объединяет с транскрибацией.
    """

    def __init__(self, config: dict):
        cfg = config.get("diarization", {})
        self.model_name: str = cfg.get("model", "pyannote/speaker-diarization-3.1")
        self.hf_token: str = cfg.get("hf_token", "")
        self.min_speakers: int | None = cfg.get("min_speakers")
        self.max_speakers: int | None = cfg.get("max_speakers")
        self.device: str = cfg.get("device", "cpu")

        self._pipeline = None

    def _load_pipeline(self):
        """Ленивая загрузка pipeline (pyannote тяжёлый)."""
        if self._pipeline is not None:
            return

        logger.info(f"Загрузка pyannote pipeline: {self.model_name}...")
        start = time.time()

        from pyannote.audio import Pipeline
        import torch

        if not self.hf_token:
            raise ValueError(
                "Для pyannote нужен HuggingFace token. "
                "Укажите hf_token в config.yaml секции diarization. "
                "Получить: https://huggingface.co/settings/tokens"
            )

        self._pipeline = Pipeline.from_pretrained(
            self.model_name,
            token=self.hf_token,
        )

        # Устройство
        if self.device == "mps":
            try:
                import torch
                if torch.backends.mps.is_available():
                    self._pipeline.to(torch.device("mps"))
                    logger.info("pyannote: используется MPS (экспериментально)")
                else:
                    logger.warning("MPS недоступен, используется CPU")
            except Exception:
                logger.warning("MPS не удалось инициализировать, используется CPU")
        elif self.device == "cuda":
            import torch
            if torch.cuda.is_available():
                self._pipeline.to(torch.device("cuda"))
                logger.info("pyannote: используется CUDA")

        logger.info(f"pyannote pipeline загружен за {time.time() - start:.1f} сек")

    def diarize(self, audio_path: str) -> list[dict]:
        """Выполняет диаризацию аудиофайла.

        Returns:
            [{"start": float, "end": float, "speaker": str}, ...]
        """
        self._load_pipeline()

        logger.info(f"Диаризация: {Path(audio_path).name}")
        start = time.time()

        params = {}
        if self.min_speakers is not None:
            params["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            params["max_speakers"] = self.max_speakers

        result = self._pipeline(audio_path, **params)

        # pyannote >= 3.3 may return DiarizeOutput (named tuple) instead of
        # a bare Annotation.  Extract the annotation in that case.
        if hasattr(result, "itertracks"):
            diarization = result
        elif hasattr(result, "__iter__"):
            # DiarizeOutput is tuple-like; first element is the Annotation
            diarization = result[0] if result else result
        else:
            diarization = result

        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })

        elapsed = time.time() - start
        speakers = set(t["speaker"] for t in turns)
        logger.info(
            f"Диаризация завершена за {elapsed:.1f} сек "
            f"({len(speakers)} спикеров, {len(turns)} сегментов)"
        )
        return turns

    @staticmethod
    def merge_transcription_and_diarization(
        transcription: list[dict],
        diarization: list[dict],
    ) -> list[dict]:
        """Объединяет транскрибацию с диаризацией по таймкодам.

        Для каждого сегмента транскрибации находит спикера,
        чей интервал максимально перекрывается.

        Returns:
            [{"start": float, "end": float, "text": str, "speaker": str}, ...]
        """
        if not diarization:
            return [
                {**seg, "speaker": "Speaker_0"} for seg in transcription
            ]

        merged = []
        for seg in transcription:
            seg_start = seg["start"]
            seg_end = seg["end"]

            best_speaker = "Unknown"
            best_overlap = 0.0

            for turn in diarization:
                # Пересечение интервалов
                overlap_start = max(seg_start, turn["start"])
                overlap_end = min(seg_end, turn["end"])
                overlap = max(0.0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = turn["speaker"]

            merged.append({
                "start": seg_start,
                "end": seg_end,
                "text": seg["text"],
                "speaker": best_speaker,
            })

        return merged

    @staticmethod
    def format_diarized_text(segments: list[dict]) -> str:
        """Форматирует диаризованный текст для сохранения.

        Объединяет последовательные сегменты одного спикера.
        """
        if not segments:
            return ""

        lines = []
        current_speaker = None
        current_texts = []
        current_start = 0.0
        current_end = 0.0

        for seg in segments:
            if seg["speaker"] != current_speaker:
                # Сохраняем предыдущий блок
                if current_speaker is not None and current_texts:
                    _start = _format_time(current_start)
                    _end = _format_time(current_end)
                    text = " ".join(current_texts)
                    lines.append(f"[{_start}-{_end}] {current_speaker}: {text}")

                current_speaker = seg["speaker"]
                current_texts = [seg["text"]]
                current_start = seg["start"]
                current_end = seg["end"]
            else:
                current_texts.append(seg["text"])
                current_end = seg["end"]

        # Последний блок
        if current_speaker is not None and current_texts:
            _start = _format_time(current_start)
            _end = _format_time(current_end)
            text = " ".join(current_texts)
            lines.append(f"[{_start}-{_end}] {current_speaker}: {text}")

        return "\n".join(lines)


def _format_time(seconds: float) -> str:
    """Форматирует секунды в MM:SS или HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
