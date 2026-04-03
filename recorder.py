"""Запись аудио с микрофона и сохранение в Opus."""

from __future__ import annotations

import platform
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from logger import logger

# Дни недели на английском (для названия папки)
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _make_session_dir(base_dir: str) -> Path:
    """Создаёт папку вида recordings/2026-04-03_Thursday/"""
    now = datetime.now()
    day_name = _WEEKDAYS[now.weekday()]
    folder_name = f"{now.strftime('%Y-%m-%d')}_{day_name}"
    path = Path(base_dir) / folder_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_filename() -> str:
    """Имя файла вида meeting_14-30"""
    return f"meeting_{datetime.now().strftime('%H-%M')}"


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _encode_opus(wav_path: str, opus_path: str, bitrate: str = "48k") -> None:
    """Кодирует WAV → Opus через ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", wav_path,
        "-c:a", "libopus",
        "-b:a", bitrate,
        "-ac", "1",
        "-ar", "16000",
        "-application", "voip",
        opus_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr}")


class Recorder:
    """Записывает аудио с микрофона.

    Использование:
        rec = Recorder(config)
        audio_path = rec.record()  # блокирует до Ctrl+C или stop()
    """

    def __init__(self, config: dict):
        cfg = config.get("recording", {})
        self.output_dir: str = cfg.get("output_dir", "recordings")
        self.sample_rate: int = cfg.get("sample_rate", 16000)
        self.channels: int = cfg.get("channels", 1)
        self.audio_format: str = cfg.get("format", "opus")
        self.bitrate: str = str(cfg.get("bitrate", "48k"))

        self._frames: list[np.ndarray] = []
        self._recording = False
        self._lock = threading.Lock()

    def list_devices(self) -> str:
        """Возвращает список доступных аудиоустройств."""
        return sd.query_devices()

    def record(self, device: int | None = None) -> Path:
        """Записывает аудио до вызова stop() или Ctrl+C.

        Returns:
            Path к сохранённому аудиофайлу.
        """
        session_dir = _make_session_dir(self.output_dir)
        base_name = _make_filename()

        self._frames = []
        self._recording = True

        logger.info(
            f"Запись начата (sample_rate={self.sample_rate}, "
            f"channels={self.channels}, format={self.audio_format})"
        )
        if device is not None:
            logger.info(f"Устройство: {sd.query_devices(device)['name']}")
        else:
            logger.info(f"Устройство: {sd.query_devices(sd.default.device[0])['name']}")

        start_time = time.time()

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=device,
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * 0.5),  # 500ms chunks
            ):
                logger.info("Нажмите Ctrl+C для остановки записи...")
                while self._recording:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self._recording = False

        duration = time.time() - start_time
        logger.info(f"Запись остановлена. Длительность: {duration:.1f} сек")

        if not self._frames:
            logger.warning("Нет записанных данных")
            return Path()

        # Собираем аудио
        audio_data = np.concatenate(self._frames, axis=0)

        # Сохраняем
        output_path = self._save(audio_data, session_dir, base_name)
        logger.info(f"Сохранено: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} МБ)")
        return output_path

    def stop(self) -> None:
        """Останавливает запись из другого потока."""
        self._recording = False

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Callback для sounddevice — вызывается из аудио-потока."""
        if status:
            logger.warning(f"Audio status: {status}")
        with self._lock:
            self._frames.append(indata.copy())

        # Показываем уровень громкости
        rms = np.sqrt(np.mean(indata ** 2))
        bars = int(rms * 200)
        bars = min(bars, 40)
        print(f"\r  {'█' * bars}{'░' * (40 - bars)} {rms:.4f}", end="", flush=True)

    def _save(self, audio_data: np.ndarray, session_dir: Path, base_name: str) -> Path:
        """Сохраняет аудио в нужном формате."""
        if self.audio_format == "opus" and _has_ffmpeg():
            return self._save_opus(audio_data, session_dir, base_name)
        elif self.audio_format == "mp3" and _has_ffmpeg():
            return self._save_ffmpeg(audio_data, session_dir, base_name, "mp3")
        else:
            return self._save_wav(audio_data, session_dir, base_name)

    def _save_wav(self, audio_data: np.ndarray, session_dir: Path, base_name: str) -> Path:
        """Сохраняет как WAV."""
        path = session_dir / f"{base_name}.wav"
        sf.write(str(path), audio_data, self.sample_rate)
        return path

    def _save_opus(self, audio_data: np.ndarray, session_dir: Path, base_name: str) -> Path:
        """Сохраняет как Opus через ffmpeg."""
        opus_path = session_dir / f"{base_name}.opus"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        try:
            sf.write(tmp_wav, audio_data, self.sample_rate)
            _encode_opus(tmp_wav, str(opus_path), self.bitrate)
        finally:
            Path(tmp_wav).unlink(missing_ok=True)
        return opus_path

    def _save_ffmpeg(
        self, audio_data: np.ndarray, session_dir: Path, base_name: str, fmt: str
    ) -> Path:
        """Сохраняет через ffmpeg в произвольном формате."""
        out_path = session_dir / f"{base_name}.{fmt}"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        try:
            sf.write(tmp_wav, audio_data, self.sample_rate)
            cmd = ["ffmpeg", "-y", "-i", tmp_wav, "-ac", "1", str(out_path)]
            subprocess.run(cmd, capture_output=True, check=True)
        finally:
            Path(tmp_wav).unlink(missing_ok=True)
        return out_path
