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
    """Имя файла вида 2026-04-06 14-00"""
    return datetime.now().strftime("%Y-%m-%d %H-%M")


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _mix_device_frames(dev_frames: list[list]) -> np.ndarray:
    """Миксует аудио с нескольких устройств (сложение сигналов + клиппинг)."""
    if len(dev_frames) == 1:
        return np.concatenate(dev_frames[0], axis=0)
    min_blocks = min(len(df) for df in dev_frames)
    mixed = []
    for j in range(min_blocks):
        block = dev_frames[0][j].astype(np.float64)
        for i in range(1, len(dev_frames)):
            block += dev_frames[i][j]
        np.clip(block, -1.0, 1.0, out=block)
        mixed.append(block.astype(np.float32))
    return np.concatenate(mixed, axis=0)


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
        self._muted = False
        self._lock = threading.Lock()

    def list_devices(self) -> str:
        """Возвращает список доступных аудиоустройств."""
        return sd.query_devices()

    def record(self, devices=None, vu_callback=None, custom_name: str = "") -> Path:
        """Записывает аудио до вызова stop() или Ctrl+C.

        Args:
            devices: int, list[int], или None. Несколько устройств → микширование.
            vu_callback: callback(device_idx: int, rms: float) для VU-метра. Если None — консоль.
            custom_name: пользовательское название для файла (добавляется к дате).

        Returns:
            Path к сохранённому аудиофайлу.
        """
        # Normalize devices
        if devices is None:
            dev_list = [None]
        elif isinstance(devices, int):
            dev_list = [devices]
        else:
            dev_list = devices if devices else [None]

        session_dir = _make_session_dir(self.output_dir)
        base_name = _make_filename()
        if custom_name.strip():
            # Убираем символы, недопустимые в имени файла (/ \ : * ? " < > |)
            safe_name = custom_name.strip().replace("/", "-").replace("\\", "-")
            safe_name = safe_name.translate(str.maketrans(':"*?<>|', "_" * 7))
            base_name = f"{base_name} {safe_name}"

        # Подпапка для каждой записи: recordings/2026-04-10_Friday/2026-04-10 16-04 Название/
        session_dir = session_dir / base_name
        session_dir.mkdir(parents=True, exist_ok=True)

        self._recording = True
        self._dev_frames: list[list[np.ndarray]] = [[] for _ in dev_list]
        self._vu_callback = vu_callback
        self._streams: list = []
        blocksize = int(self.sample_rate * 0.5)

        def make_callback(idx):
            def cb(indata, frames, time_info, status):
                if status:
                    logger.warning(f"Audio status: {status}")
                if self._muted:
                    silence = np.zeros_like(indata)
                    with self._lock:
                        self._dev_frames[idx].append(silence)
                    if self._vu_callback:
                        self._vu_callback(idx, 0.0)
                    return
                with self._lock:
                    self._dev_frames[idx].append(indata.copy())
                rms = float(np.sqrt(np.mean(indata ** 2)))
                if self._vu_callback:
                    self._vu_callback(idx, rms)
                else:
                    bars = min(int(rms * 200), 40)
                    print(f"\r  {'█' * bars}{'░' * (40 - bars)} {rms:.4f}", end="", flush=True)
            return cb
        self._make_callback = make_callback

        logger.info(
            f"Запись начата (sample_rate={self.sample_rate}, "
            f"channels={self.channels}, format={self.audio_format})"
        )
        for i, d in enumerate(dev_list):
            name = sd.query_devices(d if d is not None else sd.default.device[0])["name"]
            label = f" {i + 1}" if len(dev_list) > 1 else ""
            logger.info(f"Устройство{label}: {name}")

        start_time = time.time()

        for i, dev in enumerate(dev_list):
            self._streams.append(sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=dev,
                callback=make_callback(i),
                blocksize=blocksize,
            ))

        try:
            for s in self._streams:
                s.start()
            logger.info("Нажмите Ctrl+C для остановки записи...")
            while self._recording:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self._recording = False
            for s in self._streams:
                s.stop()
                s.close()

        duration = time.time() - start_time
        logger.info(f"Запись остановлена. Длительность: {duration:.1f} сек")

        if not any(self._dev_frames):
            logger.warning("Нет записанных данных")
            return Path()

        audio_data = _mix_device_frames(self._dev_frames)

        output_path = self._save(audio_data, session_dir, base_name)
        logger.info(f"Сохранено: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} МБ)")
        return output_path

    def stop(self) -> None:
        """Останавливает запись из другого потока."""
        self._recording = False

    def switch_devices(self, devices) -> None:
        """Переключает устройства записи на лету без остановки.

        Args:
            devices: int, list[int], или None — новые устройства.
        """
        if not self._recording:
            return

        if devices is None:
            new_devs = [None]
        elif isinstance(devices, int):
            new_devs = [devices]
        else:
            new_devs = devices if devices else [None]

        # Останавливаем текущие streams
        for s in self._streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass

        blocksize = int(self.sample_rate * 0.5)

        # Подгоняем dev_frames под новое количество устройств
        with self._lock:
            while len(self._dev_frames) < len(new_devs):
                self._dev_frames.append([])

        # Открываем новые streams
        self._streams = []
        for i, dev in enumerate(new_devs):
            self._streams.append(sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=dev,
                callback=self._make_callback(i),
                blocksize=blocksize,
            ))

        for s in self._streams:
            s.start()

        for i, d in enumerate(new_devs):
            name = sd.query_devices(d if d is not None else sd.default.device[0])["name"]
            logger.info(f"Устройство переключено: {name}")

    def mute(self) -> None:
        """Выключает микрофон (записывает тишину)."""
        self._muted = True
        logger.info("🔇 Микрофон выключен (mute)")

    def unmute(self) -> None:
        """Включает микрофон."""
        self._muted = False
        logger.info("🔊 Микрофон включён")

    @property
    def is_muted(self) -> bool:
        return self._muted

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
