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


def transcribe_file(audio_path: str, config: dict, unload: bool = False) -> dict:
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

    if unload:
        transcriber.unload()

    return result


def diarize_file(audio_path: str, config: dict, transcription: dict | None = None,
                 unload: bool = False) -> list[dict]:
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
        if unload:
            diar.unload()
        return merged

    if unload:
        diar.unload()
    return turns


def summarize_file(text_path: str, config: dict, unload: bool = False) -> str:
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

    if unload and hasattr(s, "unload"):
        s.unload()

    return summary


def process_file(audio_path: str, config: dict) -> None:
    """Полный pipeline для одного файла: транскрибация → диаризация → суммаризация."""
    audio = Path(audio_path)
    logger.info(f"═══ Обработка: {audio.name} ═══")
    total_start = time.time()

    # 1. Транскрибация
    logger.info("[1/4] Транскрибация...")
    t1 = time.time()
    result = transcribe_file(str(audio), config, unload=True)
    logger.info(f"[1/4] Транскрибация — {time.time() - t1:.1f} сек")

    # 2. Диаризация
    diar_cfg = config.get("diarization", {})
    if diar_cfg.get("enabled", True) and diar_cfg.get("hf_token"):
        logger.info("[2/4] Диаризация...")
        t2 = time.time()
        try:
            diarize_file(str(audio), config, transcription=result, unload=True)
            logger.info(f"[2/4] Диаризация — {time.time() - t2:.1f} сек")
        except Exception as e:
            logger.error(f"Ошибка диаризации: {e}")
    elif diar_cfg.get("enabled", True):
        logger.warning("[2/4] Диаризация — пропуск (нет hf_token)")
    else:
        logger.info("[2/4] Диаризация — отключена")

    # 3. Суммаризация
    summary = ""
    sum_cfg = config.get("summarization", {})
    llm_cfg = config.get("llm", {})
    if sum_cfg.get("enabled") or llm_cfg.get("enabled"):
        # Предпочитаем _diar.txt (со спикерами) если есть, иначе .txt
        diar_path = audio.parent / f"{audio.stem}_diar.txt"
        txt_path = diar_path if diar_path.exists() else audio.with_suffix(".txt")
        if txt_path.exists():
            logger.info("[3/4] Суммаризация...")
            t3 = time.time()
            try:
                summary = summarize_file(str(txt_path), config, unload=True)
                logger.info(f"[3/4] Суммаризация — {time.time() - t3:.1f} сек")
            except Exception as e:
                logger.error(f"Ошибка суммаризации: {e}")
    else:
        logger.info("[3/4] Суммаризация — отключена")

    # 4. Экспорт в Obsidian
    obs_cfg = config.get("obsidian", {})
    if obs_cfg.get("enabled") and summary:
        logger.info("[4/4] Экспорт в Obsidian...")
        t4 = time.time()
        try:
            export_obsidian_note(audio, summary, config)
            logger.info(f"[4/4] Obsidian — {time.time() - t4:.1f} сек")
        except Exception as e:
            logger.error(f"Ошибка экспорта Obsidian: {e}")
    else:
        logger.info("[4/4] Obsidian — отключён")

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


# ── Obsidian export ───────────────────────────────────────────────────────


def _parse_meeting_meta(audio_path: Path) -> dict:
    """Извлекает метаданные встречи из имени файла и сопутствующих файлов."""
    stem = audio_path.stem  # "2026-04-08 11:00 Gen AI"
    parts = stem.split(" ", 2)
    date = parts[0] if len(parts) > 0 else ""
    time_str = parts[1] if len(parts) > 1 else ""
    title = parts[2] if len(parts) > 2 else stem

    # Количество спикеров из _diar.txt
    speakers = 0
    diar_file = audio_path.parent / f"{stem}_diar.txt"
    if diar_file.exists():
        text = diar_file.read_text(encoding="utf-8")
        import re
        speakers = len(set(re.findall(r"SPEAKER_\d+", text)))

    return {
        "date": date,
        "time": time_str,
        "title": title,
        "speakers": speakers,
        "source": audio_path.name,
    }


def export_obsidian_note(
    audio_path: Path, summary: str, config: dict
) -> Path | None:
    """Экспортирует саммари встречи как Markdown-заметку в Obsidian vault."""
    obs_cfg = config.get("obsidian", {})
    vault_path = obs_cfg.get("vault_path", "")
    folder = obs_cfg.get("folder", "Meetings")

    if not vault_path:
        logger.warning("Obsidian vault_path не указан")
        return None

    vault = Path(vault_path)
    if not vault.exists():
        logger.warning(f"Obsidian vault не найден: {vault}")
        return None

    meta = _parse_meeting_meta(audio_path)
    safe_title = meta["title"].replace("/", "-").replace("\\", "-")

    # Отдельная папка для каждой встречи: Meetings/2026-04-10 Название/
    meeting_dir = vault / folder / f"{meta['date']} {safe_title}"
    meeting_dir.mkdir(parents=True, exist_ok=True)

    md_name = f"{meta['date']} {safe_title}.md"
    md_path = meeting_dir / md_name

    # Frontmatter
    lines = [
        "---",
        f"date: {meta['date']}",
        f"time: \"{meta['time']}\"",
        f"type: meeting",
    ]
    if meta["speakers"]:
        lines.append(f"speakers: {meta['speakers']}")
    lines += [
        f"source: \"{meta['source']}\"",
        f"tags: [meeting]",
        "---",
        "",
        f"# {safe_title} — {meta['date']}",
        "",
    ]

    # Саммари (уже в markdown формате от LLM)
    lines.append(summary)

    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Obsidian: {md_path}")
    return md_path


# ── Incremental processing ────────────────────────────────────────────────


def find_unprocessed(config: dict) -> list[Path]:
    """Находит аудиофайлы в recordings/ без готового _summary.txt."""
    output_dir = Path(
        config.get("recording", {}).get("output_dir", "recordings"))
    if not output_dir.exists():
        return []

    unprocessed = []
    # Ищем рекурсивно — аудио может быть в подпапках дня или встречи
    for audio_file in sorted(output_dir.rglob("*")):
        if not audio_file.is_file() or audio_file.suffix.lower() not in SUPPORTED_AUDIO:
            continue
        summary_file = audio_file.parent / f"{audio_file.stem}_summary.txt"
        if not summary_file.exists():
            unprocessed.append(audio_file)
    return unprocessed


def process_new(config: dict) -> int:
    """Обрабатывает все новые (без саммари) записи. Возвращает кол-во."""
    files = find_unprocessed(config)
    if not files:
        logger.info("Нет новых записей для обработки")
        return 0
    logger.info(f"Найдено {len(files)} необработанных записей")
    for f in files:
        process_file(str(f), config)
    return len(files)


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


def record_live(
    config: dict,
    devices=None,
    stop_event=None,
    on_chunk=None,
    vu_callback=None,
    custom_name: str = "",
) -> Path:
    """Запись с real-time транскрибацией: текст пишется в файл прямо во время записи.

    Args:
        devices: int, list[int], или None. Несколько устройств → микширование.
        stop_event: threading.Event для остановки извне (GUI). Если None — Ctrl+C.
        on_chunk: callback(text: str) — вызывается при готовности нового текста.
        vu_callback: callback(device_idx: int, rms: float) — уровень громкости для VU-метра.
        custom_name: пользовательское название для файла.
    """
    import tempfile
    import threading

    import numpy as np
    import sounddevice as sd
    import soundfile as sf

    from recorder import (
        _make_session_dir, _make_filename, _encode_opus, _has_ffmpeg,
        _mix_device_frames,
    )
    from transcriber import create_transcriber

    cfg_rec = config.get("recording", {})
    sample_rate = cfg_rec.get("sample_rate", 16000)
    channels = cfg_rec.get("channels", 1)
    output_dir = cfg_rec.get("output_dir", "recordings")
    chunk_seconds = config.get("live", {}).get("chunk_seconds", 30)

    # Normalize devices
    if devices is None:
        dev_list = [None]
    elif isinstance(devices, int):
        dev_list = [devices]
    else:
        dev_list = devices if devices else [None]

    # Загружаем модель до начала записи
    logger.info("Загрузка модели транскрибации...")
    transcriber = create_transcriber(config)

    session_dir = _make_session_dir(output_dir)
    base_name = _make_filename()
    if custom_name.strip():
        base_name = f"{base_name} {custom_name.strip()}"

    # Подпапка для каждой записи
    session_dir = session_dir / base_name
    session_dir.mkdir(parents=True, exist_ok=True)
    output_txt = session_dir / f"{base_name}_live.txt"

    dev_frames: list[list[np.ndarray]] = [[] for _ in dev_list]
    lock = threading.Lock()
    _stop = stop_event or threading.Event()
    last_chunk_idx = 0
    blocksize = int(sample_rate * 0.5)  # 500ms блоки

    def make_callback(idx):
        def cb(indata, frame_count, time_info, status):
            if status:
                logger.warning(f"Audio status: {status}")
            with lock:
                dev_frames[idx].append(indata.copy())
            rms = float(np.sqrt(np.mean(indata ** 2)))
            if vu_callback:
                vu_callback(idx, rms)
            else:
                bars = min(int(rms * 200), 40)
                print(f"\r  {'█' * bars}{'░' * (40 - bars)} {rms:.4f}", end="", flush=True)
        return cb

    def _transcribe_new_frames():
        nonlocal last_chunk_idx
        with lock:
            available = min(len(df) for df in dev_frames) - last_chunk_idx
        if available <= 0:
            return
        with lock:
            end = last_chunk_idx + available
            if len(dev_list) == 1:
                audio_data = np.concatenate(dev_frames[0][last_chunk_idx:end], axis=0)
            else:
                slices = [df[last_chunk_idx:end] for df in dev_frames]
                audio_data = _mix_device_frames(slices)
            time_offset = last_chunk_idx * blocksize / sample_rate
            last_chunk_idx = end

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, audio_data, sample_rate)
            result = transcriber.transcribe(tmp_path)
            segments = result.get("segments", [])
            if segments:
                lines = []
                for seg in segments:
                    ts = _fmt(seg["start"] + time_offset)
                    lines.append(f"[{ts}] {seg['text']}")
                chunk_text = "\n".join(lines) + "\n"
                with open(output_txt, "a", encoding="utf-8") as f:
                    f.write(chunk_text)
                if on_chunk:
                    on_chunk(chunk_text)
                else:
                    print(f"\n  +{len(segments)} сегментов -> {output_txt.name}")
                logger.info(
                    f"[Live] +{int(time_offset)}s: {len(segments)} сегментов"
                )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def transcription_loop():
        while True:
            if _stop.wait(timeout=chunk_seconds):
                _transcribe_new_frames()
                break
            _transcribe_new_frames()

    t = threading.Thread(target=transcription_loop, daemon=True)
    t.start()

    logger.info(
        f"Запись + live-транскрибация (чанки по {chunk_seconds} сек). "
        f"Ctrl+C для остановки."
    )

    # Открываем потоки для всех устройств
    streams = []
    for i, dev in enumerate(dev_list):
        streams.append(sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            device=dev,
            callback=make_callback(i),
            blocksize=blocksize,
        ))

    try:
        for s in streams:
            s.start()
        while not _stop.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        _stop.set()
        for s in streams:
            s.stop()
            s.close()
        if not stop_event:
            print()

    logger.info("Транскрибация остатка...")
    t.join(timeout=300)

    # Сохраняем полный аудиофайл
    if any(dev_frames):
        all_audio = _mix_device_frames(dev_frames)
        audio_format = cfg_rec.get("format", "opus")
        if audio_format == "opus" and _has_ffmpeg():
            audio_path = session_dir / f"{base_name}.opus"
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                sf.write(tmp_path, all_audio, sample_rate)
                _encode_opus(tmp_path, str(audio_path), str(cfg_rec.get("bitrate", "48k")))
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        else:
            audio_path = session_dir / f"{base_name}.wav"
            sf.write(str(audio_path), all_audio, sample_rate)
        logger.info(f"Аудио сохранено: {audio_path}")

    logger.info(f"Текст: {output_txt}")
    return output_txt


def _fmt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
