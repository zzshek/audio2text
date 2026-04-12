"""База голосов: автоматическое извлечение и идентификация спикеров."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import yaml

from logger import logger

SPEAKERS_DIR = Path("speakers_db")
SPEAKERS_FILE = SPEAKERS_DIR / "speakers.yaml"
EMBEDDINGS_DIR = SPEAKERS_DIR / "embeddings"

# Порог cosine similarity для автоматического сопоставления
DEFAULT_THRESHOLD = 0.55


def _ensure_dirs():
    SPEAKERS_DIR.mkdir(exist_ok=True)
    EMBEDDINGS_DIR.mkdir(exist_ok=True)


def _load_speakers() -> dict:
    """Загружает speakers.yaml."""
    if not SPEAKERS_FILE.exists():
        return {"speakers": {}}
    with open(SPEAKERS_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "speakers" not in data:
        data["speakers"] = {}
    return data


def _save_speakers(data: dict):
    """Сохраняет speakers.yaml."""
    _ensure_dirs()
    with open(SPEAKERS_FILE, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False,
                  sort_keys=False)


def _load_embedding_model(hf_token: str):
    """Загружает модель для извлечения voice embeddings."""
    from pyannote.audio import Model, Inference

    logger.info("Загрузка модели voice embeddings...")
    start = time.time()
    model = Model.from_pretrained(
        "pyannote/embedding",
        use_auth_token=hf_token,
    )
    inference = Inference(model, window="whole")
    logger.info(f"Модель embeddings загружена за {time.time() - start:.1f} сек")
    return inference


def extract_speaker_embedding(
    audio_path: str,
    segments: list[dict],
    speaker_id: str,
    hf_token: str,
    max_seconds: float = 30.0,
) -> np.ndarray | None:
    """Извлекает усреднённый embedding для одного спикера из аудио.

    Берёт сегменты указанного спикера (до max_seconds суммарно),
    извлекает embedding каждого и усредняет.
    """
    from pyannote.core import Segment

    inference = _load_embedding_model(hf_token)

    # Собираем сегменты этого спикера
    speaker_segs = [s for s in segments if s["speaker"] == speaker_id]
    if not speaker_segs:
        logger.warning(f"Спикер {speaker_id} не найден в сегментах")
        return None

    # Берём самые длинные сегменты (до max_seconds)
    speaker_segs.sort(key=lambda s: s["end"] - s["start"], reverse=True)
    total = 0.0
    embeddings = []

    for seg in speaker_segs:
        duration = seg["end"] - seg["start"]
        if duration < 2.0:  # слишком короткий
            continue
        try:
            excerpt = Segment(seg["start"], seg["end"])
            emb = inference.crop(audio_path, excerpt)
            embeddings.append(emb)
            total += duration
            if total >= max_seconds:
                break
        except Exception as e:
            logger.debug(f"Пропуск сегмента {seg['start']:.1f}-{seg['end']:.1f}: {e}")

    if not embeddings:
        logger.warning(f"Не удалось извлечь embedding для {speaker_id}")
        return None

    # Усреднение
    avg_embedding = np.mean(np.vstack(embeddings), axis=0)
    logger.info(
        f"Embedding {speaker_id}: {len(embeddings)} сегментов, "
        f"{total:.1f} сек речи"
    )
    return avg_embedding


def register_speakers_from_diarization(
    audio_path: str,
    diar_segments: list[dict],
    hf_token: str,
) -> list[str]:
    """Извлекает embeddings всех спикеров из диаризации и сохраняет в базу.

    Для каждого нового SPEAKER_XX создаёт запись в speakers.yaml
    с пустым полем name (для заполнения пользователем).

    Returns:
        Список ID новых спикеров, добавленных в базу.
    """
    _ensure_dirs()
    data = _load_speakers()
    speakers = data["speakers"]

    # Уникальные спикеры в этой записи
    unique_speakers = sorted(set(s["speaker"] for s in diar_segments))
    audio_name = Path(audio_path).stem
    new_ids = []

    for sp_id in unique_speakers:
        # Уникальный ключ: имя файла + спикер
        voice_key = f"{audio_name}__{sp_id}"

        # Если уже есть embedding — пропускаем
        emb_file = EMBEDDINGS_DIR / f"{voice_key}.npy"
        if emb_file.exists():
            continue

        embedding = extract_speaker_embedding(
            audio_path, diar_segments, sp_id, hf_token
        )
        if embedding is None:
            continue

        # Сохраняем embedding
        np.save(str(emb_file), embedding)

        # Добавляем в speakers.yaml если нет
        if voice_key not in speakers:
            speakers[voice_key] = {
                "name": "",  # ← пользователь заполняет вручную
                "source": str(Path(audio_path).name),
                "speaker_id": sp_id,
                "embedding_file": str(emb_file.name),
            }
            new_ids.append(voice_key)

    _save_speakers(data)

    if new_ids:
        logger.info(
            f"Добавлено {len(new_ids)} новых голосов в speakers.yaml. "
            f"Заполните поле 'name' в {SPEAKERS_FILE}"
        )
    return new_ids


def identify_speakers(
    audio_path: str,
    diar_segments: list[dict],
    hf_token: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, str]:
    """Сопоставляет SPEAKER_XX с известными именами из базы.

    Returns:
        Маппинг {SPEAKER_XX: "Имя Фамилия"} для распознанных спикеров.
    """
    data = _load_speakers()
    speakers = data.get("speakers", {})

    # Собираем именованные embeddings
    named_embeddings: list[tuple[str, np.ndarray]] = []
    for key, info in speakers.items():
        name = info.get("name", "").strip()
        if not name:
            continue
        emb_file = EMBEDDINGS_DIR / info.get("embedding_file", "")
        if not emb_file.exists():
            continue
        emb = np.load(str(emb_file))
        named_embeddings.append((name, emb))

    if not named_embeddings:
        return {}

    # Извлекаем embeddings текущих спикеров
    inference = _load_embedding_model(hf_token)
    unique_speakers = sorted(set(s["speaker"] for s in diar_segments))
    mapping: dict[str, str] = {}

    from pyannote.core import Segment
    from scipy.spatial.distance import cosine

    for sp_id in unique_speakers:
        sp_segs = [s for s in diar_segments if s["speaker"] == sp_id]
        # Берём самые длинные сегменты
        sp_segs.sort(key=lambda s: s["end"] - s["start"], reverse=True)

        embeddings = []
        total = 0.0
        for seg in sp_segs[:10]:
            duration = seg["end"] - seg["start"]
            if duration < 2.0:
                continue
            try:
                emb = inference.crop(audio_path, Segment(seg["start"], seg["end"]))
                embeddings.append(emb)
                total += duration
                if total >= 20.0:
                    break
            except Exception:
                continue

        if not embeddings:
            continue

        current_emb = np.mean(np.vstack(embeddings), axis=0)

        # Сравниваем со всеми именованными
        best_name = None
        best_sim = 0.0
        for name, ref_emb in named_embeddings:
            sim = 1.0 - cosine(current_emb, ref_emb)
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_name and best_sim >= threshold:
            mapping[sp_id] = best_name
            logger.info(f"  {sp_id} → {best_name} ({best_sim:.0%})")
        else:
            logger.info(f"  {sp_id} — не распознан (лучший: {best_sim:.0%})")

    return mapping


def apply_speaker_names(
    segments: list[dict],
    mapping: dict[str, str],
) -> list[dict]:
    """Заменяет SPEAKER_XX на имена в сегментах."""
    if not mapping:
        return segments
    result = []
    for seg in segments:
        new_seg = dict(seg)
        sp = seg.get("speaker", "")
        if sp in mapping:
            new_seg["speaker"] = mapping[sp]
        result.append(new_seg)
    return result
