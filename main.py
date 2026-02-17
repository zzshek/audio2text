from pathlib import Path

import whisperx
import torch
import ssl
import warnings
import time

from logger import logging
from converter import AudioConverter



# Игнорируем предупреждения от pydub
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Отключение проверки SSL (если необходимо)
ssl._create_default_https_context = ssl._create_unverified_context




class WhisperXTranscriber:
    """
    Транскрибатор с использованием WhisperX
    Автоматически определяет доступные устройства и настройки
    """

    def __init__(self, model_size: str = "medium", language: str = "ru"):
        self.model_size = model_size
        self.language = language
        self.device = self._get_available_device()
        self.compute_type = self._get_compute_type()
        self.model = self._load_model()
        self.alignment_model = None
        self.diarization_model = None

    def _get_available_device(self):
        """Определяет доступное устройство"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _get_compute_type(self):
        """Определяет оптимальный тип вычислений"""
        if self.device == "cpu":
            return "int8"  # Для CPU используем int8 для лучшей производительности
        else:
            return "float16" if self.device != "mps" else "float32"

    def _load_model(self):
        """Загружает модель WhisperX с учетом возможностей устройства"""
        logging.info(
            f"Загрузка модели {self.model_size} на {self.device.upper()} (compute_type={self.compute_type})...")
        start_time = time.time()

        try:
            model = whisperx.load_model(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                language=self.language
            )
            logging.info(f"Модель загружена за {time.time() - start_time:.1f} сек")
            return model
        except Exception as e:
            logging.error(f"Ошибка загрузки модели: {str(e)}")
            # Fallback на CPU если другие устройства не работают
            if self.device != "cpu":
                logging.warning("Попытка загрузки на CPU...")
                self.device = "cpu"
                self.compute_type = "int8"
                return self._load_model()
            raise

    def _load_alignment_model(self, language: str):
        """Загружает модель для выравнивания слов"""
        if self.alignment_model is None:
            logging.info(f"Загрузка модели выравнивания для языка {language}...")
            try:
                self.alignment_model, self.metadata = whisperx.load_align_model(
                    language_code=language,
                    device=self.device
                )
            except Exception as e:
                logging.error(f"Ошибка загрузки модели выравнивания: {str(e)}")
                raise

    def transcribe(self, audio_path: str, enable_diarization: bool = False) -> dict:
        """Выполняет транскрибацию с дополнительными функциями WhisperX"""
        try:
            logging.info(f"Начата транскрибация {audio_path}")
            start_time = time.time()

            # 1. Базовая транскрибация
            result = self.model.transcribe(
                audio_path,
                language=self.language,
                batch_size=4 if self.device != "cpu" else 1
            )
            logging.info(f"Базовая транскрибация завершена за {time.time() - start_time:.1f} сек")

            # 2. Выравнивание слов
            try:
                self._load_alignment_model(result["language"])
                aligned_result = whisperx.align(
                    result["segments"],
                    self.alignment_model,
                    self.metadata,
                    audio_path,
                    device=self.device
                )
                logging.info(f"Выравнивание слов завершено за {time.time() - start_time:.1f} сек")
                result = aligned_result
            except Exception as e:
                logging.error(f"Ошибка выравнивания: {str(e)}")

            return result

        except Exception as e:
            logging.error(f"Ошибка транскрибации: {str(e)}")
            raise


class AudioWorker:
    """Оркестратор процесса транскрибации с поддержкой конвертации форматов"""

    SUPPORTED_INPUT_FORMATS = ('.m4a', '.mp3', '.wav', '.ogg', '.flac')
    OUTPUT_FORMAT = 'mp3'

    def __init__(self, input_dir: str = "input", output_dir: str = "output"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self._setup_dirs()

    def _setup_dirs(self):
        """Создает необходимые директории"""
        (self.output_dir / "audio").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "text").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "processed").mkdir(parents=True, exist_ok=True)

    def _convert_audio(self, input_file: Path) -> Path:
        """Конвертирует аудиофайл в MP3 и возвращает путь к новому файлу"""
        try:
            output_path = self.output_dir / "audio" / f"{input_file.stem}.{self.OUTPUT_FORMAT}"

            # Если файл уже существует и новее оригинала
            if output_path.exists() and output_path.stat().st_mtime > input_file.stat().st_mtime:
                return output_path

            logging.info(f"Конвертация {input_file.name} -> {output_path.name}")
            start_time = time.time()

            audio = AudioSegment.from_file(input_file)
            audio.export(
                output_path,
                format=self.OUTPUT_FORMAT,
                bitrate="192k",
                parameters=["-ac", "1"]  # Моно для лучшей транскрибации
            )

            logging.info(f"Конвертация завершена за {time.time() - start_time:.1f} сек")
            return output_path

        except Exception as e:
            logging.error(f"Ошибка конвертации {input_file.name}: {str(e)}")
            raise

    def _move_processed(self, input_file: Path):
        """Перемещает обработанный файл в архив"""
        processed_path = self.output_dir / "processed" / input_file.name
        input_file.rename(processed_path)

    def _save_results(self, result: dict, input_file: Path):
        """Сохраняет результаты транскрибации"""
        try:
            # 1. Полный текст
            full_text = "\n".join([seg["text"] for seg in result["segments"]])
            txt_path = self.output_dir / "text" / f"{input_file.stem}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            # 2. Текст с таймкодами
            timed_text = "\n".join(
                f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}"
                for seg in result["segments"]
            )
            timed_path = self.output_dir / "text" / f"{input_file.stem}_timed.txt"
            with open(timed_path, "w", encoding="utf-8") as f:
                f.write(timed_text)

        except Exception as e:
            logging.error(f"Ошибка сохранения результатов: {str(e)}")
            raise

    def process_all_files(self):
        """Обрабатывает все поддерживаемые файлы в директории"""
        transcriber = WhisperXTranscriber(model_size="medium")

        for input_file in self.input_dir.glob("*"):
            try:
                # Пропускаем неподдерживаемые форматы
                if input_file.suffix.lower() not in self.SUPPORTED_INPUT_FORMATS:
                    continue

                logging.info(f"Начата обработка {input_file.name}")

                # Конвертируем в MP3 (кроме уже MP3)
                if input_file.suffix.lower() != f".{self.OUTPUT_FORMAT}":
                    audio_path = self._convert_audio(input_file)
                else:
                    audio_path = input_file

                # Транскрибация
                result = transcriber.transcribe(str(audio_path))
                self._save_results(result, input_file)

                # Перемещаем обработанный файл
                self._move_processed(input_file)

                logging.info(f"Файл {input_file.name} успешно обработан")

            except Exception as e:
                logging.error(f"Ошибка обработки {input_file.name}: {str(e)}")
                continue


if __name__ == "__main__":
    worker = AudioWorker(input_dir="m4a_files", output_dir="results")
    logging.info("Начало обработки файлов...")
    start_total = time.time()

    worker.process_all_files()

    logging.info(f"Все файлы обработаны за {time.time() - start_total:.1f} сек")