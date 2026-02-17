from pathlib import Path

import time

from faster_whisper import WhisperModel

from logger import logging


class FasterWhisperTranscriber:
    """
    Транскрибатор с использованием faster-whisper (CTranslate2)
    Легковесный, работает offline на CPU, кроссплатформенный
    """

    def __init__(self, model_size: str = "medium", language: str = "ru"):
        self.model_size = model_size
        self.language = language

        logging.info(f"Загрузка модели faster-whisper ({model_size})...")
        start_time = time.time()
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        logging.info(f"Модель загружена за {time.time() - start_time:.1f} сек")

    def transcribe(self, audio_path: str) -> dict:
        """Выполняет транскрибацию аудиофайла"""
        try:
            logging.info(f"Начата транскрибация {audio_path}")
            start_time = time.time()

            segments_iter, info = self.model.transcribe(
                audio_path,
                language=self.language,
                beam_size=5,
            )

            segments = []
            for seg in segments_iter:
                segments.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                })

            logging.info(
                f"Транскрибация завершена за {time.time() - start_time:.1f} сек "
                f"(язык: {info.language}, вероятность: {info.language_probability:.2f})"
            )
            return {"segments": segments}

        except Exception as e:
            logging.error(f"Ошибка транскрибации: {str(e)}")
            raise


class AudioWorker:
    """Оркестратор процесса транскрибации с поддержкой конвертации форматов"""

    SUPPORTED_INPUT_FORMATS = ('.m4a', '.mp3', '.wav', '.ogg', '.flac')

    def __init__(self, input_dir: str = "input", output_dir: str = "output"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self._setup_dirs()

    def _setup_dirs(self):
        """Создает необходимые директории"""
        (self.output_dir / "text").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "processed").mkdir(parents=True, exist_ok=True)

    def _move_processed(self, input_file: Path):
        """Перемещает обработанный файл в архив"""
        processed_path = self.output_dir / "processed" / input_file.name
        input_file.rename(processed_path)

    def _save_results(self, result: dict, input_file: Path):
        """Сохраняет результаты транскрибации"""
        try:
            full_text = "\n".join([seg["text"] for seg in result["segments"]])
            txt_path = self.output_dir / "text" / f"{input_file.stem}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)

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
        transcriber = FasterWhisperTranscriber()

        for input_file in self.input_dir.glob("*"):
            try:
                if input_file.suffix.lower() not in self.SUPPORTED_INPUT_FORMATS:
                    continue

                logging.info(f"Начата обработка {input_file.name}")

                result = transcriber.transcribe(str(input_file))
                self._save_results(result, input_file)

                self._move_processed(input_file)

                logging.info(f"Файл {input_file.name} успешно обработан")

            except Exception as e:
                logging.error(f"Ошибка обработки {input_file.name}: {str(e)}")
                continue


def main():
    worker = AudioWorker(input_dir="m4a_files", output_dir="results")
    logging.info("Начало обработки файлов...")
    start_total = time.time()

    worker.process_all_files()

    logging.info(f"Все файлы обработаны за {time.time() - start_total:.1f} сек")


if __name__ == "__main__":
    main()
