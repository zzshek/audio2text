from pathlib import Path

import json
import subprocess
import wave
import time
import os
import urllib.request
import zipfile

from vosk import Model, KaldiRecognizer

from logger import logging


class VoskTranscriber:
    """
    Транскрибатор с использованием Vosk
    Легковесный, работает offline на CPU
    """

    MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip"
    MODEL_DIR = "vosk-model-ru-0.42"

    def __init__(self, model_path: str = None):
        self.model_path = model_path or self.MODEL_DIR
        self._ensure_model()
        logging.info(f"Загрузка модели Vosk из {self.model_path}...")
        start_time = time.time()
        self.model = Model(self.model_path)
        logging.info(f"Модель загружена за {time.time() - start_time:.1f} сек")

    def _ensure_model(self):
        """Скачивает модель если её нет"""
        if os.path.isdir(self.model_path):
            return

        zip_path = f"{self.model_path}.zip"
        if not os.path.exists(zip_path):
            logging.info(f"Скачивание модели Vosk ({self.MODEL_URL})...")
            urllib.request.urlretrieve(self.MODEL_URL, zip_path)
            logging.info("Модель скачана")

        logging.info("Распаковка модели...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(".")
        os.remove(zip_path)
        logging.info("Модель распакована")

    def transcribe(self, audio_path: str) -> dict:
        """Выполняет транскрибацию аудиофайла"""
        try:
            logging.info(f"Начата транскрибация {audio_path}")
            start_time = time.time()

            wav_path = self._to_wav(audio_path)

            wf = wave.open(wav_path, "rb")
            rec = KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(True)

            segments = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    part = json.loads(rec.Result())
                    if part.get("text"):
                        segment = {"text": part["text"]}
                        if "result" in part:
                            segment["start"] = part["result"][0]["start"]
                            segment["end"] = part["result"][-1]["end"]
                        segments.append(segment)

            # Последний фрагмент
            part = json.loads(rec.FinalResult())
            if part.get("text"):
                segment = {"text": part["text"]}
                if "result" in part:
                    segment["start"] = part["result"][0]["start"]
                    segment["end"] = part["result"][-1]["end"]
                segments.append(segment)

            wf.close()

            # Удаляем временный wav если конвертировали
            if wav_path != audio_path:
                os.remove(wav_path)

            logging.info(f"Транскрибация завершена за {time.time() - start_time:.1f} сек")
            return {"segments": segments}

        except Exception as e:
            logging.error(f"Ошибка транскрибации: {str(e)}")
            raise

    def _to_wav(self, audio_path: str) -> str:
        """Конвертирует аудио в WAV 16kHz mono через ffmpeg"""
        if audio_path.endswith(".wav"):
            return audio_path

        wav_path = audio_path.rsplit(".", 1)[0] + "_tmp.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            capture_output=True, check=True
        )
        return wav_path


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
            # 1. Полный текст
            full_text = "\n".join([seg["text"] for seg in result["segments"]])
            txt_path = self.output_dir / "text" / f"{input_file.stem}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            # 2. Текст с таймкодами
            timed_text = "\n".join(
                f"[{seg.get('start', 0):.2f}-{seg.get('end', 0):.2f}] {seg['text']}"
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
        transcriber = VoskTranscriber()

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
