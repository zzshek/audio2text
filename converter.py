from pydub import AudioSegment
from logger import logging
import time


class AudioConverter:
    """Конвертер аудиофайлов в MP3"""

    @staticmethod
    def convert_to_mp3(input_path: str, output_path: str) -> str:
        """Конвертирует аудио в MP3 с обработкой ошибок"""
        try:
            start_time = time.time()
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format="mp3", bitrate="192k")
            logging.info(f"Конвертация {input_path} -> {output_path} завершена за {time.time() - start_time:.1f} сек")
            return output_path
        except Exception as e:
            logging.error(f"Ошибка конвертации {input_path}: {str(e)}")
            raise
