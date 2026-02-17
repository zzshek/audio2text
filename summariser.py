import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datetime import datetime
from tqdm import tqdm
from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    input_dir: str = "results/text"
    output_dir: str = "results/summaries"
    model_name: str = "t-tech/T-pro-it-1.0"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8  # Оптимально для 3090
    max_length: int = 200
    min_length: int = 50
    num_beams: int = 4


class TextSummarizer:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self._init_model()

    def _init_model(self):
        """Инициализация модели с обработкой ошибок"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)
            self.model.to(self.config.device)
            self.model.eval()
            torch.cuda.empty_cache()  # Очистка памяти GPU
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели: {str(e)}")

    def summarize_batch(self, texts: List[str]) -> List[str]:
        """Суммаризация батча текстов"""
        try:
            inputs = self.tokenizer(
                texts,
                max_length=1024,
                truncation=True,
                padding="longest",
                return_tensors="pt"
            ).to(self.config.device)

            with torch.no_grad():
                summaries = self.model.generate(
                    inputs["input_ids"],
                    num_beams=self.config.num_beams,
                    max_length=self.config.max_length,
                    min_length=self.config.min_length,
                    early_stopping=True
                )

            return [self.tokenizer.decode(s, skip_special_tokens=True) for s in summaries]

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return self._handle_oom(texts)

    def _handle_oom(self, texts: List[str]) -> List[str]:
        """Обработка нехватки памяти"""
        print("Обнаружена нехватка памяти GPU. Уменьшаем batch size...")
        half = len(texts) // 2
        return (self.summarize_batch(texts[:half]) +
                self.summarize_batch(texts[half:]))


class FileProcessor:
    def __init__(self, summarizer: TextSummarizer, config: Config):
        self.summarizer = summarizer
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)

    def get_input_files(self) -> List[str]:
        """Получение списка файлов для обработки"""
        return [
            f for f in os.listdir(self.config.input_dir)
            if f.endswith("_timed.txt")
        ]

    def process_files(self):
        """Основной метод обработки файлов"""
        files = self.get_input_files()
        if not files:
            print("Файлы для обработки не найдены!")
            return

        print(f"Начало обработки {len(files)} файлов на {self.config.device}...")

        for i in tqdm(range(0, len(files), self.config.batch_size)):
            batch_files = files[i:i + self.config.batch_size]
            self._process_batch(batch_files)

    def _process_batch(self, filenames: List[str]):
        """Обработка батча файлов"""
        try:
            # Чтение файлов
            texts = []
            for filename in filenames:
                with open(os.path.join(self.config.input_dir, filename), 'r', encoding='utf-8') as f:
                    texts.append(f.read())

            # Суммаризация
            summaries = self.summarizer.summarize_batch(texts)

            # Сохранение результатов
            for filename, summary in zip(filenames, summaries):
                self._save_summary(filename, summary)

        except Exception as e:
            print(f"Ошибка при обработке батча: {str(e)}")

    def _save_summary(self, input_filename: str, summary: str):
        """Сохранение результата суммаризации"""
        base_name = os.path.splitext(input_filename)[0].replace("_timed", "")
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_filename = f"{base_name}_summary_{timestamp}.txt"

        with open(os.path.join(self.config.output_dir, output_filename), 'w', encoding='utf-8') as f:
            f.write(summary)


def main():
    # Инициализация конфига с оптимизацией под 3090
    config = Config(
        batch_size=16,  # Можно увеличить для 3090
        max_length=250,
        num_beams=6
    )

    # Проверка GPU
    if config.device == "cuda":
        print(f"Используется GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM доступно: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # Создание компонентов
    summarizer = TextSummarizer(config)
    processor = FileProcessor(summarizer, config)

    # Запуск обработки
    processor.process_files()


if __name__ == "__main__":
    main()