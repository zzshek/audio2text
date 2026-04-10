"""Суммаризация текстов: локальные seq2seq модели и LLM."""

from __future__ import annotations

import re
import time
from pathlib import Path

from logger import logger


def _truncate_repetitions(text: str) -> str:
    """Агрессивно убирает все повторы из вывода LLM."""
    # 1. Убираем inline-повторы (одна фраза повторяется внутри строки)
    # "С Новым годом. С Новым годом. С Новым годом." → "С Новым годом."
    text = re.sub(r"(.{15,}?)\1{2,}", r"\1", text)

    # 2. Разбиваем на блоки по --- и убираем дубликаты
    blocks = re.split(r"\n\s*---\s*\n|\n{3,}", text)
    seen = set()
    result = []
    for block in blocks:
        key = block.strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append(block)
    text = "\n\n".join(result)

    # 3. Убираем повторяющиеся строки подряд
    lines = text.split("\n")
    out = []
    prev = None
    for line in lines:
        s = line.strip()
        if s == prev and s:
            continue
        prev = s
        out.append(line)
    return "\n".join(out).strip()


def _clean_transcript(text: str) -> str:
    """Чистит таймкоды, оставляет спикеров в читаемом виде."""
    # [00:01-00:06] SPEAKER_02: текст  →  Спикер 2: текст
    text = re.sub(r"\[[\d:]+[-–][\d:]+\]\s*", "", text)
    # SPEAKER_02 → Спикер 2 (убираем ведущий ноль)
    text = re.sub(
        r"SPEAKER_0*(\d+)",
        lambda m: f"Спикер {m.group(1)}",
        text)
    # Unknown → Спикер ?
    text = re.sub(r"Unknown:", "Спикер ?:", text)
    # Убираем пустые строки подряд
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _detect_torch_device(preferred: str = "auto") -> str:
    """Определяет устройство для PyTorch."""
    if preferred not in ("auto", ""):
        return preferred
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class Summarizer:
    """Суммаризация через seq2seq модели (MBart, T5 и т.д.)."""

    def __init__(self, config: dict):
        cfg = config.get("summarization", {})
        self.model_name: str = cfg.get("model", "Kirili4ik/mbart_ruDialogSum")
        self.device: str = _detect_torch_device(cfg.get("device", "auto"))
        self.max_length: int = cfg.get("max_length", 250)
        self.min_length: int = cfg.get("min_length", 50)
        self.num_beams: int = cfg.get("num_beams", 4)
        self.context: str = cfg.get("context", "")

        self._tokenizer = None
        self._model = None

    def _load_model(self):
        """Ленивая загрузка модели."""
        if self._model is not None:
            return

        logger.info(f"Загрузка модели суммаризации: {self.model_name} (device={self.device})")
        start = time.time()

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

        if self.device == "cuda":
            torch.cuda.empty_cache()

        logger.info(f"Модель загружена за {time.time() - start:.1f} сек")

    def summarize(self, text: str) -> str:
        """Суммаризирует один текст."""
        self._load_model()

        import torch

        if self.context:
            text = f"Контекст: {self.context}\n\n{text}"

        inputs = self._tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self._model.generate(
                inputs["input_ids"],
                num_beams=self.num_beams,
                max_length=self.max_length,
                min_length=self.min_length,
                early_stopping=True,
            )

        return self._tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def summarize_file(self, input_path: str, output_path: str | None = None) -> str:
        """Суммаризирует текстовый файл.

        Args:
            input_path: путь к текстовому файлу.
            output_path: путь для сохранения (None = рядом с input как _summary.txt).

        Returns:
            Текст суммаризации.
        """
        text = Path(input_path).read_text(encoding="utf-8")
        logger.info(f"Суммаризация: {Path(input_path).name} ({len(text)} символов)")
        start = time.time()

        summary = self.summarize(text)

        if output_path is None:
            p = Path(input_path)
            output_path = str(p.parent / f"{p.stem}_summary.txt")

        Path(output_path).write_text(summary, encoding="utf-8")
        logger.info(f"Суммаризация завершена за {time.time() - start:.1f} сек → {output_path}")
        return summary


class LLMSummarizer:
    """Суммаризация через LLM (локальные через mlx-lm или API)."""

    def __init__(self, config: dict):
        cfg = config.get("llm", {})
        self.provider: str = cfg.get("provider", "local")
        self.local_model: str = cfg.get("local_model", "mlx-community/Qwen2.5-7B-Instruct-4bit")
        self.api_model: str = cfg.get("api_model", "")
        self.api_key: str = cfg.get("api_key", "")
        self.api_base_url: str = cfg.get("api_base_url", "")
        self.task: str = cfg.get("task", "summarize")
        self.context: str = config.get("summarization", {}).get("context", "")

        self._mlx_model = None
        self._mlx_tokenizer = None

    def _get_prompt(self, text: str) -> str:
        """Формирует промпт в зависимости от задачи."""
        text = _clean_transcript(text)

        context_line = ""
        if self.context:
            context_line = f"Контекст: {self.context}\n"

        prompts = {
            "summarize": (
                f"{context_line}"
                "Ниже транскрипция рабочей встречи. "
                "Напиши структурированное резюме на русском языке "
                "в формате Markdown. Используй именно такую структуру:\n\n"
                "## Резюме\nКраткое описание встречи (2-4 предложения).\n\n"
                "## Ключевые моменты\n- пункт 1\n- пункт 2\n\n"
                "## Решения\n- решение 1\n- решение 2\n\n"
                "## Задачи\n- [ ] задача (ответственный)\n\n"
                "## Открытые вопросы\n- вопрос 1\n\n"
                "Транскрипция:\n"
            ),
            "format": (
                f"{context_line}"
                "Отформатируй эту транскрибацию встречи в читаемый вид "
                "с абзацами и заголовками в Markdown. Ответ на русском:\n\n"
            ),
            "extract_actions": (
                f"{context_line}"
                "Извлеки из транскрибации встречи список задач (action items) "
                "с указанием ответственного (если упоминается). "
                "Формат: - [ ] задача (ответственный). Ответ на русском:\n\n"
            ),
        }
        prefix = prompts.get(self.task, prompts["summarize"])
        return prefix + text

    def summarize(self, text: str) -> str:
        """Обрабатывает текст через LLM."""
        prompt = self._get_prompt(text)

        if self.provider == "local":
            return self._run_local(prompt)
        elif self.provider == "openai":
            return self._run_openai(prompt)
        elif self.provider == "anthropic":
            return self._run_anthropic(prompt)
        else:
            raise ValueError(f"Неизвестный LLM provider: {self.provider}")

    def _run_local(self, prompt: str) -> str:
        """Запуск через mlx-lm (Apple Silicon)."""
        logger.info(f"LLM (mlx-lm): {self.local_model}")
        start = time.time()

        from mlx_lm import load, generate

        if self._mlx_model is None:
            logger.info(f"Загрузка модели {self.local_model}...")
            self._mlx_model, self._mlx_tokenizer = load(self.local_model)

        result = generate(
            self._mlx_model,
            self._mlx_tokenizer,
            prompt=prompt,
            max_tokens=4096,
        )

        # Обрезаем зацикленный вывод (если модель повторяется)
        result = _truncate_repetitions(result)

        logger.info(f"LLM завершил за {time.time() - start:.1f} сек")
        return result

    def _run_openai(self, prompt: str) -> str:
        """Запуск через OpenAI API."""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.api_base_url or None)
        response = client.chat.completions.create(
            model=self.api_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        )
        return response.choices[0].message.content

    def _run_anthropic(self, prompt: str) -> str:
        """Запуск через Anthropic API."""
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.api_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
