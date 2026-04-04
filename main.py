"""audio2text — CLI для записи, транскрибации и диаризации встреч."""

from __future__ import annotations

from pathlib import Path

import click

from processor import load_config


@click.group()
@click.option("--config", "-c", default="config.yaml", help="Путь к конфигурации")
@click.pass_context
def cli(ctx, config):
    """audio2text — запись и транскрибация встреч (Apple Silicon optimized)."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)


# ── record ──────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--device", "-d", type=int, default=None, help="ID аудиоустройства (см. audio2text devices)")
@click.pass_context
def record(ctx, device):
    """Записать аудио с микрофона."""
    from recorder import Recorder

    rec = Recorder(ctx.obj["config"])
    try:
        path = rec.record(device=device)
        if path and path.exists():
            click.echo(f"\nФайл: {path}")
    except KeyboardInterrupt:
        click.echo("\nЗапись остановлена.")


# ── transcribe ──────────────────────────────────────────────────────────────


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--model", "-m", default=None, help="Модель (переопределяет конфиг)")
@click.option("--language", "-l", default=None, help="Язык (ru, en, auto...)")
@click.option("--backend", "-b", type=click.Choice(["auto", "mlx", "faster-whisper"]), default=None)
@click.pass_context
def transcribe(ctx, path, model, language, backend):
    """Транскрибировать аудиофайл или папку."""
    from processor import transcribe_file, SUPPORTED_AUDIO

    config = ctx.obj["config"]
    cfg_t = config.setdefault("transcription", {})
    if backend:
        cfg_t["backend"] = backend
    if model:
        if backend == "faster-whisper" or (not backend and cfg_t.get("backend") == "faster-whisper"):
            cfg_t["fw_model"] = model
        else:
            cfg_t["mlx_model"] = model
    if language:
        cfg_t["language"] = language

    p = Path(path)
    if p.is_file():
        transcribe_file(str(p), config)
    elif p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_AUDIO)
        if not files:
            click.echo(f"Нет аудиофайлов в {p}")
            return
        for f in files:
            transcribe_file(str(f), config)
    click.echo("Готово.")


# ── diarize ─────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--min-speakers", type=int, default=None, help="Мин. число спикеров")
@click.option("--max-speakers", type=int, default=None, help="Макс. число спикеров")
@click.pass_context
def diarize(ctx, path, min_speakers, max_speakers):
    """Диаризация аудиофайла (определение спикеров)."""
    from processor import diarize_file, transcribe_file, SUPPORTED_AUDIO

    config = ctx.obj["config"]
    cfg_d = config.setdefault("diarization", {})
    if min_speakers is not None:
        cfg_d["min_speakers"] = min_speakers
    if max_speakers is not None:
        cfg_d["max_speakers"] = max_speakers

    p = Path(path)
    files = [p] if p.is_file() else sorted(f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_AUDIO)

    for f in files:
        # Сначала транскрибируем (если ещё нет), затем диаризуем
        txt_path = f.with_suffix(".txt")
        if txt_path.exists():
            # Транскрибация уже есть — нужно восстановить segments
            result = transcribe_file(str(f), config)
        else:
            result = transcribe_file(str(f), config)
        diarize_file(str(f), config, transcription=result)

    click.echo("Готово.")


# ── process ─────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.pass_context
def process(ctx, path):
    """Полный pipeline: транскрибация + диаризация + суммаризация."""
    from processor import process_file, process_directory

    config = ctx.obj["config"]
    p = Path(path)
    if p.is_file():
        process_file(str(p), config)
    elif p.is_dir():
        process_directory(str(p), config)
    click.echo("Готово.")


# ── devices ─────────────────────────────────────────────────────────────────


@cli.command()
def devices():
    """Показать доступные аудиоустройства."""
    import sounddevice as sd
    click.echo(sd.query_devices())


# ── info ────────────────────────────────────────────────────────────────────


@cli.command()
@click.pass_context
def info(ctx):
    """Показать информацию о системе и доступных backend'ах."""
    from processor import show_info
    click.echo(show_info(ctx.obj["config"]))


# ── gui ────────────────────────────────────────────────────────────────────


@cli.command()
def gui():
    """Открыть графический интерфейс (macOS / desktop)."""
    from gui import main as gui_main
    gui_main()


# ── точка входа ─────────────────────────────────────────────────────────────


def main():
    cli()


if __name__ == "__main__":
    main()
