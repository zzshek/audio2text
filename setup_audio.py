"""Автоматическая настройка Multi-Output Device для записи системного звука.

Создаёт Multi-Output Device (динамики + BlackHole) в macOS,
чтобы записывать звук собеседника из Telegram, Zoom и т.д.

Использование:
    python setup_audio.py          # создать Multi-Output
    python setup_audio.py --remove # удалить созданный
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _get_audio_devices() -> list[dict]:
    """Получает список аудиоустройств через system_profiler."""
    r = _run(["system_profiler", "SPAudioDataType", "-json"])
    if r.returncode != 0:
        return []
    try:
        data = json.loads(r.stdout)
        items = data.get("SPAudioDataType", [])
        devices = []
        for item in items:
            for dev in item.get("_items", []):
                devices.append(dev)
        return devices
    except (json.JSONDecodeError, KeyError):
        return []


def _has_blackhole() -> bool:
    """Проверяет установлен ли BlackHole."""
    r = _run(["system_profiler", "SPAudioDataType"])
    return "BlackHole" in r.stdout


def _has_multi_output() -> bool:
    """Проверяет есть ли уже Multi-Output Device."""
    r = _run(["system_profiler", "SPAudioDataType"])
    return "Multi-Output Device" in r.stdout


def create_multi_output():
    """Создаёт Multi-Output Device через AppleScript + Audio MIDI Setup."""
    if platform.system() != "Darwin":
        print("Эта команда работает только на macOS")
        sys.exit(1)

    if not _has_blackhole():
        print("BlackHole не установлен!")
        print("Установите: brew install blackhole-2ch")
        print("Затем перезагрузите Mac и запустите скрипт снова.")
        sys.exit(1)

    if _has_multi_output():
        print("Multi-Output Device уже существует.")
        print("Проверьте: Системные настройки → Звук → Выход")
        return

    # Создаём Multi-Output через AppleScript (открывает Audio MIDI Setup)
    script = '''
    tell application "Audio MIDI Setup"
        activate
    end tell

    delay 1

    tell application "System Events"
        tell process "Audio MIDI Setup"
            -- Нажимаем + внизу
            try
                click menu button 1 of window 1
                delay 0.5
                -- Выбираем "Create Multi-Output Device"
                click menu item "Create Multi-Output Device" of menu 1 of menu button 1 of window 1
                delay 1
            on error
                -- Альтернативный способ через меню
                click menu item "Create Multi-Output Device" of menu "Edit" of menu bar 1
                delay 1
            end try
        end tell
    end tell
    '''

    print("Создание Multi-Output Device...")
    print()
    print("ВАЖНО: macOS откроет Audio MIDI Setup.")
    print("В открывшемся окне:")
    print("  1. Найдите 'Multi-Output Device' в списке слева")
    print("  2. Отметьте галочками:")
    print("     ✅ MacBook Pro Speakers (или ваши наушники)")
    print("     ✅ BlackHole 2ch")
    print("  3. Закройте Audio MIDI Setup")
    print()
    print("Затем:")
    print("  Системные настройки → Звук → Выход → Multi-Output Device")
    print()

    r = _run(["osascript", "-e", script])
    if r.returncode != 0:
        # Если AppleScript не сработал, просто откроем Audio MIDI Setup
        print("Автоматическое создание не удалось.")
        print("Открываю Audio MIDI Setup — создайте вручную.")
        _run(["open", "-a", "Audio MIDI Setup"])


def show_status():
    """Показывает текущий статус аудионастройки."""
    if platform.system() != "Darwin":
        print("Эта команда работает только на macOS")
        return

    blackhole = _has_blackhole()
    multi = _has_multi_output()

    print("Статус аудионастройки:")
    print(f"  BlackHole 2ch:      {'✅ установлен' if blackhole else '❌ не установлен'}")
    print(f"  Multi-Output Device: {'✅ создан' if multi else '❌ не создан'}")
    print()

    if not blackhole:
        print("Шаг 1: brew install blackhole-2ch && перезагрузка")
    elif not multi:
        print("Шаг 2: запустите 'python setup_audio.py' для создания Multi-Output")
    else:
        print("Всё готово! В audio2text выберите:")
        print("  Микрофон: ваш микрофон")
        print("  Системный звук: BlackHole 2ch")
        print()
        print("В Системных настройках → Звук → Выход → Multi-Output Device")


if __name__ == "__main__":
    if "--status" in sys.argv:
        show_status()
    elif "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
    else:
        show_status()
        print()
        if _has_blackhole() and not _has_multi_output():
            create_multi_output()
