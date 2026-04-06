"""audio2text — macOS GUI (tkinter/ttk)."""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

from processor import load_config, SUPPORTED_AUDIO


# ── Logging handler → GUI ──────────────────────────────────────────────────


class QueueHandler(logging.Handler):
    """Отправляет log-записи в очередь для потокобезопасного вывода в GUI."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


# ── App ────────────────────────────────────────────────────────────────────


class Audio2TextApp:
    """Главное окно приложения."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("audio2text")
        self.root.geometry("820x660")
        self.root.minsize(700, 550)

        # macOS native feel
        self.root.option_add("*tearOff", False)
        try:
            self.root.tk.call("tk::unsupported::MacWindowStyle", "style",
                              self.root._w, "document", "closeBox collapseBox")
        except tk.TclError:
            pass

        self.config = load_config("config.yaml")
        self.log_queue: queue.Queue[str] = queue.Queue()
        self._running_task: threading.Thread | None = None

        self._setup_logging()
        self._build_ui()
        self._poll_log_queue()

    # ── logging ────────────────────────────────────────────────────────

    def _setup_logging(self):
        logger = logging.getLogger("audio2text")
        handler = QueueHandler(self.log_queue)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)

    def _poll_log_queue(self):
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._log(msg)
        self.root.after(100, self._poll_log_queue)

    # ── UI ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=(10, 0))

        self._build_record_tab(notebook)
        self._build_live_tab(notebook)
        self._build_transcribe_tab(notebook)
        self._build_diarize_tab(notebook)
        self._build_process_tab(notebook)
        self._build_settings_tab(notebook)

        # Log panel
        log_frame = ttk.LabelFrame(self.root, text="Лог")
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.log_text = tk.Text(log_frame, height=10, state="disabled",
                                wrap="word", font=("Menlo", 11))
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)

    # ── Record tab ─────────────────────────────────────────────────────

    def _build_record_tab(self, notebook: ttk.Notebook):
        frame = ttk.Frame(notebook, padding=15)
        notebook.add(frame, text="  Запись  ")

        # Микрофон + VU
        self.rec_mic_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Микрофон:", variable=self.rec_mic_enabled).grid(
            row=0, column=0, sticky="w", pady=(5, 0))
        self.rec_mic_var = tk.StringVar(value="По умолчанию")
        self.rec_mic_combo = ttk.Combobox(
            frame, textvariable=self.rec_mic_var, state="readonly", width=50)
        self.rec_mic_combo.grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=(5, 0))

        self.rec_mic_vu = tk.DoubleVar(value=0)
        ttk.Progressbar(frame, variable=self.rec_mic_vu,
                        maximum=1.0, mode="determinate").grid(
            row=1, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=(2, 5))

        # Системный звук + VU
        self.rec_sys_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Системный звук:", variable=self.rec_sys_enabled).grid(
            row=2, column=0, sticky="w", pady=(5, 0))
        self.rec_sys_var = tk.StringVar(value="Не выбрано")
        self.rec_sys_combo = ttk.Combobox(
            frame, textvariable=self.rec_sys_var, state="readonly", width=50)
        self.rec_sys_combo.grid(row=2, column=1, sticky="ew", padx=(10, 0), pady=(5, 0))

        self.rec_sys_vu = tk.DoubleVar(value=0)
        ttk.Progressbar(frame, variable=self.rec_sys_vu,
                        maximum=1.0, mode="determinate").grid(
            row=3, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=(2, 5))

        ttk.Button(frame, text="Обновить", width=10,
                   command=self._refresh_all_devices).grid(
            row=0, column=2, rowspan=2, padx=(5, 0), pady=5)

        # Hint (если нет BlackHole)
        self.rec_hint_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.rec_hint_var, wraplength=500,
                  foreground="gray", font=("Helvetica", 10)).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(0, 5))

        self._refresh_all_devices()

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=5, column=0, columnspan=3, pady=10)

        ttk.Button(btn_frame, text="Проверка звука",
                   command=self._test_record_devices).pack(side="left", padx=5)

        self.record_btn = ttk.Button(btn_frame, text="Начать запись",
                                     command=self._toggle_record)
        self.record_btn.pack(side="left", padx=5)

        self.record_status = ttk.Label(btn_frame, text="")
        self.record_status.pack(side="left", padx=10)

        frame.columnconfigure(1, weight=1)

        self._recorder = None
        self._recording = False

    def _test_record_devices(self):
        self._test_devices(
            self.rec_mic_enabled, self.rec_mic_var, self.rec_mic_vu,
            self.rec_sys_enabled, self.rec_sys_var, self.rec_sys_vu,
        )

    # Виртуальные loopback-устройства для захвата системного звука
    _VIRTUAL_PATTERNS = ["blackhole", "soundflower", "loopback audio", "virtual cable", "vb-cable"]

    def _get_input_device_list(self) -> list[tuple[int, str]]:
        """Возвращает [(id, name), ...] входных аудиоустройств."""
        try:
            import sounddevice as sd
            devs = sd.query_devices()
            return [(i, d["name"]) for i, d in enumerate(devs)
                    if d["max_input_channels"] > 0]
        except Exception as e:
            self._log(f"Ошибка получения устройств: {e}")
            return []

    @classmethod
    def _is_virtual_device(cls, name: str) -> bool:
        lower = name.lower()
        return any(p in lower for p in cls._VIRTUAL_PATTERNS)

    def _refresh_all_devices(self):
        devs = self._get_input_device_list()
        items = [f"{i}: {name}" for i, name in devs]
        mic_values = ["По умолчанию"] + items
        sys_values = ["Не выбрано"] + items

        # Обновляем все combos
        for combo_pair in [
            (getattr(self, "rec_mic_combo", None), getattr(self, "rec_sys_combo", None)),
            (getattr(self, "live_mic_combo", None), getattr(self, "live_sys_combo", None)),
        ]:
            mic_c, sys_c = combo_pair
            if mic_c:
                mic_c["values"] = mic_values
            if sys_c:
                sys_c["values"] = sys_values

        # Авто-выбор виртуального устройства для системного звука
        virtual_found = None
        for dev_id, name in devs:
            if self._is_virtual_device(name):
                virtual_found = f"{dev_id}: {name}"
                break

        for sys_var, hint_var in [
            (getattr(self, "rec_sys_var", None), getattr(self, "rec_hint_var", None)),
            (getattr(self, "live_sys_var", None), getattr(self, "live_hint_var", None)),
        ]:
            if sys_var and sys_var.get() == "Не выбрано" and virtual_found:
                sys_var.set(virtual_found)
            if hint_var:
                if virtual_found:
                    hint_var.set("")
                else:
                    hint_var.set(
                        "Для захвата системного звука установите BlackHole: "
                        "brew install blackhole-2ch"
                    )

    @staticmethod
    def _parse_device_id(selection: str) -> int | None:
        """Парсит ID устройства из строки '2: Device Name'."""
        if selection in ("По умолчанию", "Не выбрано", ""):
            return None
        try:
            return int(selection.split(":")[0])
        except (ValueError, IndexError):
            return None

    def _collect_devices(self, mic_enabled, mic_var, sys_enabled, sys_var):
        """Собирает список устройств и тегов ("mic"/"sys")."""
        devs = []
        tags = []
        if mic_enabled.get():
            devs.append(self._parse_device_id(mic_var.get()))
            tags.append("mic")
        if sys_enabled.get():
            dev = self._parse_device_id(sys_var.get())
            if dev is not None:
                devs.append(dev)
                tags.append("sys")
        if not devs:
            return [None], ["mic"]
        return devs, tags

    def _test_devices(self, mic_enabled, mic_var, mic_vu,
                      sys_enabled, sys_var, sys_vu):
        """Проверка звука: 3 секунды мониторинга выбранных устройств."""
        if self._running_task and self._running_task.is_alive():
            messagebox.showwarning("audio2text", "Задача уже выполняется.")
            return

        import numpy as np
        import sounddevice as sd

        devs, tags = self._collect_devices(mic_enabled, mic_var, sys_enabled, sys_var)
        cfg = self.config.get("recording", {})
        sample_rate = cfg.get("sample_rate", 16000)

        max_rms: dict[str, float] = {"mic": 0.0, "sys": 0.0}

        def make_cb(idx):
            tag = tags[idx]
            def cb(indata, frames, time_info, status):
                rms = float(np.sqrt(np.mean(indata ** 2)))
                val = min(rms * 10, 1.0)
                if tag == "mic":
                    mic_vu.set(val)
                else:
                    sys_vu.set(val)
                max_rms[tag] = max(max_rms[tag], rms)
            return cb

        streams = []
        for i, dev in enumerate(devs):
            try:
                s = sd.InputStream(
                    samplerate=sample_rate, channels=1, dtype="float32",
                    device=dev, callback=make_cb(i),
                    blocksize=int(sample_rate * 0.5),
                )
                s.start()
                streams.append(s)
            except Exception as e:
                self._log(f"Ошибка устройства {dev}: {e}")

        if not streams:
            return

        self._log("Проверка звука... (3 секунды, говорите / включите звук)")

        def stop_test():
            for s in streams:
                s.stop()
                s.close()
            mic_vu.set(0)
            sys_vu.set(0)
            lines = []
            for tag in tags:
                name = "Микрофон" if tag == "mic" else "Системный звук"
                r = max_rms[tag]
                if r < 0.001:
                    status = "НЕТ СИГНАЛА"
                elif r < 0.005:
                    status = "очень тихо"
                else:
                    status = f"OK (макс. уровень {r:.4f})"
                lines.append(f"  {name}: {status}")
            self._log("Результат проверки:\n" + "\n".join(lines))

        self.root.after(3000, stop_test)

    def _toggle_record(self):
        if not self._recording:
            self._start_record()
        else:
            self._stop_record()

    def _start_record(self):
        if self._running_task and self._running_task.is_alive():
            messagebox.showwarning("audio2text", "Задача уже выполняется, дождитесь завершения.")
            return

        from recorder import Recorder
        self._recorder = Recorder(self.config)
        self._recording = True
        self.record_btn.configure(text="Остановить")
        self.record_status.configure(text="Идёт запись...")

        devs, tags = self._collect_devices(
            self.rec_mic_enabled, self.rec_mic_var,
            self.rec_sys_enabled, self.rec_sys_var,
        )

        def vu_cb(idx, rms):
            val = min(rms * 10, 1.0)
            if idx < len(tags):
                if tags[idx] == "mic":
                    self.rec_mic_vu.set(val)
                else:
                    self.rec_sys_vu.set(val)

        def do_record():
            try:
                path = self._recorder.record(devices=devs, vu_callback=vu_cb)
                if path and path.exists():
                    self.log_queue.put(f"Файл сохранён: {path}")
            except Exception as e:
                self.log_queue.put(f"Ошибка записи: {e}")
            finally:
                self.root.after(0, self._on_record_done)

        self._running_task = threading.Thread(target=do_record, daemon=True)
        self._running_task.start()

    def _stop_record(self):
        if self._recorder:
            self._recorder.stop()

    def _on_record_done(self):
        self._recording = False
        self.record_btn.configure(text="Начать запись")
        self.record_status.configure(text="")
        self.rec_mic_vu.set(0)
        self.rec_sys_vu.set(0)

    # ── Live tab ───────────────────────────────────────────────────────

    def _build_live_tab(self, notebook: ttk.Notebook):
        frame = ttk.Frame(notebook, padding=15)
        notebook.add(frame, text="  Live  ")

        # Микрофон + VU
        self.live_mic_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Микрофон:", variable=self.live_mic_enabled).grid(
            row=0, column=0, sticky="w", pady=(5, 0))
        self.live_mic_var = tk.StringVar(value="По умолчанию")
        self.live_mic_combo = ttk.Combobox(
            frame, textvariable=self.live_mic_var, state="readonly", width=50)
        self.live_mic_combo.grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=(5, 0))

        self.live_mic_vu = tk.DoubleVar(value=0)
        ttk.Progressbar(frame, variable=self.live_mic_vu,
                        maximum=1.0, mode="determinate").grid(
            row=1, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=(2, 5))

        # Системный звук + VU
        self.live_sys_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Системный звук:", variable=self.live_sys_enabled).grid(
            row=2, column=0, sticky="w", pady=(5, 0))
        self.live_sys_var = tk.StringVar(value="Не выбрано")
        self.live_sys_combo = ttk.Combobox(
            frame, textvariable=self.live_sys_var, state="readonly", width=50)
        self.live_sys_combo.grid(row=2, column=1, sticky="ew", padx=(10, 0), pady=(5, 0))

        self.live_sys_vu = tk.DoubleVar(value=0)
        ttk.Progressbar(frame, variable=self.live_sys_vu,
                        maximum=1.0, mode="determinate").grid(
            row=3, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=(2, 5))

        ttk.Button(frame, text="Обновить", width=10,
                   command=self._refresh_all_devices).grid(
            row=0, column=2, rowspan=2, padx=(5, 0), pady=5)

        # Hint
        self.live_hint_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.live_hint_var, wraplength=500,
                  foreground="gray", font=("Helvetica", 10)).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(0, 5))

        # Заполняем списки
        devs = self._get_input_device_list()
        items = [f"{i}: {name}" for i, name in devs]
        self.live_mic_combo["values"] = ["По умолчанию"] + items
        self.live_sys_combo["values"] = ["Не выбрано"] + items
        # Авто-выбор виртуального устройства
        for dev_id, name in devs:
            if self._is_virtual_device(name):
                self.live_sys_var.set(f"{dev_id}: {name}")
                break
        else:
            self.live_hint_var.set(
                "Для захвата системного звука установите BlackHole: "
                "brew install blackhole-2ch"
            )

        # Chunk size
        ttk.Label(frame, text="Чанк (сек):").grid(row=5, column=0, sticky="w", pady=5)
        self.live_chunk_var = tk.StringVar(value="30")
        ttk.Spinbox(frame, textvariable=self.live_chunk_var,
                    from_=10, to=120, increment=5, width=6).grid(
            row=5, column=1, sticky="w", padx=(10, 0), pady=5)

        # Button + status
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=6, column=0, columnspan=3, pady=10)

        ttk.Button(btn_frame, text="Проверка звука",
                   command=self._test_live_devices).pack(side="left", padx=5)

        self.live_btn = ttk.Button(btn_frame, text="Live Запись",
                                   command=self._toggle_live)
        self.live_btn.pack(side="left", padx=5)

        self.live_status = ttk.Label(btn_frame, text="")
        self.live_status.pack(side="left", padx=10)

        # Live transcription text
        trans_frame = ttk.LabelFrame(frame, text="Транскрипция (real-time)")
        trans_frame.grid(row=7, column=0, columnspan=3, sticky="nsew", pady=(10, 0))

        self.live_text = tk.Text(trans_frame, height=12, state="disabled",
                                 wrap="word", font=("Menlo", 11))
        scrollbar = ttk.Scrollbar(trans_frame, command=self.live_text.yview)
        self.live_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.live_text.pack(fill="both", expand=True)

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(7, weight=1)

        self._live_recording = False
        self._live_stop_event = None
        self._live_text_queue: queue.Queue[str] = queue.Queue()

    def _toggle_live(self):
        if not self._live_recording:
            self._start_live()
        else:
            self._stop_live()

    def _start_live(self):
        if self._running_task and self._running_task.is_alive():
            messagebox.showwarning("audio2text", "Задача уже выполняется, дождитесь завершения.")
            return

        import threading as _threading

        self._live_recording = True
        self._live_stop_event = _threading.Event()
        self.live_btn.configure(text="Остановить")
        self.live_status.configure(text="Загрузка модели...")

        # Очищаем окно транскрипции
        self.live_text.configure(state="normal")
        self.live_text.delete("1.0", "end")
        self.live_text.configure(state="disabled")

        devs, tags = self._collect_devices(
            self.live_mic_enabled, self.live_mic_var,
            self.live_sys_enabled, self.live_sys_var,
        )
        try:
            chunk = int(self.live_chunk_var.get())
        except ValueError:
            chunk = 30
        self.config.setdefault("live", {})["chunk_seconds"] = chunk

        def on_chunk(text):
            self._live_text_queue.put(text)

        def vu_cb(idx, rms):
            val = min(rms * 10, 1.0)
            if idx < len(tags):
                if tags[idx] == "mic":
                    self.live_mic_vu.set(val)
                else:
                    self.live_sys_vu.set(val)

        def do_live():
            try:
                from processor import record_live
                self.root.after(0, lambda: self.live_status.configure(
                    text="Запись + транскрибация..."))
                path = record_live(
                    self.config,
                    devices=devs,
                    stop_event=self._live_stop_event,
                    on_chunk=on_chunk,
                    vu_callback=vu_cb,
                )
                if path and path.exists():
                    self.log_queue.put(f"Live транскрипция: {path}")
            except Exception as e:
                self.log_queue.put(f"Ошибка live: {e}")
            finally:
                self.root.after(0, self._on_live_done)

        self._running_task = _threading.Thread(target=do_live, daemon=True)
        self._running_task.start()
        self._poll_live_text()

    def _stop_live(self):
        if self._live_stop_event:
            self._live_stop_event.set()
        self.live_status.configure(text="Остановка, транскрибация остатка...")

    def _test_live_devices(self):
        self._test_devices(
            self.live_mic_enabled, self.live_mic_var, self.live_mic_vu,
            self.live_sys_enabled, self.live_sys_var, self.live_sys_vu,
        )

    def _on_live_done(self):
        self._poll_live_text()
        self._live_recording = False
        self.live_btn.configure(text="Live Запись")
        self.live_status.configure(text="")
        self.live_mic_vu.set(0)
        self.live_sys_vu.set(0)

    def _poll_live_text(self):
        while True:
            try:
                text = self._live_text_queue.get_nowait()
                self.live_text.configure(state="normal")
                self.live_text.insert("end", text)
                self.live_text.see("end")
                self.live_text.configure(state="disabled")
            except queue.Empty:
                break
        if self._live_recording:
            self.root.after(200, self._poll_live_text)

    # ── Transcribe tab ─────────────────────────────────────────────────

    def _build_transcribe_tab(self, notebook: ttk.Notebook):
        frame = ttk.Frame(notebook, padding=15)
        notebook.add(frame, text="  Транскрибация  ")

        # File path
        ttk.Label(frame, text="Файл / папка:").grid(row=0, column=0, sticky="w", pady=5)
        self.trans_path_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.trans_path_var).grid(
            row=0, column=1, sticky="ew", padx=(10, 0), pady=5)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=0, column=2, padx=(5, 0), pady=5)
        ttk.Button(btn_frame, text="Файл", width=6,
                   command=lambda: self._pick_audio_file(self.trans_path_var)).pack(side="left", padx=1)
        ttk.Button(btn_frame, text="Папка", width=6,
                   command=lambda: self._pick_dir(self.trans_path_var)).pack(side="left", padx=1)

        # Options
        ttk.Label(frame, text="Язык:").grid(row=1, column=0, sticky="w", pady=5)
        self.trans_lang_var = tk.StringVar(
            value=self.config.get("transcription", {}).get("language", "ru"))
        ttk.Combobox(frame, textvariable=self.trans_lang_var, width=10,
                     values=["ru", "en", "auto", "de", "fr", "es", "zh", "ja"]).grid(
            row=1, column=1, sticky="w", padx=(10, 0), pady=5)

        ttk.Label(frame, text="Backend:").grid(row=2, column=0, sticky="w", pady=5)
        self.trans_backend_var = tk.StringVar(
            value=self.config.get("transcription", {}).get("backend", "auto"))
        ttk.Combobox(frame, textvariable=self.trans_backend_var, width=20,
                     state="readonly",
                     values=["auto", "mlx", "faster-whisper"]).grid(
            row=2, column=1, sticky="w", padx=(10, 0), pady=5)

        # Run
        ttk.Button(frame, text="Транскрибировать",
                   command=self._run_transcribe).grid(
            row=3, column=0, columnspan=3, pady=20)

        frame.columnconfigure(1, weight=1)

    def _run_transcribe(self):
        path = self.trans_path_var.get().strip()
        if not path:
            messagebox.showwarning("audio2text", "Выберите файл или папку.")
            return
        self._apply_transcription_overrides()
        self._run_in_thread(self._do_transcribe, path)

    def _do_transcribe(self, path: str):
        from processor import transcribe_file, SUPPORTED_AUDIO
        p = Path(path)
        if p.is_file():
            transcribe_file(str(p), self.config)
        elif p.is_dir():
            files = sorted(f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_AUDIO)
            if not files:
                self.log_queue.put(f"Нет аудиофайлов в {p}")
                return
            for f in files:
                transcribe_file(str(f), self.config)
        self.log_queue.put("Транскрибация завершена.")

    # ── Diarize tab ────────────────────────────────────────────────────

    def _build_diarize_tab(self, notebook: ttk.Notebook):
        frame = ttk.Frame(notebook, padding=15)
        notebook.add(frame, text="  Диаризация  ")

        # File path
        ttk.Label(frame, text="Файл / папка:").grid(row=0, column=0, sticky="w", pady=5)
        self.diar_path_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.diar_path_var).grid(
            row=0, column=1, sticky="ew", padx=(10, 0), pady=5)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=0, column=2, padx=(5, 0), pady=5)
        ttk.Button(btn_frame, text="Файл", width=6,
                   command=lambda: self._pick_audio_file(self.diar_path_var)).pack(side="left", padx=1)
        ttk.Button(btn_frame, text="Папка", width=6,
                   command=lambda: self._pick_dir(self.diar_path_var)).pack(side="left", padx=1)

        # Speakers
        ttk.Label(frame, text="Мин. спикеров:").grid(row=1, column=0, sticky="w", pady=5)
        self.diar_min_var = tk.StringVar(value="")
        ttk.Entry(frame, textvariable=self.diar_min_var, width=6).grid(
            row=1, column=1, sticky="w", padx=(10, 0), pady=5)

        ttk.Label(frame, text="Макс. спикеров:").grid(row=2, column=0, sticky="w", pady=5)
        self.diar_max_var = tk.StringVar(value="")
        ttk.Entry(frame, textvariable=self.diar_max_var, width=6).grid(
            row=2, column=1, sticky="w", padx=(10, 0), pady=5)

        # Run
        ttk.Button(frame, text="Диаризовать",
                   command=self._run_diarize).grid(
            row=3, column=0, columnspan=3, pady=20)

        frame.columnconfigure(1, weight=1)

    def _run_diarize(self):
        path = self.diar_path_var.get().strip()
        if not path:
            messagebox.showwarning("audio2text", "Выберите файл или папку.")
            return
        self._run_in_thread(self._do_diarize, path)

    def _do_diarize(self, path: str):
        from processor import diarize_file, transcribe_file, SUPPORTED_AUDIO

        cfg_d = self.config.setdefault("diarization", {})
        min_sp = self.diar_min_var.get().strip()
        max_sp = self.diar_max_var.get().strip()
        if min_sp.isdigit():
            cfg_d["min_speakers"] = int(min_sp)
        if max_sp.isdigit():
            cfg_d["max_speakers"] = int(max_sp)

        p = Path(path)
        files = [p] if p.is_file() else sorted(
            f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_AUDIO)

        for f in files:
            result = transcribe_file(str(f), self.config)
            diarize_file(str(f), self.config, transcription=result)

        self.log_queue.put("Диаризация завершена.")

    # ── Process tab ────────────────────────────────────────────────────

    def _build_process_tab(self, notebook: ttk.Notebook):
        frame = ttk.Frame(notebook, padding=15)
        notebook.add(frame, text="  Полный pipeline  ")

        # File path
        ttk.Label(frame, text="Файл / папка:").grid(row=0, column=0, sticky="w", pady=5)
        self.proc_path_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.proc_path_var).grid(
            row=0, column=1, sticky="ew", padx=(10, 0), pady=5)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=0, column=2, padx=(5, 0), pady=5)
        ttk.Button(btn_frame, text="Файл", width=6,
                   command=lambda: self._pick_audio_file(self.proc_path_var)).pack(side="left", padx=1)
        ttk.Button(btn_frame, text="Папка", width=6,
                   command=lambda: self._pick_dir(self.proc_path_var)).pack(side="left", padx=1)

        # Description
        desc = (
            "Полный pipeline: транскрибация + диаризация + суммаризация.\n"
            "Настройки берутся из вкладки «Настройки» и config.yaml."
        )
        ttk.Label(frame, text=desc, wraplength=500, justify="left",
                  foreground="gray").grid(row=1, column=0, columnspan=3,
                                          sticky="w", pady=(10, 0))

        # Run
        ttk.Button(frame, text="Запустить pipeline",
                   command=self._run_process).grid(
            row=2, column=0, columnspan=3, pady=20)

        frame.columnconfigure(1, weight=1)

    def _run_process(self):
        path = self.proc_path_var.get().strip()
        if not path:
            messagebox.showwarning("audio2text", "Выберите файл или папку.")
            return
        self._apply_transcription_overrides()
        self._run_in_thread(self._do_process, path)

    def _do_process(self, path: str):
        from processor import process_file, process_directory
        p = Path(path)
        if p.is_file():
            process_file(str(p), self.config)
        elif p.is_dir():
            process_directory(str(p), self.config)
        self.log_queue.put("Pipeline завершён.")

    # ── Settings tab ───────────────────────────────────────────────────

    def _build_settings_tab(self, notebook: ttk.Notebook):
        frame = ttk.Frame(notebook, padding=15)
        notebook.add(frame, text="  Настройки  ")

        canvas = tk.Canvas(frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, padding=5)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        row = 0
        cfg_t = self.config.get("transcription", {})
        cfg_d = self.config.get("diarization", {})

        # ── Transcription ──
        ttk.Label(inner, text="Транскрибация",
                  font=("Helvetica", 13, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        row += 1

        ttk.Label(inner, text="Backend:").grid(row=row, column=0, sticky="w", pady=3)
        self.set_backend_var = tk.StringVar(value=cfg_t.get("backend", "auto"))
        ttk.Combobox(inner, textvariable=self.set_backend_var, width=20,
                     state="readonly",
                     values=["auto", "mlx", "faster-whisper"]).grid(
            row=row, column=1, sticky="w", padx=(10, 0), pady=3)
        row += 1

        ttk.Label(inner, text="MLX модель:").grid(row=row, column=0, sticky="w", pady=3)
        self.set_mlx_model_var = tk.StringVar(
            value=cfg_t.get("mlx_model", "mlx-community/whisper-large-v3-turbo"))
        ttk.Entry(inner, textvariable=self.set_mlx_model_var, width=45).grid(
            row=row, column=1, sticky="ew", padx=(10, 0), pady=3)
        row += 1

        ttk.Label(inner, text="FW модель:").grid(row=row, column=0, sticky="w", pady=3)
        self.set_fw_model_var = tk.StringVar(
            value=cfg_t.get("fw_model", "large-v3-turbo"))
        ttk.Entry(inner, textvariable=self.set_fw_model_var, width=30).grid(
            row=row, column=1, sticky="ew", padx=(10, 0), pady=3)
        row += 1

        ttk.Label(inner, text="Язык:").grid(row=row, column=0, sticky="w", pady=3)
        self.set_lang_var = tk.StringVar(value=cfg_t.get("language", "ru"))
        ttk.Combobox(inner, textvariable=self.set_lang_var, width=10,
                     values=["ru", "en", "auto", "de", "fr", "es", "zh", "ja"]).grid(
            row=row, column=1, sticky="w", padx=(10, 0), pady=3)
        row += 1

        ttk.Label(inner, text="Beam size:").grid(row=row, column=0, sticky="w", pady=3)
        self.set_beam_var = tk.StringVar(value=str(cfg_t.get("beam_size", 5)))
        ttk.Entry(inner, textvariable=self.set_beam_var, width=6).grid(
            row=row, column=1, sticky="w", padx=(10, 0), pady=3)
        row += 1

        # ── Diarization ──
        ttk.Separator(inner, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1
        ttk.Label(inner, text="Диаризация",
                  font=("Helvetica", 13, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        row += 1

        ttk.Label(inner, text="Включена:").grid(row=row, column=0, sticky="w", pady=3)
        self.set_diar_enabled = tk.BooleanVar(value=cfg_d.get("enabled", True))
        ttk.Checkbutton(inner, variable=self.set_diar_enabled).grid(
            row=row, column=1, sticky="w", padx=(10, 0), pady=3)
        row += 1

        ttk.Label(inner, text="HF Token:").grid(row=row, column=0, sticky="w", pady=3)
        self.set_hf_token_var = tk.StringVar(value=cfg_d.get("hf_token", ""))
        ttk.Entry(inner, textvariable=self.set_hf_token_var, width=45, show="*").grid(
            row=row, column=1, sticky="ew", padx=(10, 0), pady=3)
        row += 1

        # ── System info ──
        ttk.Separator(inner, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1

        ttk.Button(inner, text="Информация о системе",
                   command=self._show_info).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=5)
        row += 1

        ttk.Button(inner, text="Аудиоустройства",
                   command=self._show_devices).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=5)
        row += 1

        # Apply button
        ttk.Separator(inner, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1
        ttk.Button(inner, text="Применить настройки",
                   command=self._apply_settings).grid(
            row=row, column=0, columnspan=2, pady=5)

        inner.columnconfigure(1, weight=1)

    def _apply_settings(self):
        cfg_t = self.config.setdefault("transcription", {})
        cfg_t["backend"] = self.set_backend_var.get()
        cfg_t["mlx_model"] = self.set_mlx_model_var.get()
        cfg_t["fw_model"] = self.set_fw_model_var.get()
        cfg_t["language"] = self.set_lang_var.get()
        try:
            cfg_t["beam_size"] = int(self.set_beam_var.get())
        except ValueError:
            pass

        cfg_d = self.config.setdefault("diarization", {})
        cfg_d["enabled"] = self.set_diar_enabled.get()
        cfg_d["hf_token"] = self.set_hf_token_var.get()

        self._log("Настройки применены.")

    def _apply_transcription_overrides(self):
        """Применяет выбранные на вкладке транскрибации параметры в config."""
        cfg_t = self.config.setdefault("transcription", {})

        # From transcribe tab
        if hasattr(self, "trans_lang_var"):
            lang = self.trans_lang_var.get()
            if lang:
                cfg_t["language"] = lang
        if hasattr(self, "trans_backend_var"):
            be = self.trans_backend_var.get()
            if be:
                cfg_t["backend"] = be

        # From settings tab
        if hasattr(self, "set_backend_var"):
            cfg_t["backend"] = self.set_backend_var.get()
            cfg_t["mlx_model"] = self.set_mlx_model_var.get()
            cfg_t["fw_model"] = self.set_fw_model_var.get()
            cfg_t["language"] = self.set_lang_var.get()
            try:
                cfg_t["beam_size"] = int(self.set_beam_var.get())
            except ValueError:
                pass

    def _show_info(self):
        from processor import show_info
        self._log(show_info(self.config))

    def _show_devices(self):
        try:
            import sounddevice as sd
            self._log(str(sd.query_devices()))
        except Exception as e:
            self._log(f"Ошибка: {e}")

    # ── Helpers ─────────────────────────────────────────────────────────

    def _pick_audio_file(self, var: tk.StringVar):
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_AUDIO))
        path = filedialog.askopenfilename(
            title="Выберите аудиофайл",
            filetypes=[("Аудио", exts), ("Все файлы", "*.*")],
        )
        if path:
            var.set(path)

    def _pick_dir(self, var: tk.StringVar):
        path = filedialog.askdirectory(title="Выберите папку с аудио")
        if path:
            var.set(path)

    def _log(self, text: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _run_in_thread(self, target, *args):
        if self._running_task and self._running_task.is_alive():
            messagebox.showwarning("audio2text", "Задача уже выполняется, дождитесь завершения.")
            return

        def wrapper():
            try:
                target(*args)
            except Exception as e:
                self.log_queue.put(f"Ошибка: {e}")

        self._running_task = threading.Thread(target=wrapper, daemon=True)
        self._running_task.start()

    def run(self):
        self.root.mainloop()


# ── Entry point ────────────────────────────────────────────────────────────


def main():
    app = Audio2TextApp()
    app.run()


if __name__ == "__main__":
    main()
