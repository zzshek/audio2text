"""audio2text — macOS GUI (tkinter/ttk)."""

from __future__ import annotations

import logging
import platform
import queue
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

from processor import load_config, SUPPORTED_AUDIO


# ── Logging handler → GUI ────────────────────────────────────────────────────


class QueueHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


# ── App ──────────────────────────────────────────────────────────────────────


class Audio2TextApp:
    """Главное окно приложения."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("audio2text")
        self.root.geometry("800x620")
        self.root.minsize(780, 520)

        # macOS native feel
        self.root.option_add("*tearOff", False)
        try:
            self.root.tk.call("tk::unsupported::MacWindowStyle", "style",
                              self.root._w, "document", "closeBox collapseBox")
        except tk.TclError:
            pass

        # App icon
        icon_path = Path(__file__).parent / "icon.png"
        if icon_path.exists():
            try:
                icon = tk.PhotoImage(file=str(icon_path))
                self.root.iconphoto(True, icon)
                self._icon = icon  # prevent GC
            except Exception:
                pass

        self.config = load_config("config.yaml")
        self.log_queue: queue.Queue[str] = queue.Queue()
        self._running_task: threading.Thread | None = None

        self._tab_logs: list[tk.Text] = []
        self._setup_logging()
        self._build_ui()
        self._poll_log_queue()

    # ── Logging ───────────────────────────────────────────────────────

    def _setup_logging(self):
        logger = logging.getLogger("audio2text")
        handler = QueueHandler(self.log_queue)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                              datefmt="%H:%M:%S"))
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
        # Добавляем padding вокруг текста вкладок
        style = ttk.Style()
        style.configure("TNotebook.Tab", padding=[10, 4])

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=8, pady=(6, 0))

        self._build_record_tab(notebook)
        self._build_live_tab(notebook)
        self._build_transcribe_tab(notebook)
        self._build_diarize_tab(notebook)
        self._build_summarize_tab(notebook)
        self._build_process_tab(notebook)
        self._build_monitor_tab(notebook)
        self._build_settings_tab(notebook)

        # Глобальный лог (для старых сообщений)
        self.log_text = tk.Text(
            self.root, height=1, state="disabled", wrap="word",
            font=("Menlo", 10))
        # Скрыт — каждая вкладка имеет свой лог

    def _make_log_widget(self, parent, row=None, col_span=3) -> tk.Text:
        """Создаёт лог-виджет внутри вкладки. Авто-определяет grid vs pack."""
        log_frame = ttk.LabelFrame(parent, text="Лог")
        if row is not None:
            log_frame.grid(row=row, column=0, columnspan=col_span,
                           sticky="nsew", pady=(6, 0))
        else:
            # Определяем следующую свободную строку grid
            try:
                max_row = max(
                    (info["row"] for child in parent.winfo_children()
                     if (info := child.grid_info())),
                    default=-1)
                log_frame.grid(row=max_row + 1, column=0,
                               columnspan=col_span, sticky="nsew",
                               pady=(6, 0))
                parent.rowconfigure(max_row + 1, weight=1)
            except Exception:
                log_frame.pack(fill="both", expand=True,
                               padx=0, pady=(6, 0))
        # Кнопка копирования в правом верхнем углу
        top_bar = ttk.Frame(log_frame)
        top_bar.pack(fill="x", side="top")
        copy_btn = ttk.Button(top_bar, text="⧉", width=2,
                              command=lambda: self._copy_log(log))
        copy_btn.pack(side="right", padx=2, pady=1)

        log = tk.Text(log_frame, height=6, state="normal",
                      wrap="word", font=("Menlo", 10))
        sb = ttk.Scrollbar(log_frame, command=log.yview)
        log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        log.pack(fill="both", expand=True)
        # Доступен для выделения и копирования, но не для ввода
        log.bind("<Key>", lambda e: "break"
                 if e.keysym not in ("c", "a") or not (e.state & 0x4)
                 else None)
        return log

    # ── Record tab ─────────────────────────────────────────────────────

    def _build_record_tab(self, notebook: ttk.Notebook):
        frame = ttk.Frame(notebook, padding=12)
        notebook.add(frame, text="Запись")

        # ── Правая часть: устройства ──
        dev_frame = ttk.Frame(frame)
        dev_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        # Микрофон
        self.rec_mic_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(dev_frame, text="Микрофон:",
                        variable=self.rec_mic_enabled,
                        command=lambda: self._toggle_combo(
                            self.rec_mic_enabled, self.rec_mic_combo)
                        ).grid(row=0, column=0, sticky="w")
        self.rec_mic_var = tk.StringVar(value="По умолчанию")
        self.rec_mic_combo = ttk.Combobox(
            dev_frame, textvariable=self.rec_mic_var,
            state="readonly", width=25)
        self.rec_mic_combo.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        self.rec_mic_vu = tk.DoubleVar(value=0)
        ttk.Progressbar(dev_frame, variable=self.rec_mic_vu,
                        maximum=1.0, mode="determinate").grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=(2, 6))

        # Системный звук
        self.rec_sys_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(dev_frame, text="Системный звук:",
                        variable=self.rec_sys_enabled,
                        command=lambda: self._toggle_combo(
                            self.rec_sys_enabled, self.rec_sys_combo)
                        ).grid(row=2, column=0, sticky="w")
        self.rec_sys_var = tk.StringVar(value="Не выбрано")
        self.rec_sys_combo = ttk.Combobox(
            dev_frame, textvariable=self.rec_sys_var,
            state="readonly", width=25)
        self.rec_sys_combo.grid(row=2, column=1, sticky="ew", padx=(5, 0))

        ttk.Button(dev_frame, text="↻", width=3,
                   command=self._refresh_all_devices).grid(
            row=2, column=2, padx=(4, 0))

        self.rec_sys_vu = tk.DoubleVar(value=0)
        ttk.Progressbar(dev_frame, variable=self.rec_sys_vu,
                        maximum=1.0, mode="determinate").grid(
            row=3, column=0, columnspan=3, sticky="ew", pady=(2, 6))

        # Hint
        self.rec_hint_var = tk.StringVar(value="")
        ttk.Label(dev_frame, textvariable=self.rec_hint_var,
                  wraplength=350, foreground="gray",
                  font=("Helvetica", 10)).grid(
            row=4, column=0, columnspan=3, sticky="w")

        # Название
        ttk.Label(dev_frame, text="Название:").grid(
            row=5, column=0, sticky="w", pady=(6, 0))
        self.rec_name_var = tk.StringVar(value="")
        ttk.Entry(dev_frame, textvariable=self.rec_name_var).grid(
            row=5, column=1, columnspan=2, sticky="ew",
            padx=(5, 0), pady=(6, 0))

        dev_frame.columnconfigure(1, weight=1)

        self._refresh_all_devices()

        # ── Левая часть: кнопки в колонку ──
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=0, column=0, sticky="nw")

        ttk.Button(btn_frame, text="Проверка звука", width=16,
                   command=self._test_record_devices).pack(
            pady=(0, 4), anchor="w")

        rec_row = ttk.Frame(btn_frame)
        rec_row.pack(pady=(0, 4), anchor="w")
        self._rec_dot_canvas = tk.Canvas(
            rec_row, width=12, height=12,
            bg=self.root.cget("bg"), highlightthickness=0)
        self._rec_dot = self._rec_dot_canvas.create_oval(
            1, 1, 11, 11, fill="red", outline="", state="hidden")
        self._rec_dot_canvas.pack(side="left", padx=(0, 3))
        self.record_btn = ttk.Button(
            rec_row, text="Начать запись", width=14,
            command=self._toggle_record)
        self.record_btn.pack(side="left")

        ttk.Button(btn_frame, text="Открыть папку", width=16,
                   command=self._open_recordings_folder).pack(
            pady=(0, 4), anchor="w")

        ttk.Button(btn_frame, text="Pipeline", width=16,
                   command=self._run_record_pipeline).pack(
            pady=(0, 4), anchor="w")

        self.record_status = ttk.Label(
            btn_frame, text="", foreground="gray",
            font=("Helvetica", 10))
        self.record_status.pack(anchor="w", pady=(4, 0))

        frame.columnconfigure(1, weight=1)
        self._tab_logs.append(self._make_log_widget(frame))
        self._recorder = None
        self._recording = False

    def _test_record_devices(self):
        self._test_devices(
            self.rec_mic_enabled, self.rec_mic_var, self.rec_mic_vu,
            self.rec_sys_enabled, self.rec_sys_var, self.rec_sys_vu)

    # ── Device management ─────────────────────────────────────────────

    _VIRTUAL_PATTERNS = [
        "blackhole", "soundflower", "loopback audio",
        "virtual cable", "vb-cable"]

    def _get_input_device_list(self) -> list[tuple[int, str]]:
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

        for mic_c, sys_c in [
            (getattr(self, "rec_mic_combo", None),
             getattr(self, "rec_sys_combo", None)),
            (getattr(self, "live_mic_combo", None),
             getattr(self, "live_sys_combo", None)),
        ]:
            if mic_c:
                mic_c["values"] = mic_values
            if sys_c:
                sys_c["values"] = sys_values

        virtual_found = None
        for dev_id, name in devs:
            if self._is_virtual_device(name):
                virtual_found = f"{dev_id}: {name}"
                break

        for sys_var, hint_var in [
            (getattr(self, "rec_sys_var", None),
             getattr(self, "rec_hint_var", None)),
            (getattr(self, "live_sys_var", None),
             getattr(self, "live_hint_var", None)),
        ]:
            if sys_var and sys_var.get() == "Не выбрано" and virtual_found:
                sys_var.set(virtual_found)
            if hint_var:
                hint_var.set("")

    @staticmethod
    def _parse_device_id(selection: str) -> int | None:
        if selection in ("По умолчанию", "Не выбрано", ""):
            return None
        try:
            return int(selection.split(":")[0])
        except (ValueError, IndexError):
            return None

    def _collect_devices(self, mic_enabled, mic_var, sys_enabled, sys_var):
        devs, tags = [], []
        if mic_enabled.get():
            devs.append(self._parse_device_id(mic_var.get()))
            tags.append("mic")
        if sys_enabled.get():
            dev = self._parse_device_id(sys_var.get())
            if dev is not None:
                devs.append(dev)
                tags.append("sys")
        return (devs, tags) if devs else ([None], ["mic"])

    def _test_devices(self, mic_enabled, mic_var, mic_vu,
                      sys_enabled, sys_var, sys_vu):
        if self._running_task and self._running_task.is_alive():
            messagebox.showwarning("audio2text", "Задача уже выполняется.")
            return

        import numpy as np
        import sounddevice as sd

        devs, tags = self._collect_devices(
            mic_enabled, mic_var, sys_enabled, sys_var)
        sample_rate = self.config.get("recording", {}).get("sample_rate", 16000)
        max_rms: dict[str, float] = {"mic": 0.0, "sys": 0.0}

        def make_cb(idx):
            tag = tags[idx]
            def cb(indata, frames, time_info, status):
                rms = float(np.sqrt(np.mean(indata ** 2)))
                val = min(rms * 10, 1.0)
                (mic_vu if tag == "mic" else sys_vu).set(val)
                max_rms[tag] = max(max_rms[tag], rms)
            return cb

        streams = []
        for i, dev in enumerate(devs):
            try:
                st = sd.InputStream(
                    samplerate=sample_rate, channels=1, dtype="float32",
                    device=dev, callback=make_cb(i),
                    blocksize=int(sample_rate * 0.5))
                st.start()
                streams.append(st)
            except Exception as e:
                self._log(f"Ошибка устройства {dev}: {e}")
        if not streams:
            return

        self._log("Проверка звука... (3 секунды)")

        def stop_test():
            for st in streams:
                st.stop(); st.close()
            mic_vu.set(0); sys_vu.set(0)
            lines = []
            for tag in tags:
                nm = "Микрофон" if tag == "mic" else "Системный звук"
                r = max_rms[tag]
                st = ("НЕТ СИГНАЛА" if r < 0.001
                      else "очень тихо" if r < 0.005
                      else f"OK ({r:.4f})")
                lines.append(f"  {nm}: {st}")
            self._log("Результат:\n" + "\n".join(lines))

        self.root.after(3000, stop_test)

    # ── Record controls ───────────────────────────────────────────────

    def _toggle_record(self):
        if not self._recording:
            self._start_record()
        else:
            self._stop_record()

    def _start_record(self):
        if self._running_task and self._running_task.is_alive():
            messagebox.showwarning("audio2text",
                                   "Задача уже выполняется.")
            return

        from recorder import Recorder
        self._recorder = Recorder(self.config)
        self._recording = True
        self.record_btn.configure(text="Остановить")
        self.record_status.configure(text="Идёт запись...")
        self._do_blink(self._rec_dot_canvas, self._rec_dot, "_recording")

        devs, tags = self._collect_devices(
            self.rec_mic_enabled, self.rec_mic_var,
            self.rec_sys_enabled, self.rec_sys_var)
        custom_name = self.rec_name_var.get().strip()

        def vu_cb(idx, rms):
            val = min(rms * 10, 1.0)
            if idx < len(tags):
                (self.rec_mic_vu if tags[idx] == "mic"
                 else self.rec_sys_vu).set(val)

        def do_record():
            try:
                path = self._recorder.record(
                    devices=devs, vu_callback=vu_cb,
                    custom_name=custom_name)
                if path and path.exists():
                    self.log_queue.put(f"Файл сохранён: {path}")
            except Exception as e:
                self.log_queue.put(f"Ошибка записи: {e}")
            finally:
                self.root.after(0, self._on_record_done)

        self._running_task = threading.Thread(
            target=do_record, daemon=True)
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

    def _run_record_pipeline(self):
        """Запуск pipeline для последнего записанного файла."""
        output_dir = Path(
            self.config.get("recording", {}).get("output_dir", "recordings"))
        if not output_dir.exists():
            messagebox.showwarning("audio2text", "Папка recordings не найдена.")
            return
        # Находим последний аудиофайл
        latest = None
        for day_dir in sorted(output_dir.iterdir(), reverse=True):
            if not day_dir.is_dir():
                continue
            for f in sorted(day_dir.iterdir(), reverse=True):
                if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO:
                    latest = f
                    break
            if latest:
                break
        if not latest:
            messagebox.showwarning("audio2text", "Нет аудиофайлов.")
            return
        self._log(f"Pipeline: {latest.name}")
        self._apply_transcription_overrides()
        self._run_in_thread(self._do_record_pipeline, str(latest))

    def _do_record_pipeline(self, path: str):
        from processor import process_file
        process_file(path, self.config)
        self.log_queue.put("Pipeline завершён.")

    # ── Live tab ───────────────────────────────────────────────────────

    def _build_live_tab(self, notebook: ttk.Notebook):
        frame = ttk.Frame(notebook, padding=12)
        notebook.add(frame, text="Live")

        # ── Правая часть: устройства ──
        dev_frame = ttk.Frame(frame)
        dev_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        # Микрофон
        self.live_mic_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(dev_frame, text="Микрофон:",
                        variable=self.live_mic_enabled,
                        command=lambda: self._toggle_combo(
                            self.live_mic_enabled, self.live_mic_combo)
                        ).grid(row=0, column=0, sticky="w")
        self.live_mic_var = tk.StringVar(value="По умолчанию")
        self.live_mic_combo = ttk.Combobox(
            dev_frame, textvariable=self.live_mic_var,
            state="readonly", width=25)
        self.live_mic_combo.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        self.live_mic_vu = tk.DoubleVar(value=0)
        ttk.Progressbar(dev_frame, variable=self.live_mic_vu,
                        maximum=1.0, mode="determinate").grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=(2, 6))

        # Системный звук
        self.live_sys_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(dev_frame, text="Системный звук:",
                        variable=self.live_sys_enabled,
                        command=lambda: self._toggle_combo(
                            self.live_sys_enabled, self.live_sys_combo)
                        ).grid(row=2, column=0, sticky="w")
        self.live_sys_var = tk.StringVar(value="Не выбрано")
        self.live_sys_combo = ttk.Combobox(
            dev_frame, textvariable=self.live_sys_var,
            state="readonly", width=25)
        self.live_sys_combo.grid(row=2, column=1, sticky="ew", padx=(5, 0))

        ttk.Button(dev_frame, text="↻", width=3,
                   command=self._refresh_all_devices).grid(
            row=2, column=2, padx=(4, 0))

        self.live_sys_vu = tk.DoubleVar(value=0)
        ttk.Progressbar(dev_frame, variable=self.live_sys_vu,
                        maximum=1.0, mode="determinate").grid(
            row=3, column=0, columnspan=3, sticky="ew", pady=(2, 6))

        # Hint
        self.live_hint_var = tk.StringVar(value="")
        ttk.Label(dev_frame, textvariable=self.live_hint_var,
                  wraplength=350, foreground="gray",
                  font=("Helvetica", 10)).grid(
            row=4, column=0, columnspan=3, sticky="w")

        # Чанк + Название
        ttk.Label(dev_frame, text="Чанк (сек):").grid(
            row=5, column=0, sticky="w", pady=(4, 0))
        self.live_chunk_var = tk.StringVar(value="30")
        ttk.Spinbox(dev_frame, textvariable=self.live_chunk_var,
                    from_=10, to=120, increment=5, width=5).grid(
            row=5, column=1, sticky="w", padx=(5, 0), pady=(4, 0))

        ttk.Label(dev_frame, text="Название:").grid(
            row=6, column=0, sticky="w", pady=(4, 0))
        self.live_name_var = tk.StringVar(value="")
        ttk.Entry(dev_frame, textvariable=self.live_name_var).grid(
            row=6, column=1, columnspan=2, sticky="ew",
            padx=(5, 0), pady=(4, 0))

        dev_frame.columnconfigure(1, weight=1)

        # Заполняем списки
        devs = self._get_input_device_list()
        items = [f"{i}: {name}" for i, name in devs]
        self.live_mic_combo["values"] = ["По умолчанию"] + items
        self.live_sys_combo["values"] = ["Не выбрано"] + items
        for dev_id, name in devs:
            if self._is_virtual_device(name):
                self.live_sys_var.set(f"{dev_id}: {name}")
                break
        else:
            pass

        # ── Левая часть: кнопки в колонку ──
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=0, column=0, sticky="nw")

        ttk.Button(btn_frame, text="Проверка звука", width=16,
                   command=self._test_live_devices).pack(
            pady=(0, 4), anchor="w")

        rec_row = ttk.Frame(btn_frame)
        rec_row.pack(pady=(0, 4), anchor="w")
        self._live_dot_canvas = tk.Canvas(
            rec_row, width=12, height=12,
            bg=self.root.cget("bg"), highlightthickness=0)
        self._live_dot = self._live_dot_canvas.create_oval(
            1, 1, 11, 11, fill="red", outline="", state="hidden")
        self._live_dot_canvas.pack(side="left", padx=(0, 3))
        self.live_btn = ttk.Button(
            rec_row, text="Live Запись", width=14,
            command=self._toggle_live)
        self.live_btn.pack(side="left")

        ttk.Button(btn_frame, text="Открыть папку", width=16,
                   command=self._open_recordings_folder).pack(
            pady=(0, 4), anchor="w")

        self.live_status = ttk.Label(
            btn_frame, text="", foreground="gray",
            font=("Helvetica", 10))
        self.live_status.pack(anchor="w", pady=(4, 0))

        # Live transcription (glass panel)
        trans_frame = ttk.LabelFrame(
            frame, text="Транскрипция (real-time)")
        trans_frame.grid(
            row=8, column=0, columnspan=3,
            sticky="nsew", pady=(10, 0))

        self.live_text = tk.Text(
            trans_frame, height=10, state="disabled", wrap="word",
            font=("Menlo", 11))
        scrollbar = ttk.Scrollbar(
            trans_frame, command=self.live_text.yview)
        self.live_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.live_text.pack(fill="both", expand=True)

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(8, weight=1)
        self._tab_logs.append(self._make_log_widget(frame))

        self._live_recording = False
        self._live_stop_event = None
        self._live_text_queue: queue.Queue[str] = queue.Queue()

    # ── Live controls ─────────────────────────────────────────────────

    def _toggle_live(self):
        if not self._live_recording:
            self._start_live()
        else:
            self._stop_live()

    def _start_live(self):
        if self._running_task and self._running_task.is_alive():
            messagebox.showwarning("audio2text",
                                   "Задача уже выполняется.")
            return

        import threading as _threading
        self._live_recording = True
        self._live_stop_event = _threading.Event()
        self.live_btn.configure(text="Остановить")
        self.live_status.configure(text="Загрузка модели...")
        self._do_blink(self._live_dot_canvas, self._live_dot,
                       "_live_recording")

        self.live_text.configure(state="normal")
        self.live_text.delete("1.0", "end")
        self.live_text.configure(state="disabled")

        devs, tags = self._collect_devices(
            self.live_mic_enabled, self.live_mic_var,
            self.live_sys_enabled, self.live_sys_var)
        try:
            chunk = int(self.live_chunk_var.get())
        except ValueError:
            chunk = 30
        self.config.setdefault("live", {})["chunk_seconds"] = chunk
        custom_name = self.live_name_var.get().strip()

        def on_chunk(text):
            self._live_text_queue.put(text)

        def vu_cb(idx, rms):
            val = min(rms * 10, 1.0)
            if idx < len(tags):
                (self.live_mic_vu if tags[idx] == "mic"
                 else self.live_sys_vu).set(val)

        def do_live():
            try:
                from processor import record_live
                self.root.after(0, lambda: self.live_status.configure(
                    text="Запись + транскрибация..."))
                path = record_live(
                    self.config, devices=devs,
                    stop_event=self._live_stop_event,
                    on_chunk=on_chunk, vu_callback=vu_cb,
                    custom_name=custom_name)
                if path and path.exists():
                    self.log_queue.put(f"Live транскрипция: {path}")
            except Exception as e:
                self.log_queue.put(f"Ошибка live: {e}")
            finally:
                self.root.after(0, self._on_live_done)

        self._running_task = _threading.Thread(
            target=do_live, daemon=True)
        self._running_task.start()
        self._poll_live_text()

    def _stop_live(self):
        if self._live_stop_event:
            self._live_stop_event.set()
        self.live_status.configure(
            text="Остановка, транскрибация остатка...")

    def _test_live_devices(self):
        self._test_devices(
            self.live_mic_enabled, self.live_mic_var, self.live_mic_vu,
            self.live_sys_enabled, self.live_sys_var, self.live_sys_vu)

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
        frame = ttk.Frame(notebook, padding=18)
        notebook.add(frame, text="Транскр.")

        ttk.Label(frame, text="Файл / папка:").grid(
            row=0, column=0, sticky="w", pady=5)
        self.trans_path_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.trans_path_var).grid(
            row=0, column=1, sticky="ew", padx=(10, 0), pady=5)

        bf = ttk.Frame(frame)
        bf.grid(row=0, column=2, padx=(5, 0), pady=5)
        ttk.Button(bf, text="Файл",
                   command=lambda: self._pick_audio_file(
                       self.trans_path_var)).pack(side="left", padx=1)
        ttk.Button(bf, text="Папка",
                   command=lambda: self._pick_dir(
                       self.trans_path_var)).pack(side="left", padx=1)

        ttk.Label(frame, text="Язык:").grid(
            row=1, column=0, sticky="w", pady=5)
        self.trans_lang_var = tk.StringVar(
            value=self.config.get(
                "transcription", {}).get("language", "ru"))
        ttk.Combobox(
            frame, textvariable=self.trans_lang_var, width=10,
            values=["ru", "en", "auto", "de", "fr",
                    "es", "zh", "ja"]).grid(
            row=1, column=1, sticky="w", padx=(10, 0), pady=5)

        ttk.Label(frame, text="Backend:").grid(
            row=2, column=0, sticky="w", pady=5)
        self.trans_backend_var = tk.StringVar(
            value=self.config.get(
                "transcription", {}).get("backend", "auto"))
        ttk.Combobox(
            frame, textvariable=self.trans_backend_var, width=20,
            state="readonly",
            values=["auto", "mlx", "faster-whisper"]).grid(
            row=2, column=1, sticky="w", padx=(10, 0), pady=5)

        ttk.Button(frame, text="Транскрибировать",
                   command=self._run_transcribe).grid(
            row=3, column=0, columnspan=3, pady=10)

        frame.columnconfigure(1, weight=1)
        self._tab_logs.append(self._make_log_widget(frame))

    def _run_transcribe(self):
        path = self.trans_path_var.get().strip()
        if not path:
            messagebox.showwarning("audio2text",
                                   "Выберите файл или папку.")
            return
        self._apply_transcription_overrides()
        self._run_in_thread(self._do_transcribe, path)

    def _do_transcribe(self, path: str):
        from processor import transcribe_file, SUPPORTED_AUDIO
        p = Path(path)
        if p.is_file():
            transcribe_file(str(p), self.config)
        elif p.is_dir():
            files = sorted(
                f for f in p.iterdir()
                if f.suffix.lower() in SUPPORTED_AUDIO)
            if not files:
                self.log_queue.put(f"Нет аудиофайлов в {p}")
                return
            for f in files:
                transcribe_file(str(f), self.config)
        self.log_queue.put("Транскрибация завершена.")

    # ── Diarize tab ────────────────────────────────────────────────────

    def _build_diarize_tab(self, notebook: ttk.Notebook):
        frame = ttk.Frame(notebook, padding=18)
        notebook.add(frame, text="Диариз.")

        ttk.Label(frame, text="Файл / папка:").grid(
            row=0, column=0, sticky="w", pady=5)
        self.diar_path_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.diar_path_var).grid(
            row=0, column=1, sticky="ew", padx=(10, 0), pady=5)

        bf = ttk.Frame(frame)
        bf.grid(row=0, column=2, padx=(5, 0), pady=5)
        ttk.Button(bf, text="Файл",
                   command=lambda: self._pick_audio_file(
                       self.diar_path_var)).pack(side="left", padx=1)
        ttk.Button(bf, text="Папка",
                   command=lambda: self._pick_dir(
                       self.diar_path_var)).pack(side="left", padx=1)

        ttk.Label(frame, text="Мин. спикеров:").grid(
            row=1, column=0, sticky="w", pady=5)
        self.diar_min_var = tk.StringVar(value="")
        ttk.Entry(frame, textvariable=self.diar_min_var,
                  width=6).grid(
            row=1, column=1, sticky="w", padx=(10, 0), pady=5)

        ttk.Label(frame, text="Макс. спикеров:").grid(
            row=2, column=0, sticky="w", pady=5)
        self.diar_max_var = tk.StringVar(value="")
        ttk.Entry(frame, textvariable=self.diar_max_var,
                  width=6).grid(
            row=2, column=1, sticky="w", padx=(10, 0), pady=5)

        ttk.Button(frame, text="Диаризовать",
                   command=self._run_diarize).grid(
            row=3, column=0, columnspan=3, pady=10)

        frame.columnconfigure(1, weight=1)
        self._tab_logs.append(self._make_log_widget(frame))

    def _run_diarize(self):
        path = self.diar_path_var.get().strip()
        if not path:
            messagebox.showwarning("audio2text",
                                   "Выберите файл или папку.")
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
        files = ([p] if p.is_file() else sorted(
            f for f in p.iterdir()
            if f.suffix.lower() in SUPPORTED_AUDIO))
        for f in files:
            result = transcribe_file(str(f), self.config)
            diarize_file(str(f), self.config, transcription=result)
        self.log_queue.put("Диаризация завершена.")

    # ── Summarize tab ────────────────────────────────────────────────

    def _build_summarize_tab(self, notebook: ttk.Notebook):
        frame = ttk.Frame(notebook, padding=18)
        notebook.add(frame, text="Саммари")

        # Файл
        top = ttk.Frame(frame)
        top.grid(row=0, column=0, sticky="ew")

        ttk.Label(top, text="Файл:").pack(side="left")
        self.sum_path_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.sum_path_var).pack(
            side="left", fill="x", expand=True, padx=(8, 0))
        ttk.Button(top, text="Выбрать",
                   command=self._pick_text_file).pack(
            side="left", padx=(4, 0))

        # Превью файла
        ttk.Label(frame, text="Превью файла:").grid(
            row=1, column=0, sticky="w", pady=(8, 2))

        preview_frame = ttk.Frame(frame)
        preview_frame.grid(row=2, column=0, sticky="nsew")

        self.sum_input = tk.Text(
            preview_frame, height=6, wrap="word",
            font=("Menlo", 10))
        sb_in = ttk.Scrollbar(preview_frame, command=self.sum_input.yview)
        self.sum_input.configure(yscrollcommand=sb_in.set)
        sb_in.pack(side="right", fill="y")
        self.sum_input.pack(fill="both", expand=True)

        # Промпт (редактируемый)
        ttk.Label(frame, text="Промпт (можно редактировать):").grid(
            row=3, column=0, sticky="w", pady=(8, 2))

        prompt_frame = ttk.Frame(frame)
        prompt_frame.grid(row=4, column=0, sticky="nsew")

        self.sum_prompt = tk.Text(
            prompt_frame, height=8, wrap="word",
            font=("Menlo", 10))
        sb_prompt = ttk.Scrollbar(prompt_frame, command=self.sum_prompt.yview)
        self.sum_prompt.configure(yscrollcommand=sb_prompt.set)
        sb_prompt.pack(side="right", fill="y")
        self.sum_prompt.pack(fill="both", expand=True)

        # Заполняем дефолтным промптом
        default_prompt = (
            "Ниже транскрипция рабочей встречи. "
            "Напиши подробное структурированное резюме на русском языке "
            "в формате Markdown. Важно: раскрывай каждый пункт детально — "
            "что именно обсуждалось, какие аргументы приводились, "
            "о чём договорились. Указывай спикеров где возможно.\n\n"
            "Используй именно такую структуру:\n\n"
            "## Резюме\nОписание встречи: участники, основная тема, контекст (3-5 предложений).\n\n"
            "## Обсуждаемые темы\nДля каждой темы опиши:\n"
            "### Тема N: название\n"
            "- Суть вопроса\n"
            "- Позиции участников и ключевые аргументы\n"
            "- К чему пришли / что решили\n\n"
            "## Решения и договорённости\n"
            "- Конкретное решение — кто, что, когда\n\n"
            "## Задачи\n- [ ] задача (ответственный, срок если озвучен)\n\n"
            "## Открытые вопросы\n- вопрос, который остался без решения\n\n"
            "Транскрипция:\n"
        )
        self.sum_prompt.insert("1.0", default_prompt)

        # Кнопка
        ttk.Button(frame, text="Суммаризировать",
                   command=self._run_summarize).grid(
            row=5, column=0, pady=8)

        # Лог
        log = self._make_log_widget(frame, row=6)
        self._tab_logs.append(log)

        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)
        frame.rowconfigure(4, weight=1)
        frame.rowconfigure(6, weight=1)

    def _pick_text_file(self):
        path = filedialog.askopenfilename(
            title="Выберите текстовый файл",
            filetypes=[("Текст", "*.txt"), ("Все файлы", "*.*")])
        if path:
            self.sum_path_var.set(path)
            self._load_text_file()

    def _load_text_file(self):
        path = self.sum_path_var.get().strip()
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8")
            self.sum_input.delete("1.0", "end")
            self.sum_input.insert("1.0", text)
            self._log(f"Загружено: {Path(path).name} "
                      f"({len(text)} символов)")
        except Exception as e:
            self._log(f"Ошибка чтения: {e}")

    def _run_summarize(self):
        text = self.sum_input.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("audio2text",
                                   "Загрузите текстовый файл.")
            return

        prompt_template = self.sum_prompt.get("1.0", "end").strip()
        if not prompt_template:
            messagebox.showwarning("audio2text",
                                   "Промпт не может быть пустым.")
            return

        self._run_in_thread(self._do_summarize, text, prompt_template)

    def _do_summarize(self, text: str, prompt_template: str):
        cfg = self.config
        llm_cfg = cfg.get("llm", {})
        sum_cfg = cfg.get("summarization", {})

        try:
            if llm_cfg.get("enabled"):
                from summariser import LLMSummarizer, _clean_transcript
                s = LLMSummarizer(cfg)
                # Используем промпт из GUI вместо встроенного
                clean_text = _clean_transcript(text)
                full_prompt = prompt_template + "\n" + clean_text
                result = s._call_llm(full_prompt)
            elif sum_cfg.get("enabled"):
                from summariser import Summarizer
                s = Summarizer(cfg)
                result = s.summarize(text)
            else:
                self.log_queue.put(
                    "Суммаризация отключена. Включите LLM или "
                    "Summarization в Настройках / config.yaml")
                return

            # Сохранить в файл рядом с исходным
            saved = ""
            src = self.sum_path_var.get().strip()
            if src:
                p = Path(src)
                out = p.parent / f"{p.stem}_summary.txt"
                out.write_text(result, encoding="utf-8")
                saved = f" → {out}"

            self.log_queue.put(
                f"Суммаризация завершена "
                f"({len(result)} символов){saved}")

            # Экспорт в Obsidian
            obs_cfg = cfg.get("obsidian", {})
            if obs_cfg.get("enabled") and result and src:
                try:
                    from processor import export_obsidian_note
                    # Ищем аудиофайл рядом с текстовым
                    from processor import SUPPORTED_AUDIO
                    p = Path(src)
                    audio_path = None
                    for ext in SUPPORTED_AUDIO:
                        # Убираем суффиксы _diar, _timed из имени
                        stem = p.stem
                        for suffix in ("_diar", "_timed", "_summary"):
                            stem = stem.removesuffix(suffix)
                        candidate = p.parent / f"{stem}{ext}"
                        if candidate.exists():
                            audio_path = candidate
                            break
                    if audio_path:
                        md = export_obsidian_note(audio_path, result, cfg)
                        if md:
                            self.log_queue.put(f"Obsidian: {md}")
                    else:
                        self.log_queue.put(
                            "Obsidian: аудиофайл не найден рядом с текстом")
                except Exception as e:
                    self.log_queue.put(f"Ошибка Obsidian: {e}")

        except Exception as e:
            self.log_queue.put(f"Ошибка суммаризации: {e}")

    # ── Process tab ────────────────────────────────────────────────────

    def _build_process_tab(self, notebook: ttk.Notebook):
        frame = ttk.Frame(notebook, padding=18)
        notebook.add(frame, text="Pipeline")

        ttk.Label(frame, text="Файл / папка:").grid(
            row=0, column=0, sticky="w", pady=5)
        self.proc_path_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.proc_path_var).grid(
            row=0, column=1, sticky="ew", padx=(10, 0), pady=5)

        bf = ttk.Frame(frame)
        bf.grid(row=0, column=2, padx=(5, 0), pady=5)
        ttk.Button(bf, text="Файл",
                   command=lambda: self._pick_audio_file(
                       self.proc_path_var)).pack(side="left", padx=1)
        ttk.Button(bf, text="Папка",
                   command=lambda: self._pick_dir(
                       self.proc_path_var)).pack(side="left", padx=1)

        ttk.Label(
            frame,
            text="Полный pipeline: транскрибация + диаризация "
                 "+ суммаризация.\n"
                 "Настройки берутся из вкладки Настройки и config.yaml.",
            wraplength=500, justify="left",
            foreground="gray", font=("Helvetica", 10)).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(10, 0))

        btn_row = ttk.Frame(frame)
        btn_row.grid(row=2, column=0, columnspan=3, pady=15)

        ttk.Button(btn_row, text="Запустить pipeline",
                   command=self._run_process).pack(side="left", padx=5)
        ttk.Button(btn_row, text="Обработать новые",
                   command=self._run_process_new).pack(side="left", padx=5)

        frame.columnconfigure(1, weight=1)
        self._tab_logs.append(self._make_log_widget(frame))

    def _run_process(self):
        path = self.proc_path_var.get().strip()
        if not path:
            messagebox.showwarning("audio2text",
                                   "Выберите файл или папку.")
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

    def _run_process_new(self):
        self._apply_transcription_overrides()
        self._run_in_thread(self._do_process_new)

    def _do_process_new(self):
        from processor import process_new
        n = process_new(self.config)
        self.log_queue.put(
            f"Обработано {n} новых записей." if n
            else "Нет новых записей.")

    # ── Settings tab ───────────────────────────────────────────────────

    # ── Monitor tab ─────────────────────────────────────────────────────

    def _build_monitor_tab(self, notebook: ttk.Notebook):
        frame = ttk.Frame(notebook, padding=12)
        notebook.add(frame, text="Монитор")

        frame.columnconfigure(1, weight=1)

        # Заголовок
        ttk.Label(frame, text="Ресурсы процесса audio2text",
                  font=("", 13, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

        # CPU
        ttk.Label(frame, text="CPU:").grid(row=1, column=0, sticky="w", pady=3)
        self._mon_cpu_var = tk.StringVar(value="—")
        ttk.Label(frame, textvariable=self._mon_cpu_var).grid(
            row=1, column=1, sticky="w", padx=6)
        self._mon_cpu_bar = ttk.Progressbar(frame, length=300, maximum=100)
        self._mon_cpu_bar.grid(row=1, column=2, sticky="ew", padx=(0, 6))

        # Memory RSS
        ttk.Label(frame, text="RAM:").grid(row=2, column=0, sticky="w", pady=3)
        self._mon_mem_var = tk.StringVar(value="—")
        ttk.Label(frame, textvariable=self._mon_mem_var).grid(
            row=2, column=1, sticky="w", padx=6)
        self._mon_mem_bar = ttk.Progressbar(frame, length=300, maximum=100)
        self._mon_mem_bar.grid(row=2, column=2, sticky="ew", padx=(0, 6))

        # Threads
        ttk.Label(frame, text="Потоки:").grid(row=3, column=0, sticky="w", pady=3)
        self._mon_threads_var = tk.StringVar(value="—")
        ttk.Label(frame, textvariable=self._mon_threads_var).grid(
            row=3, column=1, sticky="w", padx=6)

        # Uptime
        ttk.Label(frame, text="Время работы:").grid(row=4, column=0, sticky="w", pady=3)
        self._mon_uptime_var = tk.StringVar(value="—")
        ttk.Label(frame, textvariable=self._mon_uptime_var).grid(
            row=4, column=1, sticky="w", padx=6)

        # System total
        ttk.Separator(frame).grid(
            row=5, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Label(frame, text="Система", font=("", 12, "bold")).grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(0, 6))

        ttk.Label(frame, text="CPU (система):").grid(row=7, column=0, sticky="w", pady=3)
        self._mon_sys_cpu_var = tk.StringVar(value="—")
        ttk.Label(frame, textvariable=self._mon_sys_cpu_var).grid(
            row=7, column=1, sticky="w", padx=6)
        self._mon_sys_cpu_bar = ttk.Progressbar(frame, length=300, maximum=100)
        self._mon_sys_cpu_bar.grid(row=7, column=2, sticky="ew", padx=(0, 6))

        ttk.Label(frame, text="RAM (система):").grid(row=8, column=0, sticky="w", pady=3)
        self._mon_sys_mem_var = tk.StringVar(value="—")
        ttk.Label(frame, textvariable=self._mon_sys_mem_var).grid(
            row=8, column=1, sticky="w", padx=6)
        self._mon_sys_mem_bar = ttk.Progressbar(frame, length=300, maximum=100)
        self._mon_sys_mem_bar.grid(row=8, column=2, sticky="ew", padx=(0, 6))

        # История (текстовый лог)
        frame.rowconfigure(10, weight=1)
        frame.columnconfigure(2, weight=1)
        self._mon_log = self._make_log_widget(frame, row=10, col_span=3)
        self._tab_logs.append(self._mon_log)

        # Запуск обновления
        self._mon_process = None
        self._monitor_update()

    def _monitor_update(self):
        """Обновляет метрики монитора каждые 2 секунды."""
        try:
            import psutil
            import os
            import time as _time

            if self._mon_process is None:
                self._mon_process = psutil.Process(os.getpid())
                # Первый вызов cpu_percent для инициализации
                self._mon_process.cpu_percent()

            proc = self._mon_process

            # Process CPU (% от всех ядер)
            cpu_pct = proc.cpu_percent()
            self._mon_cpu_var.set(f"{cpu_pct:.1f}%")
            self._mon_cpu_bar["value"] = min(cpu_pct, 100)

            # Process Memory
            mem_info = proc.memory_info()
            rss_mb = mem_info.rss / 1024 / 1024
            rss_gb = rss_mb / 1024
            total_mem = psutil.virtual_memory().total / 1024 / 1024
            mem_pct = (mem_info.rss / psutil.virtual_memory().total) * 100
            if rss_gb >= 1:
                self._mon_mem_var.set(f"{rss_gb:.1f} GB ({mem_pct:.1f}%)")
            else:
                self._mon_mem_var.set(f"{rss_mb:.0f} MB ({mem_pct:.1f}%)")
            self._mon_mem_bar["value"] = min(mem_pct, 100)

            # Threads
            self._mon_threads_var.set(str(proc.num_threads()))

            # Uptime
            create_time = proc.create_time()
            uptime_sec = _time.time() - create_time
            h = int(uptime_sec // 3600)
            m = int((uptime_sec % 3600) // 60)
            s = int(uptime_sec % 60)
            self._mon_uptime_var.set(f"{h:02d}:{m:02d}:{s:02d}")

            # System CPU
            sys_cpu = psutil.cpu_percent()
            self._mon_sys_cpu_var.set(f"{sys_cpu:.1f}%")
            self._mon_sys_cpu_bar["value"] = min(sys_cpu, 100)

            # System Memory
            vmem = psutil.virtual_memory()
            sys_mem_used_gb = vmem.used / 1024 / 1024 / 1024
            sys_mem_total_gb = vmem.total / 1024 / 1024 / 1024
            self._mon_sys_mem_var.set(
                f"{sys_mem_used_gb:.1f} / {sys_mem_total_gb:.1f} GB ({vmem.percent}%)")
            self._mon_sys_mem_bar["value"] = vmem.percent

        except ImportError:
            self._mon_cpu_var.set("psutil не установлен")
            self._mon_mem_var.set("pip install psutil")
        except Exception as e:
            self._mon_cpu_var.set(f"Ошибка: {e}")

        self.root.after(2000, self._monitor_update)

    def _build_settings_tab(self, notebook: ttk.Notebook):
        frame = ttk.Frame(notebook, padding=18)
        notebook.add(frame, text="Настройки")

        canvas = tk.Canvas(frame, highlightthickness=0, bg=self.root.cget("bg"))
        scrollbar = ttk.Scrollbar(
            frame, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, padding=5)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(
                       scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        row = 0
        cfg_t = self.config.get("transcription", {})
        cfg_d = self.config.get("diarization", {})
        cfg_s = self.config.get("summarization", {})

        # ── Transcription ──
        ttk.Label(inner, text="Транскрибация",
                  font=("Helvetica", 13, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        row += 1

        for label, var_name, default, widget_type, opts in [
            ("Backend:", "set_backend_var",
             cfg_t.get("backend", "auto"), "combo",
             {"values": ["auto", "mlx", "faster-whisper"],
              "width": 20, "state": "readonly"}),
            ("MLX модель:", "set_mlx_model_var",
             cfg_t.get("mlx_model",
                        "mlx-community/whisper-large-v3-turbo"),
             "entry", {"width": 45}),
            ("FW модель:", "set_fw_model_var",
             cfg_t.get("fw_model", "large-v3-turbo"),
             "entry", {"width": 30}),
            ("Язык:", "set_lang_var",
             cfg_t.get("language", "ru"), "combo",
             {"values": ["ru", "en", "auto", "de", "fr",
                         "es", "zh", "ja"], "width": 10}),
            ("Beam size:", "set_beam_var",
             str(cfg_t.get("beam_size", 5)),
             "entry", {"width": 6}),
        ]:
            ttk.Label(inner, text=label).grid(
                row=row, column=0, sticky="w", pady=3)
            v = tk.StringVar(value=default)
            setattr(self, var_name, v)
            cls = ttk.Combobox if widget_type == "combo" else ttk.Entry
            w = cls(inner, textvariable=v, **opts)
            w.grid(row=row, column=1, sticky=(
                "w" if opts.get("width", 50) < 20 else "ew"),
                   padx=(10, 0), pady=3)
            row += 1

        # ── Diarization ──
        ttk.Separator(inner, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1
        ttk.Label(inner, text="Диаризация",
                  font=("Helvetica", 13, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        row += 1

        ttk.Label(inner, text="Включена:").grid(
            row=row, column=0, sticky="w", pady=3)
        self.set_diar_enabled = tk.BooleanVar(
            value=cfg_d.get("enabled", True))
        ttk.Checkbutton(inner, variable=self.set_diar_enabled).grid(
            row=row, column=1, sticky="w", padx=(10, 0), pady=3)
        row += 1

        ttk.Label(inner, text="HF Token:").grid(
            row=row, column=0, sticky="w", pady=3)
        self.set_hf_token_var = tk.StringVar(
            value=cfg_d.get("hf_token", ""))
        ttk.Entry(inner, textvariable=self.set_hf_token_var,
                  width=45, show="*").grid(
            row=row, column=1, sticky="ew", padx=(10, 0), pady=3)
        row += 1

        # ── Summarization ──
        ttk.Separator(inner, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1
        ttk.Label(inner, text="Суммаризация",
                  font=("Helvetica", 13, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        row += 1

        ttk.Label(inner, text="Включена:").grid(
            row=row, column=0, sticky="w", pady=3)
        self.set_sum_enabled = tk.BooleanVar(
            value=cfg_s.get("enabled", False))
        ttk.Checkbutton(inner, variable=self.set_sum_enabled).grid(
            row=row, column=1, sticky="w", padx=(10, 0), pady=3)
        row += 1

        ttk.Label(inner, text="Контекст:").grid(
            row=row, column=0, sticky="w", pady=3)
        self.set_sum_context_var = tk.StringVar(
            value=cfg_s.get(
                "context",
                "IT-компания, сфера аналитики данных. "
                "Используй IT-терминологию."))
        ttk.Entry(inner, textvariable=self.set_sum_context_var,
                  width=50).grid(
            row=row, column=1, sticky="ew", padx=(10, 0), pady=3)
        row += 1

        ttk.Label(inner,
                  text="Контекст помогает модели использовать "
                       "правильную терминологию",
                  foreground="gray", font=("Helvetica", 10)).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        row += 1

        # ── Obsidian ──
        ttk.Separator(inner, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1
        ttk.Label(inner, text="Obsidian",
                  font=("Helvetica", 13, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 5))
        row += 1

        cfg_o = self.config.get("obsidian", {})

        ttk.Label(inner, text="Включён:").grid(
            row=row, column=0, sticky="w", pady=3)
        self.set_obs_enabled = tk.BooleanVar(
            value=cfg_o.get("enabled", False))
        ttk.Checkbutton(inner, variable=self.set_obs_enabled).grid(
            row=row, column=1, sticky="w", padx=(10, 0), pady=3)
        row += 1

        ttk.Label(inner, text="Vault:").grid(
            row=row, column=0, sticky="w", pady=3)
        vault_row = ttk.Frame(inner)
        vault_row.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=3)
        self.set_obs_vault_var = tk.StringVar(
            value=cfg_o.get("vault_path", ""))
        ttk.Entry(vault_row, textvariable=self.set_obs_vault_var).pack(
            side="left", fill="x", expand=True)
        ttk.Button(vault_row, text="...", width=3,
                   command=self._pick_obsidian_vault).pack(
            side="left", padx=(4, 0))
        row += 1

        ttk.Label(inner, text="Папка:").grid(
            row=row, column=0, sticky="w", pady=3)
        self.set_obs_folder_var = tk.StringVar(
            value=cfg_o.get("folder", "Meetings"))
        ttk.Entry(inner, textvariable=self.set_obs_folder_var,
                  width=20).grid(
            row=row, column=1, sticky="w", padx=(10, 0), pady=3)
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

        cfg_s = self.config.setdefault("summarization", {})
        cfg_s["enabled"] = self.set_sum_enabled.get()
        cfg_s["context"] = self.set_sum_context_var.get()

        cfg_o = self.config.setdefault("obsidian", {})
        cfg_o["enabled"] = self.set_obs_enabled.get()
        cfg_o["vault_path"] = self.set_obs_vault_var.get()
        cfg_o["folder"] = self.set_obs_folder_var.get()

        self._log("Настройки применены.")

    def _apply_transcription_overrides(self):
        cfg_t = self.config.setdefault("transcription", {})
        if hasattr(self, "trans_lang_var"):
            lang = self.trans_lang_var.get()
            if lang:
                cfg_t["language"] = lang
        if hasattr(self, "trans_backend_var"):
            be = self.trans_backend_var.get()
            if be:
                cfg_t["backend"] = be
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

    def _pick_obsidian_vault(self):
        path = filedialog.askdirectory(title="Выберите Obsidian vault")
        if path:
            self.set_obs_vault_var.set(path)

    # ── Toggle combo on checkbox ─────────────────────────────────────

    @staticmethod
    def _toggle_combo(enabled_var: tk.BooleanVar, combo: ttk.Combobox):
        """Включает/выключает комбобокс по состоянию галочки."""
        combo.configure(state="readonly" if enabled_var.get() else "disabled")

    # ── Blinking red dot ──────────────────────────────────────────────

    def _do_blink(self, canvas, dot_id, flag_attr):
        if not getattr(self, flag_attr, False):
            canvas.itemconfigure(dot_id, state="hidden")
            return
        try:
            cur = canvas.itemcget(dot_id, "state")
        except tk.TclError:
            return
        canvas.itemconfigure(
            dot_id, state="hidden" if cur == "normal" else "normal")
        self.root.after(
            500, lambda: self._do_blink(canvas, dot_id, flag_attr))

    # ── Open folder ───────────────────────────────────────────────────

    def _open_recordings_folder(self):
        folder = Path(self.config.get(
            "recording", {}).get("output_dir", "recordings"))
        folder.mkdir(parents=True, exist_ok=True)
        folder = folder.resolve()
        sys = platform.system()
        cmd = (["open"] if sys == "Darwin"
               else ["explorer"] if sys == "Windows"
               else ["xdg-open"])
        subprocess.Popen(cmd + [str(folder)])

    # ── Helpers ────────────────────────────────────────────────────────

    def _pick_audio_file(self, var: tk.StringVar):
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_AUDIO))
        path = filedialog.askopenfilename(
            title="Выберите аудиофайл",
            filetypes=[("Аудио", exts), ("Все файлы", "*.*")])
        if path:
            var.set(path)

    def _pick_dir(self, var: tk.StringVar):
        path = filedialog.askdirectory(
            title="Выберите папку с аудио")
        if path:
            var.set(path)

    def _log(self, text: str):
        """Пишет в все лог-виджеты на вкладках."""
        for w in self._tab_logs:
            w.insert("end", text + "\n")
            w.see("end")

    def _copy_log(self, log_widget: tk.Text):
        """Копирует содержимое лога в буфер обмена."""
        text = log_widget.get("1.0", "end").strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self._log("Лог скопирован в буфер обмена")

    def _run_in_thread(self, target, *args):
        if self._running_task and self._running_task.is_alive():
            messagebox.showwarning("audio2text",
                                   "Задача уже выполняется.")
            return
        def wrapper():
            try:
                target(*args)
            except Exception as e:
                self.log_queue.put(f"Ошибка: {e}")
        self._running_task = threading.Thread(
            target=wrapper, daemon=True)
        self._running_task.start()

    def run(self):
        self.root.mainloop()


# ── Entry point ────────────────────────────────────────────────────────────


def main():
    app = Audio2TextApp()
    app.run()


if __name__ == "__main__":
    main()
