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

        # Device
        ttk.Label(frame, text="Устройство:").grid(row=0, column=0, sticky="w", pady=5)
        self.device_var = tk.StringVar(value="По умолчанию")
        device_combo = ttk.Combobox(frame, textvariable=self.device_var,
                                    state="readonly", width=50)
        device_combo.grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=5)
        ttk.Button(frame, text="Обновить", width=10,
                   command=lambda: self._refresh_devices(device_combo)).grid(
            row=0, column=2, padx=(5, 0), pady=5)

        self._refresh_devices(device_combo)

        # VU meter
        self.vu_var = tk.DoubleVar(value=0)
        ttk.Label(frame, text="Уровень:").grid(row=1, column=0, sticky="w", pady=5)
        self.vu_bar = ttk.Progressbar(frame, variable=self.vu_var,
                                       maximum=1.0, mode="determinate")
        self.vu_bar.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=5)

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=2, column=0, columnspan=3, pady=20)

        self.record_btn = ttk.Button(btn_frame, text="Начать запись",
                                     command=self._toggle_record)
        self.record_btn.pack(side="left", padx=5)

        self.record_status = ttk.Label(btn_frame, text="")
        self.record_status.pack(side="left", padx=10)

        frame.columnconfigure(1, weight=1)

        self._recorder = None
        self._recording = False

    def _refresh_devices(self, combo: ttk.Combobox):
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            input_devs = ["По умолчанию"]
            self._device_ids = [None]
            for i, d in enumerate(devices):
                if d["max_input_channels"] > 0:
                    input_devs.append(f"{i}: {d['name']}")
                    self._device_ids.append(i)
            combo["values"] = input_devs
        except Exception as e:
            combo["values"] = ["По умолчанию"]
            self._device_ids = [None]
            self._log(f"Ошибка получения устройств: {e}")

    def _get_selected_device(self) -> int | None:
        idx = 0
        try:
            values = list(self.device_var.master.nametowidget(
                ".").tk.splitlist(""))
        except Exception:
            pass
        selection = self.device_var.get()
        if selection == "По умолчанию":
            return None
        try:
            dev_id = int(selection.split(":")[0])
            return dev_id
        except (ValueError, IndexError):
            return None

    def _toggle_record(self):
        if not self._recording:
            self._start_record()
        else:
            self._stop_record()

    def _start_record(self):
        from recorder import Recorder
        self._recorder = Recorder(self.config)
        self._recording = True
        self.record_btn.configure(text="Остановить")
        self.record_status.configure(text="Идёт запись...")

        device = self._get_selected_device()

        # Patch callback to update VU meter
        original_cb = self._recorder._audio_callback

        def patched_cb(indata, frames, time_info, status):
            import numpy as np
            if status:
                self.log_queue.put(f"Audio status: {status}")
            with self._recorder._lock:
                self._recorder._frames.append(indata.copy())
            rms = float(np.sqrt(np.mean(indata ** 2)))
            self.vu_var.set(min(rms * 10, 1.0))

        self._recorder._audio_callback = patched_cb

        def do_record():
            try:
                path = self._recorder.record(device=device)
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
        self.vu_var.set(0)

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
