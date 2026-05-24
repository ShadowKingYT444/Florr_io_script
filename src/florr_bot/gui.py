from __future__ import annotations

import logging
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import yaml

from florr_bot.capture import ScreenCapture
from florr_bot.config import BotConfig, load_config
from florr_bot.input_control import InputController
from florr_bot.logging_utils import configure_logging
from florr_bot.main import resolve_resource_path
from florr_bot.screen_analysis import detect_minimap_roi, draw_roi
from florr_bot.state_machine import FlorrBot


class TkLogHandler(logging.Handler):
    def __init__(self, messages: queue.Queue[str]):
        super().__init__()
        self.messages = messages

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.put(self.format(record))


class BotRunner:
    def __init__(self, app: "FlorrBotApp"):
        self.app = app
        self.thread: threading.Thread | None = None
        self.bot: FlorrBot | None = None
        self.capture: ScreenCapture | None = None
        self.inputs: InputController | None = None

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, config: BotConfig, logger: logging.Logger) -> None:
        if self.running:
            return

        def target() -> None:
            try:
                self.capture = ScreenCapture(config.screen)
                self.inputs = InputController(
                    movement=config.movement,
                    dry_run=config.runtime.dry_run,
                    logger=logger,
                    emergency_stop_key=config.runtime.emergency_stop_key,
                    pause_key=config.runtime.pause_key,
                )
                self.bot = FlorrBot(config, self.capture, self.inputs, logger)
                self.app.set_status("Running")
                self.bot.run()
            except Exception:
                logger.exception("Bot crashed")
            finally:
                self.bot = None
                self.capture = None
                self.inputs = None
                self.app.after(0, self.app.on_bot_stopped)

        self.thread = threading.Thread(target=target, name="florr-bot", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.inputs is not None:
            self.inputs.stop_requested.set()
            self.inputs.set_attack_held(False)
            self.inputs.release_all_movement()


class FlorrBotApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Florr.io Bot")
        self.geometry("760x620")
        self.minsize(720, 560)

        self.root_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path.cwd()
        self.config_path = tk.StringVar(value=str(resolve_resource_path("config/default.yaml")))
        self.calibration_path = tk.StringVar(value=str(self.root_dir / "calibration.yaml"))
        self.mode = tk.StringVar(value="dry")
        self.status = tk.StringVar(value="Idle")
        self.window_title = tk.StringVar(value="florr")
        self.speed_key = tk.StringVar(value="9")
        self.damage_key = tk.StringVar(value="4")
        self.magnet_key = tk.StringVar(value="5")
        self.pause_key = tk.StringVar(value="f11")
        self.stop_key = tk.StringVar(value="f12")
        self.auto_focus = tk.BooleanVar(value=True)
        self.auto_detect = tk.BooleanVar(value=True)
        self.maximize = tk.BooleanVar(value=True)

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.logger = configure_logging(self.root_dir / "assets" / "debug", verbose=True)
        self.tk_handler = TkLogHandler(self.log_queue)
        self.tk_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%H:%M:%S"))
        self.tk_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(self.tk_handler)

        self.runner = BotRunner(self)
        self._build_ui()
        self.load_config_into_fields()
        self.after(100, self.flush_logs)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer)
        header.pack(fill=tk.X)
        ttk.Label(header, text="Florr.io Bot", font=("Segoe UI", 18, "bold")).pack(side=tk.LEFT)
        ttk.Label(header, textvariable=self.status, font=("Segoe UI", 11)).pack(side=tk.RIGHT)

        controls = ttk.LabelFrame(outer, text="Controls", padding=10)
        controls.pack(fill=tk.X, pady=(12, 8))
        ttk.Button(controls, text="Calibrate Screen", command=self.calibrate_screen).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(controls, text="Start", command=self.start_bot).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(controls, text="Stop", command=self.stop_bot).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(controls, text="Dry Run", variable=self.mode, value="dry").pack(side=tk.LEFT, padx=(18, 4))
        ttk.Radiobutton(controls, text="Live", variable=self.mode, value="live").pack(side=tk.LEFT)

        files = ttk.LabelFrame(outer, text="Files", padding=10)
        files.pack(fill=tk.X, pady=8)
        self._path_row(files, "Config", self.config_path, self.pick_config)
        self._path_row(files, "Calibration", self.calibration_path, self.pick_calibration)

        settings = ttk.LabelFrame(outer, text="Settings", padding=10)
        settings.pack(fill=tk.X, pady=8)
        for col in range(6):
            settings.columnconfigure(col, weight=1)
        self._entry(settings, "Window title", self.window_title, 0, 0, width=18)
        self._entry(settings, "Travel", self.speed_key, 0, 2, width=7)
        self._entry(settings, "Damage", self.damage_key, 0, 4, width=7)
        self._entry(settings, "Magnet", self.magnet_key, 1, 0, width=7)
        self._entry(settings, "Pause", self.pause_key, 1, 2, width=7)
        self._entry(settings, "Stop", self.stop_key, 1, 4, width=7)
        ttk.Checkbutton(settings, text="Focus browser", variable=self.auto_focus).grid(row=2, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Checkbutton(settings, text="Maximize", variable=self.maximize).grid(row=2, column=2, sticky=tk.W, pady=(8, 0))
        ttk.Checkbutton(settings, text="Auto-detect minimap", variable=self.auto_detect).grid(row=2, column=4, sticky=tk.W, pady=(8, 0))
        ttk.Button(settings, text="Save Calibration", command=self.save_calibration).grid(row=3, column=0, sticky=tk.W, pady=(10, 0))

        log_frame = ttk.LabelFrame(outer, text="Log", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.log_text = tk.Text(log_frame, height=14, wrap=tk.WORD, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _path_row(self, parent: ttk.Frame, label: str, variable: tk.StringVar, command) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, width=12).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        ttk.Button(row, text="Browse", command=command).pack(side=tk.RIGHT)

    def _entry(self, parent: ttk.Frame, label: str, variable: tk.StringVar, row: int, column: int, width: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky=tk.W, padx=(0, 4), pady=3)
        ttk.Entry(parent, textvariable=variable, width=width).grid(row=row, column=column + 1, sticky=tk.W, pady=3)

    def pick_config(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")])
        if path:
            self.config_path.set(path)
            self.load_config_into_fields()

    def pick_calibration(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")])
        if path:
            self.calibration_path.set(path)
            self.load_config_into_fields()

    def build_config(self) -> BotConfig:
        calibration = Path(self.calibration_path.get())
        calibration_path = calibration if calibration.exists() else None
        config = load_config(Path(self.config_path.get()), calibration_path)
        config.runtime.dry_run = self.mode.get() != "live"
        config.screen.game_window_title_contains = self.window_title.get().strip() or "florr"
        config.screen.auto_focus_window = self.auto_focus.get()
        config.screen.auto_detect_minimap = self.auto_detect.get()
        config.screen.maximize_window = self.maximize.get()
        config.hotkeys.speed = self.speed_key.get().strip() or "9"
        config.hotkeys.battle = self.damage_key.get().strip() or "4"
        config.hotkeys.magnet = self.magnet_key.get().strip() or "5"
        config.runtime.pause_key = self.pause_key.get().strip() or "f11"
        config.runtime.emergency_stop_key = self.stop_key.get().strip() or "f12"
        config.runtime.debug_dir = self.root_dir / "assets" / "debug"
        return config

    def load_config_into_fields(self) -> None:
        try:
            config = self.build_config()
        except Exception:
            return
        self.window_title.set(config.screen.game_window_title_contains)
        self.speed_key.set(config.hotkeys.speed)
        self.damage_key.set(config.hotkeys.battle)
        self.magnet_key.set(config.hotkeys.magnet)
        self.pause_key.set(config.runtime.pause_key)
        self.stop_key.set(config.runtime.emergency_stop_key)
        self.auto_focus.set(config.screen.auto_focus_window)
        self.auto_detect.set(config.screen.auto_detect_minimap)
        self.maximize.set(config.screen.maximize_window)

    def start_bot(self) -> None:
        if self.runner.running:
            messagebox.showinfo("Already running", "The bot is already running.")
            return
        try:
            config = self.build_config()
        except Exception as exc:
            messagebox.showerror("Config error", str(exc))
            return
        self.set_status("Starting")
        self.logger.info("Starting from UI in %s mode", "dry-run" if config.runtime.dry_run else "live")
        self.runner.start(config, self.logger)

    def stop_bot(self) -> None:
        self.logger.info("Stop requested from UI")
        self.set_status("Stopping")
        self.runner.stop()

    def on_bot_stopped(self) -> None:
        self.set_status("Idle")

    def calibrate_screen(self) -> None:
        try:
            config = self.build_config()
            capture = ScreenCapture(config.screen)
            inputs = InputController(
                movement=config.movement,
                dry_run=True,
                logger=self.logger,
                emergency_stop_key=config.runtime.emergency_stop_key,
                pause_key=config.runtime.pause_key,
            )
            try:
                window = inputs.focus_window(config.screen.game_window_title_contains, maximize=config.screen.maximize_window)
                if window is not None:
                    capture.set_world_roi(capture.absolute_to_monitor_roi(*window))
                world = capture.grab_world()
                detection = detect_minimap_roi(world.bgr, config.screen.minimap_search_roi)
                debug_dir = config.runtime.debug_dir
                debug_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(debug_dir / "calibration_world.png"), world.bgr)
                data = {"screen": {"world_roi": list(config.screen.world_roi)}}
                if detection is not None:
                    wx, wy, _, _ = config.screen.world_roi
                    minimap_roi = (wx + detection.roi[0], wy + detection.roi[1], detection.roi[2], detection.roi[3])
                    data["screen"]["minimap_roi"] = list(minimap_roi)
                    data["screen"]["auto_detect_minimap"] = False
                    cv2.imwrite(str(debug_dir / "calibration_minimap_detected.png"), draw_roi(world.bgr, detection))
                    self.logger.info("Detected minimap ROI %s", minimap_roi)
                else:
                    self.logger.warning("Minimap was not detected; wrote calibration_world.png")
                self._write_calibration(data)
            finally:
                capture.close()
                inputs.close()
        except Exception as exc:
            self.logger.exception("Calibration failed")
            messagebox.showerror("Calibration failed", str(exc))

    def save_calibration(self) -> None:
        try:
            config = self.build_config()
            data = {
                "screen": {
                    "game_window_title_contains": config.screen.game_window_title_contains,
                    "auto_focus_window": config.screen.auto_focus_window,
                    "maximize_window": config.screen.maximize_window,
                    "auto_detect_minimap": config.screen.auto_detect_minimap,
                },
                "hotkeys": {
                    "speed": config.hotkeys.speed,
                    "battle": config.hotkeys.battle,
                    "magnet": config.hotkeys.magnet,
                },
                "runtime": {
                    "pause_key": config.runtime.pause_key,
                    "emergency_stop_key": config.runtime.emergency_stop_key,
                },
            }
            self._write_calibration(data)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def _write_calibration(self, data: dict) -> None:
        path = Path(self.calibration_path.get())
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                existing = yaml.safe_load(handle) or {}
        merged = self._deep_merge(existing, data)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(merged, handle, sort_keys=False)
        self.logger.info("Wrote calibration: %s", path)
        self.calibration_path.set(str(path))

    @staticmethod
    def _deep_merge(base: dict, overlay: dict) -> dict:
        merged = dict(base)
        for key, value in overlay.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = FlorrBotApp._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def set_status(self, value: str) -> None:
        self.after(0, self.status.set, value)

    def flush_logs(self) -> None:
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, line + "\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)
        self.after(100, self.flush_logs)


def main() -> int:
    app = FlorrBotApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
