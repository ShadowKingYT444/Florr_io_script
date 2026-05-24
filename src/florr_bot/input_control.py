from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from threading import Event
from typing import Iterable

import pyautogui
import pydirectinput
import pygetwindow as gw
from pynput import keyboard

from .config import MovementConfig


@dataclass(frozen=True)
class MovementIntent:
    dx: float
    dy: float

    @property
    def magnitude(self) -> float:
        return math.hypot(self.dx, self.dy)


class InputController:
    def __init__(
        self,
        movement: MovementConfig,
        dry_run: bool,
        logger: logging.Logger,
        emergency_stop_key: str = "f12",
        pause_key: str = "f11",
    ):
        self.movement = movement
        self.dry_run = dry_run
        self.logger = logger
        self.stop_requested = Event()
        self.paused = Event()
        self._listener: keyboard.Listener | None = None
        self._emergency_stop_key = emergency_stop_key.lower()
        self._pause_key = pause_key.lower()
        self._held_movement_keys: set[str] = set()
        self._attack_held = False
        pyautogui.FAILSAFE = True
        pydirectinput.PAUSE = 0.0

    def start_hotkeys(self) -> None:
        if self._listener:
            return

        def on_press(key: keyboard.Key | keyboard.KeyCode) -> None:
            name = self._key_name(key)
            if name == self._emergency_stop_key:
                self.logger.warning("Emergency stop requested")
                self.stop_requested.set()
            elif name == self._pause_key:
                if self.paused.is_set():
                    self.logger.info("Resuming from pause")
                    self.paused.clear()
                else:
                    self.logger.info("Paused")
                    self.paused.set()

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.daemon = True
        self._listener.start()

    def close(self) -> None:
        self.set_attack_held(False)
        self.release_all_movement()
        if self._listener:
            self._listener.stop()
            self._listener = None

    def focus_window(self, title_contains: str, *, maximize: bool = False) -> tuple[int, int, int, int] | None:
        needle = title_contains.lower().strip()
        windows = [window for window in gw.getAllWindows() if needle in window.title.lower()]
        windows = [window for window in windows if window.width > 100 and window.height > 100]
        if not windows:
            self.logger.warning("Could not find a window title containing %r", title_contains)
            return None

        window = max(windows, key=lambda item: item.width * item.height)
        self.logger.info("Focusing window: %s", window.title)
        try:
            if maximize:
                window.maximize()
                time.sleep(0.2)
            window.activate()
            time.sleep(0.2)
            # Even dry-run needs the browser/game visible so capture sees florr.io,
            # but movement, hotkeys, and attack still stay disabled.
            pyautogui.click(window.left + window.width // 2, window.top + window.height // 2)
        except Exception as exc:
            self.logger.warning("Window focus failed: %s", exc)
        return int(window.left), int(window.top), int(window.width), int(window.height)

    @staticmethod
    def _key_name(key: keyboard.Key | keyboard.KeyCode) -> str:
        if isinstance(key, keyboard.KeyCode):
            return str(key.char or "").lower()
        return str(key).replace("Key.", "").lower()

    def press(self, key: str) -> None:
        if self.dry_run:
            self.logger.debug("dry-run press %s", key)
            return
        pydirectinput.press(key)

    def click(self, x: int | None = None, y: int | None = None) -> None:
        if self.dry_run:
            self.logger.debug("dry-run click %s %s", x, y)
            return
        pyautogui.click(x=x, y=y)

    def set_attack_held(self, enabled: bool) -> None:
        if enabled == self._attack_held:
            return
        self._attack_held = enabled
        if self.dry_run:
            self.logger.debug("dry-run mouse %s", "down" if enabled else "up")
            return
        if enabled:
            pyautogui.mouseDown()
        else:
            pyautogui.mouseUp()

    def scroll(self, clicks: int) -> None:
        if clicks == 0:
            return
        if self.dry_run:
            self.logger.debug("dry-run scroll %d", clicks)
            return
        pyautogui.scroll(clicks)

    def drag_path(self, points: Iterable[tuple[int, int]], duration_seconds: float) -> None:
        path = list(points)
        if not path:
            return
        if self.dry_run:
            self.logger.info("dry-run drag through %d points", len(path))
            return
        x0, y0 = path[0]
        pyautogui.moveTo(x0, y0, duration=0.05)
        pyautogui.mouseDown()
        step_duration = max(0.01, duration_seconds / max(1, len(path) - 1))
        try:
            for x, y in path[1:]:
                if self.stop_requested.is_set():
                    break
                pyautogui.moveTo(x, y, duration=step_duration)
        finally:
            pyautogui.mouseUp()

    def movement_keys_for_intent(self, intent: MovementIntent) -> list[str]:
        if intent.magnitude <= 0.05:
            return []
        keys: list[str] = []
        if intent.dy < -0.25:
            keys.append(self.movement.up)
        elif intent.dy > 0.25:
            keys.append(self.movement.down)
        if intent.dx < -0.25:
            keys.append(self.movement.left)
        elif intent.dx > 0.25:
            keys.append(self.movement.right)
        return keys[: self.movement.max_movement_keys]

    def pulse_movement(self, intent: MovementIntent, seconds: float | None = None) -> None:
        keys = self.movement_keys_for_intent(intent)
        if not keys:
            return
        duration = self.movement.pulse_seconds if seconds is None else seconds
        if self.dry_run:
            self.logger.debug("dry-run movement %s for %.3fs", "+".join(keys), duration)
            return
        for key in keys:
            pydirectinput.keyDown(key)
        time.sleep(duration)
        for key in reversed(keys):
            pydirectinput.keyUp(key)

    def set_movement(self, intent: MovementIntent) -> None:
        keys = set(self.movement_keys_for_intent(intent))
        if keys == self._held_movement_keys:
            return
        to_release = self._held_movement_keys - keys
        to_press = keys - self._held_movement_keys
        self._held_movement_keys = keys
        if self.dry_run:
            self.logger.debug(
                "dry-run movement hold=%s release=%s",
                "+".join(sorted(to_press)) or "-",
                "+".join(sorted(to_release)) or "-",
            )
            return
        for key in sorted(to_release):
            pydirectinput.keyUp(key)
        for key in sorted(to_press):
            pydirectinput.keyDown(key)

    def release_all_movement(self) -> None:
        keys = set(self._held_movement_keys) | {
            self.movement.up,
            self.movement.down,
            self.movement.left,
            self.movement.right,
        }
        self._held_movement_keys.clear()
        for key in keys:
            if not self.dry_run:
                pydirectinput.keyUp(key)
