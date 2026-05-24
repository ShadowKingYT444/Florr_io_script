from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mss
import numpy as np

from .config import Roi, ScreenConfig


@dataclass(frozen=True)
class CapturedFrame:
    bgr: np.ndarray
    roi: Roi
    timestamp: float


class ScreenCapture:
    def __init__(self, screen: ScreenConfig):
        self.screen = screen
        self._mss = mss.mss()
        if screen.monitor_index >= len(self._mss.monitors):
            raise ValueError(
                f"monitor_index {screen.monitor_index} is unavailable; "
                f"mss sees {len(self._mss.monitors) - 1} monitor(s)"
            )
        self.monitor = self._mss.monitors[screen.monitor_index]

    def close(self) -> None:
        self._mss.close()

    def _region(self, roi: Roi) -> dict[str, int]:
        x, y, w, h = roi
        return {
            "left": self.monitor["left"] + x,
            "top": self.monitor["top"] + y,
            "width": w,
            "height": h,
        }

    def absolute_to_monitor_roi(self, left: int, top: int, width: int, height: int) -> Roi:
        return (
            int(left - self.monitor["left"]),
            int(top - self.monitor["top"]),
            int(width),
            int(height),
        )

    def clamp_roi(self, roi: Roi) -> Roi:
        x, y, w, h = roi
        max_w = int(self.monitor["width"])
        max_h = int(self.monitor["height"])
        x = max(0, min(int(x), max_w - 1))
        y = max(0, min(int(y), max_h - 1))
        w = max(1, min(int(w), max_w - x))
        h = max(1, min(int(h), max_h - y))
        return (x, y, w, h)

    def set_world_roi(self, roi: Roi) -> None:
        self.screen.world_roi = self.clamp_roi(roi)

    def set_minimap_roi(self, roi: Roi) -> None:
        self.screen.minimap_roi = self.clamp_roi(roi)

    def grab(self, roi: Roi) -> CapturedFrame:
        roi = self.clamp_roi(roi)
        raw = np.asarray(self._mss.grab(self._region(roi)))
        bgr = raw[:, :, :3].copy()
        return CapturedFrame(bgr=bgr, roi=roi, timestamp=time.monotonic())

    def grab_world(self) -> CapturedFrame:
        return self.grab(self.screen.world_roi)

    def grab_minimap(self) -> CapturedFrame:
        return self.grab(self.screen.minimap_roi)


class DebugFrameWriter:
    def __init__(self, debug_dir: Path, interval_seconds: float):
        self.debug_dir = debug_dir
        self.interval_seconds = interval_seconds
        self._last_write = 0.0
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def maybe_write(self, label: str, frame: np.ndarray, now: float | None = None) -> Path | None:
        current = time.monotonic() if now is None else now
        if current - self._last_write < self.interval_seconds:
            return None
        self._last_write = current
        safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)
        out = self.debug_dir / f"{int(time.time())}_{safe_label}.png"
        cv2.imwrite(str(out), frame)
        return out
