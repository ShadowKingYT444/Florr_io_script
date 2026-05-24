from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


Roi = tuple[int, int, int, int]
Hsv = tuple[int, int, int]


class ScreenConfig(BaseModel):
    monitor_index: int = 1
    game_window_title_contains: str = "florr"
    auto_focus_window: bool = True
    maximize_window: bool = True
    auto_detect_minimap: bool = True
    minimap_roi: Roi = (20, 20, 300, 300)
    minimap_search_roi: Roi | None = None
    world_roi: Roi = (0, 0, 1920, 1080)
    background_sample_rois: list[Roi] = Field(default_factory=list)


class RuntimeConfig(BaseModel):
    tick_hz: float = 20.0
    dry_run: bool = True
    debug_dir: Path = Path("assets/debug")
    save_debug_frames: bool = True
    debug_frame_interval_seconds: float = 5.0
    emergency_stop_key: str = "f12"
    pause_key: str = "f11"

    @field_validator("tick_hz")
    @classmethod
    def positive_tick_hz(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("tick_hz must be positive")
        return value


class HotkeysConfig(BaseModel):
    speed: str = "9"
    battle: str = "4"
    magnet: str = "5"


class MovementConfig(BaseModel):
    up: str = "w"
    down: str = "s"
    left: str = "a"
    right: str = "d"
    pulse_seconds: float = 0.08
    waypoint_radius_px: int = 8
    max_movement_keys: int = 2


class TimersConfig(BaseModel):
    forced_death_seconds: float = 10800.0
    death_wait_seconds: float = 1800.0
    dimension_confirm_frames: int = 10
    popup_confirm_frames: int = 4
    drop_confirm_frames: int = 5


class MaskConfig(BaseModel):
    walkable_hsv_min: Hsv = (0, 0, 180)
    walkable_hsv_max: Hsv = (179, 80, 255)
    brown_background_hsv_min: Hsv = (5, 35, 20)
    brown_background_hsv_max: Hsv = (35, 255, 190)
    target_hsv_min: Hsv = (35, 50, 60)
    target_hsv_max: Hsv = (95, 255, 255)
    drop_hsv_min: Hsv = (15, 30, 120)
    drop_hsv_max: Hsv = (100, 255, 255)
    red_ball_hsv_min_1: Hsv = (0, 80, 80)
    red_ball_hsv_max_1: Hsv = (10, 255, 255)
    red_ball_hsv_min_2: Hsv = (170, 80, 80)
    red_ball_hsv_max_2: Hsv = (179, 255, 255)
    grey_path_hsv_min: Hsv = (0, 0, 70)
    grey_path_hsv_max: Hsv = (179, 70, 210)
    task_green_hsv_min: Hsv = (40, 45, 35)
    task_green_hsv_max: Hsv = (100, 255, 180)


class NavigationConfig(BaseModel):
    grid_stride_px: int = 3
    path_simplify_px: float = 8.0
    hint_reference_size_px: int = 300
    first_target_hint: tuple[int, int] = (170, 235)
    second_target_hint: tuple[int, int] = (75, 32)
    player_hsv_min: Hsv = (15, 60, 80)
    player_hsv_max: Hsv = (38, 255, 255)
    fallback_direct_movement: bool = True
    fallback_arrival_seconds: float = 8.0


class CombatConfig(BaseModel):
    enabled: bool = True
    click_interval_seconds: float = 0.08
    threat_radius_px: float = 185.0
    attack_radius_px: float = 430.0
    kite_radius_px: float = 250.0
    mob_min_area_px: int = 40
    mob_max_area_px: int = 25000
    hold_mouse_attack: bool = True
    zoom_out_scrolls: int = 3


class DropsConfig(BaseModel):
    density_enter: int = 35
    density_exit: int = 12
    min_blob_area_px: int = 4
    max_blob_area_px: int = 650


class BallTaskConfig(BaseModel):
    min_ball_area_px: int = 40
    min_path_area_px: int = 300
    drag_step_px: int = 10
    drag_duration_seconds: float = 2.5


class VisionConfig(BaseModel):
    masks: MaskConfig = Field(default_factory=MaskConfig)
    navigation: NavigationConfig = Field(default_factory=NavigationConfig)
    combat: CombatConfig = Field(default_factory=CombatConfig)
    drops: DropsConfig = Field(default_factory=DropsConfig)
    ball_task: BallTaskConfig = Field(default_factory=BallTaskConfig)


class BotConfig(BaseModel):
    screen: ScreenConfig = Field(default_factory=ScreenConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    hotkeys: HotkeysConfig = Field(default_factory=HotkeysConfig)
    movement: MovementConfig = Field(default_factory=MovementConfig)
    timers: TimersConfig = Field(default_factory=TimersConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        elif value is not None:
            merged[key] = value
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object at {path}")
    return data


def load_config(config_path: str | Path, calibration_path: str | Path | None = None) -> BotConfig:
    config_file = Path(config_path)
    raw = load_yaml(config_file)
    if calibration_path:
        calibration_file = Path(calibration_path)
        if calibration_file.exists():
            raw = deep_merge(raw, load_yaml(calibration_file))
    config = BotConfig.model_validate(raw)
    if not config.runtime.debug_dir.is_absolute():
        config.runtime.debug_dir = config_file.parent.parent / config.runtime.debug_dir
    return config
