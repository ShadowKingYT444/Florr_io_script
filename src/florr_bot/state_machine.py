from __future__ import annotations

import logging
import math
import time
from enum import Enum, auto

from .capture import DebugFrameWriter, ScreenCapture
from .combat import CombatPlanner, MobDetector
from .config import BotConfig
from .detectors import BackgroundDetector, PopupDetector
from .drops import DropDetector
from .input_control import InputController, MovementIntent
from .screen_analysis import detect_marker, detect_minimap_roi, draw_roi


class BotState(Enum):
    BOOT = auto()
    FOCUS_GAME = auto()
    ACTIVATE_SPEED = auto()
    NAVIGATE_TO_PORTAL = auto()
    WAIT_FOR_DIMENSION_CHANGE = auto()
    NAVIGATE_TO_GRIND_ZONE = auto()
    GRIND_COMBAT = auto()
    COLLECT_DROPS = auto()
    SOLVE_BALL_TASK = auto()
    FORCED_DEATH = auto()
    DEATH_WAIT = auto()
    RECOVER_AND_RESUME = auto()
    PAUSED = auto()
    ERROR = auto()


class FlorrBot:
    def __init__(
        self,
        config: BotConfig,
        capture: ScreenCapture,
        inputs: InputController,
        logger: logging.Logger,
    ):
        self.config = config
        self.capture = capture
        self.inputs = inputs
        self.logger = logger
        self.state = BotState.BOOT
        self.last_state_change = time.monotonic()
        self.grind_started_at = time.monotonic()
        self.dimension_frames = 0
        self.popup_frames = 0
        self.last_attack_at = 0.0
        self.last_nav_warning_at = 0.0
        self.grind_zoomed = False
        self.battle_mode_active = False
        self.debug_writer = DebugFrameWriter(
            config.runtime.debug_dir, config.runtime.debug_frame_interval_seconds
        )
        self.background = BackgroundDetector(config.vision.masks, config.screen.background_sample_rois)
        self.popup = PopupDetector(config.vision.masks)
        self.mob_detector = MobDetector(config.vision.masks, config.vision.combat)
        self.combat = CombatPlanner(config.vision.combat)
        self.drop_detector = DropDetector(config.vision.masks, config.vision.drops)
        self._navigator = None
        self._ball_task = None

    def run(self) -> None:
        self.inputs.start_hotkeys()
        self.logger.info("Starting bot in %s mode", "dry-run" if self.config.runtime.dry_run else "live")
        try:
            while not self.inputs.stop_requested.is_set():
                started = time.monotonic()
                if self.inputs.paused.is_set():
                    self._transition(BotState.PAUSED)
                    time.sleep(0.2)
                    continue
                self.tick()
                elapsed = time.monotonic() - started
                sleep_for = max(0.0, (1.0 / self.config.runtime.tick_hz) - elapsed)
                time.sleep(sleep_for)
        finally:
            self.inputs.set_attack_held(False)
            self.inputs.release_all_movement()
            self.capture.close()
            self.inputs.close()
            self.logger.info("Bot stopped")

    def tick(self) -> None:
        if self.state == BotState.BOOT:
            self._transition(BotState.FOCUS_GAME)
        elif self.state == BotState.FOCUS_GAME:
            self._focus_and_calibrate_screen()
            self._transition(BotState.ACTIVATE_SPEED)
        elif self.state == BotState.ACTIVATE_SPEED:
            self.inputs.press(self.config.hotkeys.speed)
            self._transition(BotState.NAVIGATE_TO_PORTAL)
        elif self.state == BotState.NAVIGATE_TO_PORTAL:
            self._navigate(first_map=True)
        elif self.state == BotState.WAIT_FOR_DIMENSION_CHANGE:
            self._wait_for_dimension()
        elif self.state == BotState.NAVIGATE_TO_GRIND_ZONE:
            self._navigate(first_map=False)
        elif self.state in (BotState.GRIND_COMBAT, BotState.COLLECT_DROPS):
            self._grind()
        elif self.state == BotState.SOLVE_BALL_TASK:
            self._solve_ball_task()
        elif self.state == BotState.FORCED_DEATH:
            self._forced_death()
        elif self.state == BotState.DEATH_WAIT:
            self._death_wait()
        elif self.state == BotState.RECOVER_AND_RESUME:
            self.inputs.press(self.config.hotkeys.speed)
            self.battle_mode_active = False
            self.grind_started_at = time.monotonic()
            self._transition(BotState.NAVIGATE_TO_GRIND_ZONE)

    def _transition(self, state: BotState) -> None:
        if state != self.state:
            self.logger.info("State %s -> %s", self.state.name, state.name)
            self.state = state
            self.last_state_change = time.monotonic()
            if state in (BotState.NAVIGATE_TO_PORTAL, BotState.NAVIGATE_TO_GRIND_ZONE):
                self.inputs.set_attack_held(False)
                self.battle_mode_active = False
            if state == BotState.GRIND_COMBAT:
                self.battle_mode_active = False

    def _focus_and_calibrate_screen(self) -> None:
        if self.config.screen.auto_focus_window:
            window = self.inputs.focus_window(
                self.config.screen.game_window_title_contains,
                maximize=self.config.screen.maximize_window,
            )
            if window is not None:
                self.capture.set_world_roi(self.capture.absolute_to_monitor_roi(*window))
                self.logger.info("Using world ROI %s", self.config.screen.world_roi)

        if not self.config.screen.auto_detect_minimap:
            return

        world = self.capture.grab_world()
        detection = detect_minimap_roi(world.bgr, self.config.screen.minimap_search_roi)
        if detection is None:
            self.logger.warning(
                "Could not auto-detect minimap; keeping configured minimap ROI %s",
                self.config.screen.minimap_roi,
            )
            if self.config.runtime.save_debug_frames:
                self.debug_writer.maybe_write("minimap_detection_failed", world.bgr, world.timestamp)
            return

        wx, wy, _, _ = self.config.screen.world_roi
        mx, my, mw, mh = detection.roi
        self.capture.set_minimap_roi((wx + mx, wy + my, mw, mh))
        self.logger.info(
            "Auto-detected minimap ROI %s score=%.3f white=%.2f dark=%.2f",
            self.config.screen.minimap_roi,
            detection.score,
            detection.white_fraction,
            detection.dark_fraction,
        )
        if self.config.runtime.save_debug_frames:
            self.debug_writer.maybe_write("minimap_detection", draw_roi(world.bgr, detection), world.timestamp)

    def _navigator_instance(self):
        if self._navigator is None:
            from .navigation import MinimapNavigator

            masks = self.config.vision.masks
            self._navigator = MinimapNavigator(
                walkable_ranges=[(masks.walkable_hsv_min, masks.walkable_hsv_max)],
                target_ranges=[(masks.target_hsv_min, masks.target_hsv_max)],
                cell_size=max(1, self.config.vision.navigation.grid_stride_px),
                min_walkable_fraction=0.45,
                min_target_area=1,
                allow_diagonal=True,
                simplify=True,
            )
        return self._navigator

    def _ball_task_instance(self):
        if self._ball_task is None:
            from .ball_task import BallTaskDetector

            ball = self.config.vision.ball_task
            self._ball_task = BallTaskDetector(
                min_ball_area=ball.min_ball_area_px,
                min_path_area=ball.min_path_area_px,
                waypoint_spacing=max(1, ball.drag_step_px),
            )
        return self._ball_task

    def _navigate(self, first_map: bool) -> None:
        if first_map:
            world = self.capture.grab_world()
            if self.background.is_brown_dimension(world.bgr):
                self.inputs.release_all_movement()
                self.dimension_frames = self.config.timers.dimension_confirm_frames
                self._transition(BotState.NAVIGATE_TO_GRIND_ZONE)
                return

        frame = self.capture.grab_minimap()
        if self.config.runtime.save_debug_frames:
            self.debug_writer.maybe_write("minimap", frame.bgr, frame.timestamp)

        navigator = self._navigator_instance()
        raw_hint = (
            self.config.vision.navigation.first_target_hint
            if first_map
            else self.config.vision.navigation.second_target_hint
        )
        height, width = frame.bgr.shape[:2]
        hint = self._scale_hint(raw_hint, width, height)
        player = detect_marker(
            frame.bgr,
            self.config.vision.navigation.player_hsv_min,
            self.config.vision.navigation.player_hsv_max,
            prefer_center=False,
        )
        origin = player.point if player is not None else (width // 2, height // 2)
        plan = navigator.plan(
            frame.bgr,
            origin=origin,
            hints=[hint],
            deadzone=self.config.movement.waypoint_radius_px,
        )
        if plan.target is None:
            self._log_nav("Navigation target not found; origin=%s hint=%s", origin, hint)
            self._fallback_navigation(origin, hint, first_map)
            return
        if plan.path is None or not plan.path.pixel_path:
            self._log_nav(
                "Navigation path unavailable; origin=%s target=%s hint=%s minimap=%s",
                origin,
                plan.target.point,
                hint,
                self.config.screen.minimap_roi,
            )
            self._fallback_navigation(origin, plan.target.point, first_map)
            return

        self.inputs.set_movement(MovementIntent(plan.movement.x, plan.movement.y))
        target_distance = math.hypot(plan.target.point[0] - origin[0], plan.target.point[1] - origin[1])
        if target_distance <= self.config.movement.waypoint_radius_px or plan.movement.distance == 0.0:
            self.inputs.release_all_movement()
            self._transition(BotState.WAIT_FOR_DIMENSION_CHANGE if first_map else BotState.GRIND_COMBAT)

    def _scale_hint(self, hint: tuple[int, int], width: int, height: int) -> tuple[int, int]:
        ref = max(1, self.config.vision.navigation.hint_reference_size_px)
        return (
            int(round(hint[0] * width / ref)),
            int(round(hint[1] * height / ref)),
        )

    def _fallback_navigation(self, origin: tuple[int, int], target: tuple[int, int], first_map: bool) -> None:
        if not self.config.vision.navigation.fallback_direct_movement:
            self.inputs.release_all_movement()
            return
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        distance = math.hypot(dx, dy)
        if distance <= self.config.movement.waypoint_radius_px:
            self.inputs.release_all_movement()
            self._transition(BotState.WAIT_FOR_DIMENSION_CHANGE if first_map else BotState.GRIND_COMBAT)
            return
        self.inputs.set_movement(MovementIntent(dx / distance, dy / distance))
        elapsed = time.monotonic() - self.last_state_change
        if elapsed >= self.config.vision.navigation.fallback_arrival_seconds:
            self.inputs.release_all_movement()
            self._transition(BotState.WAIT_FOR_DIMENSION_CHANGE if first_map else BotState.GRIND_COMBAT)

    def _log_nav(self, message: str, *args: object) -> None:
        now = time.monotonic()
        if now - self.last_nav_warning_at >= 1.5:
            self.logger.warning(message, *args)
            self.last_nav_warning_at = now

    def _wait_for_dimension(self) -> None:
        frame = self.capture.grab_world()
        score = self.background.brown_score(frame.bgr)
        if self.background.is_brown_dimension(frame.bgr):
            self.dimension_frames += 1
        else:
            self.dimension_frames = 0
        self.logger.debug("Brown dimension score %.3f frames=%d", score, self.dimension_frames)
        if self.dimension_frames >= self.config.timers.dimension_confirm_frames:
            self.inputs.press(self.config.hotkeys.speed)
            self._transition(BotState.NAVIGATE_TO_GRIND_ZONE)

    def _grind(self) -> None:
        now = time.monotonic()
        if now - self.grind_started_at >= self.config.timers.forced_death_seconds:
            self._transition(BotState.FORCED_DEATH)
            return

        frame = self.capture.grab_world()
        if self.config.runtime.save_debug_frames:
            self.debug_writer.maybe_write("world", frame.bgr, frame.timestamp)

        if self.popup.looks_like_ball_task(frame.bgr):
            self.popup_frames += 1
        else:
            self.popup_frames = 0
        if self.popup_frames >= self.config.timers.popup_confirm_frames:
            self.inputs.set_attack_held(False)
            self._transition(BotState.SOLVE_BALL_TASK)
            return

        drops = self.drop_detector.detect(frame.bgr)
        if drops.should_collect:
            if self.state != BotState.COLLECT_DROPS:
                self.inputs.set_attack_held(False)
                self.inputs.press(self.config.hotkeys.magnet)
                self.battle_mode_active = False
                self._transition(BotState.COLLECT_DROPS)
            return
        if self.state == BotState.COLLECT_DROPS and not drops.should_collect:
            self.inputs.press(self.config.hotkeys.battle)
            self.battle_mode_active = True
            self.inputs.set_attack_held(self.config.vision.combat.hold_mouse_attack)
            self._transition(BotState.GRIND_COMBAT)

        if self.state == BotState.GRIND_COMBAT:
            if not self.grind_zoomed and self.config.vision.combat.zoom_out_scrolls:
                self.inputs.scroll(-abs(self.config.vision.combat.zoom_out_scrolls))
                self.grind_zoomed = True
            if not self.battle_mode_active:
                self.inputs.press(self.config.hotkeys.battle)
                self.battle_mode_active = True
            self.inputs.set_attack_held(self.config.vision.combat.hold_mouse_attack)

        mobs = self.mob_detector.detect(frame.bgr)
        decision = self.combat.decide(frame.bgr.shape, mobs)
        if decision.movement.magnitude > 0.05:
            self.inputs.set_movement(decision.movement)
        else:
            self.inputs.release_all_movement()
        if (
            decision.should_attack
            and not self.config.vision.combat.hold_mouse_attack
            and now - self.last_attack_at >= self.config.vision.combat.click_interval_seconds
        ):
            self.inputs.click()
            self.last_attack_at = now

    def _solve_ball_task(self) -> None:
        frame = self.capture.grab_world()
        detector = self._ball_task_instance()
        result = detector.solve(frame.bgr)
        if not result.success:
            self.logger.warning("Ball task solve failed: %s", result.reason)
            self.popup_frames = 0
            self._transition(BotState.GRIND_COMBAT)
            return
        x0, y0, _, _ = self.config.screen.world_roi
        screen_points = [(x + x0, y + y0) for x, y in result.waypoints]
        self.inputs.drag_path(screen_points, self.config.vision.ball_task.drag_duration_seconds)
        self.popup_frames = 0
        self.inputs.press(self.config.hotkeys.battle)
        self.battle_mode_active = True
        self.inputs.set_attack_held(self.config.vision.combat.hold_mouse_attack)
        self._transition(BotState.GRIND_COMBAT)

    def _forced_death(self) -> None:
        self.logger.info("Forced death cycle started; releasing movement and stopping attacks")
        self.inputs.set_attack_held(False)
        self.inputs.release_all_movement()
        # Intentional death is environment-specific. The low-RAM MVP stops fighting
        # and movement so enemies can kill the player naturally.
        self._transition(BotState.DEATH_WAIT)

    def _death_wait(self) -> None:
        elapsed = time.monotonic() - self.last_state_change
        if elapsed >= self.config.timers.death_wait_seconds:
            self._transition(BotState.RECOVER_AND_RESUME)
