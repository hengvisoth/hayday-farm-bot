from threading import Event, Lock
from time import monotonic

from diagnostics import Diagnostics
from matcher import Matcher
from planner import (
    build_drag_route,
    estimate_camera_translation,
    select_center_match,
    select_nearest_match,
    translate_points,
)

import cv2
import numpy as np
import mss
import keyboard
import pyautogui as pa

pa.PAUSE = 0

# Preparation
FIELD_MATCHING_THRESHOLD = 0.7
HARVEST_MATCHING_THRESHOLD = 0.8
WHEAT_MATCHING_THRESHOLD = 0.3
WHEAT_GROWING_MATCHING_THRESHOLD = 0.7
BOAT_MATCHING_THRESHOLD = 0.7
MARKET_MATCHING_THRESHOLD = 0.6
SOLD_MATCHING_THRESHOLD = 0.6
NEW_OFFER_MATCHING_THRESHOLD = 0.6
PLANT_OFFER_MATCHING_THRESHOLD = 0.9
NEWSPAPER_MATCHING_THRESHOLD = 0.6
INSERT_BUTTON_MATCHING_THRESHOLD = 0.9
SILO_MATCHING_THRESHOLD = 0.9
CLOSE_MATCHING_THRESHOLD = 0.4

SCREEN_DIM = {
    'left': 0,
    'top': 0,
    'width': 1920,
    'height': 1080
}

BOAT_ANCHOR = (1075, 285)
MARKET_ANCHOR = (770, 1150)

plant_img = cv2.imread('templates/plants/wheat.png', cv2.IMREAD_UNCHANGED)
plant_growing_img = cv2.imread('templates/plants/wheat_growing.png', cv2.IMREAD_UNCHANGED)
planting_interface_img = cv2.imread('templates/interface/planting_wheat.png', cv2.IMREAD_UNCHANGED)
field_img = cv2.imread('templates/environment/field.png', cv2.IMREAD_UNCHANGED)
harvesting_interface_img = cv2.imread('templates/interface/harvest_scythe.png', cv2.IMREAD_UNCHANGED)
boat_img = cv2.imread('templates/environment/boat.png', cv2.IMREAD_UNCHANGED)
market_img = cv2.imread('templates/environment/market.png', cv2.IMREAD_UNCHANGED)
sold_img = cv2.imread('templates/interface/sold.png', cv2.IMREAD_UNCHANGED)
new_offer_img = cv2.imread('templates/interface/new_offer.png', cv2.IMREAD_UNCHANGED)
plant_offer_img = cv2.imread('templates/interface/wheat_market.png', cv2.IMREAD_UNCHANGED)
newspaper_img = cv2.imread('templates/interface/newspaper.png', cv2.IMREAD_UNCHANGED)
insert_button_img = cv2.imread('templates/interface/insert_button.png', cv2.IMREAD_UNCHANGED)
silo_img = cv2.imread('templates/interface/silo.png', cv2.IMREAD_UNCHANGED)
close_img = cv2.imread('templates/interface/close.png', cv2.IMREAD_UNCHANGED)


class Bot:
    def __init__(
        self,
        logger,
        set_tracking_img,
        capture=None,
        mouse=None,
        key_state=None,
        diagnostics=None,
        matcher=None,
        clock=None,
        waiter=None,
    ):
        self.logger = logger
        self.set_tracking_img = set_tracking_img
        self.m = matcher or Matcher()
        self.mouse = mouse or pa
        self.key_state = key_state or keyboard.is_pressed
        self.clock = clock or monotonic
        self.stop_event = Event()
        self.waiter = waiter or self.stop_event.wait
        self.capture = capture or self._capture_screen
        sink = getattr(logger, "log", logger if callable(logger) else None)
        self.diagnostics = diagnostics or Diagnostics(sink=sink)
        self._loop_lock = Lock()
        self.scythe_timeout = 4.0
        self.scythe_poll_interval = 0.2
        self.drag_point_duration = 0.1
        self.silo_is_full = False
        self.harvested_plants = False
        self.planted_crops = 0

    def _capture_screen(self):
        with mss.MSS() as sct:
            return np.array(sct.grab(SCREEN_DIM))

    def get_target(self):
        return self.capture()

    def request_stop(self):
        self.stop_event.set()
        self._log("CONTROL", "stop_requested")

    def reset_stop(self):
        self.stop_event.clear()

    def _stop_requested(self):
        if self.stop_event.is_set():
            return True
        try:
            if self.key_state("q"):
                self.stop_event.set()
                return True
        except Exception as error:
            self._log("CONTROL", "key_check_failed", error=repr(error))
        return False

    def _wait(self, seconds):
        if self._stop_requested():
            return True
        stopped = bool(self.waiter(max(0, seconds)))
        return stopped or self._stop_requested()

    def _log(self, state, message="", **fields):
        return self.diagnostics.log(state, message, **fields)

    def get_anchor_positions(self, target):
        anchors = {}
        boat = self.m.match_template(boat_img, target, BOAT_MATCHING_THRESHOLD)
        if boat:
            anchors["boat"] = (boat[0][0], boat[0][1])
        market = self.m.match_template(market_img, target, MARKET_MATCHING_THRESHOLD)
        if market:
            anchors["market"] = (market[0][0], market[0][1])
        return anchors

    def get_camera_pos(self, target=None):
        target = self.get_target() if target is None else target
        boat = self.m.match_template(boat_img, target, BOAT_MATCHING_THRESHOLD)
        if len(boat) > 0:
            ax, ay = BOAT_ANCHOR
            # self.logger.log("Target: BOAT_ANCHOR", ax, ay, boat[0][0], boat[0][1])
            return ax + ax - boat[0][0], ay + ay - boat[0][1]

        market = self.m.match_template(market_img, target, MARKET_MATCHING_THRESHOLD)
        if len(market) > 0:
            ax, ay = MARKET_ANCHOR
            dx, dy = ax - BOAT_ANCHOR[0], ay - BOAT_ANCHOR[1]
            # self.logger.log("Target: MARKET_ANCHOR", ax, ay, market[0][0], market[0][1])
            return ax + ax - market[0][0] - dx, ay + ay - market[0][1] - dy

        return 0, 0

    def check_fields_are_empty(self, target):
        return self.m.match_template_exists(field_img, target, FIELD_MATCHING_THRESHOLD)

    def check_plants_are_growing(self, target):
        return self.m.match_template_exists(plant_growing_img, target, WHEAT_GROWING_MATCHING_THRESHOLD)

    def check_silo_is_full(self, target):
        return len(self.m.match_template(silo_img, target, SILO_MATCHING_THRESHOLD)) > 0

    def drag_operation(self, drag_start, matches, target):
        boundary = self.m.matchs_to_boundary(matches)
        path = self.m.boundary_to_path(boundary)
        self.track_path(path, target)
        if self._stop_requested():
            return False
        self.mouse.moveTo(drag_start[0], drag_start[1])
        if self._stop_requested():
            return False
        self.mouse.mouseDown(button='left')
        try:
            for x, y in path:
                if self._stop_requested():
                    return False
                self.mouse.moveTo(x, y, duration=0.1, _pause=False)
            return True
        finally:
            self.mouse.mouseUp(button='left')

    def combine_paths(self, p1, p2):
        result = []
        for (x, y, w, h) in p1:
            result.append([x, y, w, h])
        for (x, y, w, h) in p2:
            result.append([x, y, w, h])
        return result

    def translate_path(self, path, translation):
        result = []
        tx, ty = translation
        for (x, y, w, h) in path:
            result.append([x + tx, y + ty, w, h])
        return result

    def plant_crops(self, target):
        if self._stop_requested():
            return False
        if self.m.match_template_exists(plant_img, target, WHEAT_MATCHING_THRESHOLD):
            self.logger.log("Found already grown plants. Harvesting them first...")
            return False

        cx1, cy1 = self.get_camera_pos(target)
        empty_fields = self.m.match_template(field_img, target, FIELD_MATCHING_THRESHOLD)
        if len(empty_fields) == 0:
            self.logger.log("Empty fields gone, retrying...")
            return False
        x = empty_fields[0][0]
        y = empty_fields[0][1]
        if not self._click((x, y), "PLANT_OPEN", clicks=2):
            return False
        if self._wait(2.0):
            return False

        target = self.get_target()
        cx2, cy2 = self.get_camera_pos(target)
        translation = (cx1 - cx2, cy1 - cy2)
        if cx1 == 0 or cx2 == 0:
            translation = (0, 0)
            self._log("CAMERA", "plant_stable_fallback", shift=translation)
        path = self.translate_path(empty_fields, translation)
        planting_interface = self.m.match_template(planting_interface_img, target, FIELD_MATCHING_THRESHOLD)
        if len(planting_interface) == 0:
            self.logger.log("Planting interface not found, retrying...")
            return False

        drag_start = (planting_interface[0][0], planting_interface[0][1])
        return self.drag_operation(drag_start, path, target)

    def _match_details(self, template, target, threshold):
        return self.m.match_template_details(template, target, threshold)

    @staticmethod
    def _match_centers(matches):
        return [(match[0], match[1]) for match in matches]

    def _click(self, point, state, **kwargs):
        if self._stop_requested():
            return False
        self._log("INPUT", "click", action_state=state, point=point, **kwargs)
        self.mouse.click(point[0], point[1], **kwargs)
        return True

    def _wait_for_scythe(self):
        deadline = self.clock() + self.scythe_timeout
        last_target = None
        while not self._stop_requested():
            last_target = self.get_target()
            result = self._match_details(
                harvesting_interface_img,
                last_target,
                HARVEST_MATCHING_THRESHOLD,
            )
            self._log(
                "WAIT_FOR_SCYTHE",
                "poll",
                matches=len(result.matches),
                confidence=f"{result.best_confidence:.3f}",
                candidates=self._match_centers(result.matches),
            )
            if result.matches:
                return last_target, result
            remaining = deadline - self.clock()
            if remaining <= 0:
                break
            if self._wait(min(self.scythe_poll_interval, remaining)):
                return None, None
        return last_target, None

    def _drag_harvest_route(self, scythe, route):
        if self._stop_requested():
            return False
        self._log("INPUT", "move_to_scythe", point=scythe)
        self.mouse.moveTo(scythe[0], scythe[1])
        if self._stop_requested():
            return False

        mouse_is_down = False
        try:
            self._log("INPUT", "mouse_down", point=scythe)
            self.mouse.mouseDown(button="left")
            mouse_is_down = True
            for point in route:
                if self._stop_requested():
                    return False
                self.mouse.moveTo(
                    point[0],
                    point[1],
                    duration=self.drag_point_duration,
                    _pause=False,
                )
            return True
        finally:
            if mouse_is_down:
                self.mouse.mouseUp(button="left")
                self._log("INPUT", "mouse_up")

    def _track_harvest_plan(self, target, matches, selected, scythe, path):
        tracked = target.copy()
        self.m.mark_matches(matches, tracked, (255, 0, 0))
        cv2.circle(tracked, selected, 7, (0, 255, 255), 2)
        cv2.circle(tracked, scythe, 7, (255, 255, 0), 2)
        self.m.mark_path([scythe, *path], tracked)
        self.set_tracking_img(tracked)

    def harvest_plants(self, target):
        if self._stop_requested():
            return False
        if self.check_plants_are_growing(target):
            self._log("DETECT_WHEAT", "plants_still_growing")
            return False

        wheat_result = self._match_details(
            plant_img,
            target,
            WHEAT_MATCHING_THRESHOLD,
        )
        selected_match = select_center_match(wheat_result.matches)
        self._log(
            "DETECT_WHEAT",
            "complete",
            matches=len(wheat_result.matches),
            confidence=f"{wheat_result.best_confidence:.3f}",
            candidates=self._match_centers(wheat_result.matches),
            selected=None if selected_match is None else tuple(selected_match[:2]),
        )
        if selected_match is None:
            return False

        selected_crop = (selected_match[0], selected_match[1])
        before_crop_centers = self._match_centers(wheat_result.matches)
        before_anchors = self.get_anchor_positions(target)
        self._log("OPEN_TOOL", "click_wheat", point=selected_crop)
        if not self._click(selected_crop, "OPEN_TOOL"):
            return False

        scythe_target, scythe_result = self._wait_for_scythe()
        if scythe_result is None:
            if not self._stop_requested() and scythe_target is not None:
                screenshot = self.diagnostics.save_failure(
                    "scythe_timeout",
                    scythe_target,
                    matches=wheat_result.matches,
                    selected=selected_crop,
                )
                self._log(
                    "WAIT_FOR_SCYTHE",
                    "timeout",
                    seconds=self.scythe_timeout,
                    screenshot=screenshot,
                )
            return False

        after_wheat = self._match_details(
            plant_img,
            scythe_target,
            WHEAT_MATCHING_THRESHOLD,
        )
        after_crop_centers = self._match_centers(after_wheat.matches)
        after_anchors = self.get_anchor_positions(scythe_target)
        screen_size = (scythe_target.shape[1], scythe_target.shape[0])
        translation = estimate_camera_translation(
            before_anchors,
            after_anchors,
            before_crop_centers,
            after_crop_centers,
            screen_size,
        )
        translated_crops = translate_points(before_crop_centers, translation)
        translated_selected = translate_points([selected_crop], translation)[0]
        self._log(
            "CAMERA",
            "translation",
            method=translation.method,
            shift=(translation.dx, translation.dy),
            confidence=translation.confidence,
            before_anchors=before_anchors,
            after_anchors=after_anchors,
        )

        selected_scythe_match = select_nearest_match(
            scythe_result.matches,
            translated_selected,
        )
        if selected_scythe_match is None or not translated_crops:
            screenshot = self.diagnostics.save_failure(
                "invalid_harvest_plan",
                scythe_target,
                matches=scythe_result.matches,
                selected=translated_selected,
            )
            self._log("PLAN_DRAG", "invalid", screenshot=screenshot)
            return False

        scythe = (selected_scythe_match[0], selected_scythe_match[1])
        route = build_drag_route(translated_crops, scythe, max_segment=25)
        self._log(
            "PLAN_DRAG",
            "complete",
            scythe_matches=len(scythe_result.matches),
            confidence=f"{scythe_result.best_confidence:.3f}",
            candidates=self._match_centers(scythe_result.matches),
            selected=scythe,
            points=len(route),
            start=scythe,
            end=route[-1],
        )
        self._track_harvest_plan(
            scythe_target,
            scythe_result.matches,
            translated_selected,
            scythe,
            route,
        )
        if not self._drag_harvest_route(scythe, route):
            return False

        self.harvested_plants = True
        self._log("DRAG", "complete", points=len(route))
        return True

    def sell_items(self, target):
        if self._stop_requested():
            return False
        self._log("SELL", "started")
        # Open market
        market = self.m.match_template(market_img, target, MARKET_MATCHING_THRESHOLD)
        if len(market) == 0:
            self._log("SELL", "market_not_found")
            return False
        if self._wait(0.2):
            return False
        if not self._click((market[0][0] + 50, market[0][1]), "SELL_OPEN"):
            return False
        if self._wait(1.0):
            return False

        # Collect coins
        target = self.get_target()
        sold = self.m.match_template(sold_img, target, SOLD_MATCHING_THRESHOLD)
        self.track_matches(sold, target)
        if len(sold) > 0:
            self.logger.log("Collecting coins...")
        for s in sold:
            if self._wait(0.2) or not self._click((s[0], s[1]), "SELL_COLLECT"):
                return False
        if self._wait(1.0):
            return False

        # Create offers
        target = self.get_target()
        new_offers = self.m.match_template(new_offer_img, target, NEW_OFFER_MATCHING_THRESHOLD)
        self.track_matches(new_offers, target)
        if len(new_offers) > 0:
            self.logger.log("Inserting new offers...")
        else:
            self.logger.log("No slots for offers found!")
        for offer in new_offers:
            if not self.create_offer(offer):
                break
        if self._wait(1.0):
            return False

        # Exit market
        close = self.m.match_template(close_img, target, CLOSE_MATCHING_THRESHOLD)
        if len(close) > 0:
            self.logger.log("Finished with selling. Closing market...")
            return self._click((close[0][0], close[0][1]), "SELL_CLOSE")
        return True

    def create_offer(self, offer):
        # Open new offer window
        if not self._click((offer[0], offer[1]), "OFFER_OPEN"):
            return False
        if self._wait(0.5):
            return False

        # Select target plant to sell
        target = self.get_target()
        plant_to_sell = self.m.match_template(plant_offer_img, target, PLANT_OFFER_MATCHING_THRESHOLD)
        self.track_matches(plant_to_sell, target)
        if len(plant_to_sell) > 0:
            if not self._click(
                (plant_to_sell[0][0], plant_to_sell[0][1]),
                "OFFER_SELECT_CROP",
            ):
                return False
        else:
            self.logger.log("Target plant not found... Assuming, that plants are empty")
            if not self._stop_requested():
                keyboard.send("esc")
            return False
        if self._wait(0.5):
            return False

        # Check if newspaper insert is available
        newspaper = self.m.match_template(newspaper_img, target, NEWSPAPER_MATCHING_THRESHOLD)
        self.track_matches(newspaper, target)
        if len(newspaper) > 0:
            self.logger.log("Insert to newspaper available. Inserting...")
            if not self._click(
                (newspaper[0][0], newspaper[0][1]),
                "OFFER_NEWSPAPER",
            ):
                return False
        if self._wait(0.2):
            return False

        # Insert offer
        target = self.get_target()
        insert_button = self.m.match_template(insert_button_img, target, INSERT_BUTTON_MATCHING_THRESHOLD)
        self.track_matches(insert_button, target)
        if len(insert_button) > 0:
            if not self._click(
                (insert_button[0][0], insert_button[0][1]),
                "OFFER_INSERT",
            ):
                return False
        else:
            self.logger.log("Insert button not found. This is a critical error, because no plants can be sold.")
            return False
        return not self._wait(0.2)

    def check_unexpected_behaviour(self, target, reason=None):
        if reason is None:
            self._log("RECOVERY", "skipped_without_known_failure")
            return False
        close = self.m.match_template(close_img, target, CLOSE_MATCHING_THRESHOLD)
        if len(close) > 0:
            self._log("RECOVERY", "close_window", reason=reason)
            if self._wait(0.2):
                return False
            return self._click((close[0][0], close[0][1]), "RECOVERY_CLOSE")
        return False

    def track_matches(self, matches, target):
        tracked = target.copy()
        self.m.mark_matches(matches, tracked, (255, 0, 0))
        self.set_tracking_img(tracked)

    def track_path(self, path, target):
        tracked = target.copy()
        self.m.mark_path(path, tracked)
        self.set_tracking_img(tracked)

    def bot_loop(self):
        if not self._loop_lock.acquire(blocking=False):
            self._log("CONTROL", "worker_already_running")
            return False

        try:
            try:
                mouse_size = tuple(self.mouse.size())
            except Exception as error:
                mouse_size = f"unavailable:{error!r}"
            templates_ok = all(
                image is not None
                for image in (
                    plant_img,
                    plant_growing_img,
                    planting_interface_img,
                    field_img,
                    harvesting_interface_img,
                    boat_img,
                    market_img,
                )
            )
            self._log(
                "SESSION",
                "started",
                capture_bounds=SCREEN_DIM,
                mouse_size=mouse_size,
                templates_ok=templates_ok,
                wheat_threshold=WHEAT_MATCHING_THRESHOLD,
                scythe_threshold=HARVEST_MATCHING_THRESHOLD,
                scythe_timeout=self.scythe_timeout,
            )

            while not self._stop_requested():
                screen = self.get_target()
                if self._stop_requested():
                    break

                anchors = self.get_anchor_positions(screen)
                self._log("CAMERA", "visible_anchors", anchors=anchors)

                empty_fields = self.m.match_template(
                    field_img,
                    screen,
                    FIELD_MATCHING_THRESHOLD,
                )
                self.track_matches(empty_fields, screen)
                if empty_fields:
                    self._log("PLANT", "empty_fields", matches=len(empty_fields))
                    self.plant_crops(screen)
                    self.planted_crops += len(empty_fields)
                if self._stop_requested():
                    break

                if self.harvested_plants or self.silo_is_full:
                    self.silo_is_full = False
                    self.sell_items(screen)
                self.harvested_plants = False
                if self._stop_requested():
                    break

                grown_result = self._match_details(
                    plant_img,
                    screen,
                    WHEAT_MATCHING_THRESHOLD,
                )
                self.track_matches(grown_result.matches, screen)
                if not self.silo_is_full:
                    if grown_result.matches:
                        self._log(
                            "HARVEST",
                            "grown_plants",
                            matches=len(grown_result.matches),
                            confidence=f"{grown_result.best_confidence:.3f}",
                        )
                        self.harvest_plants(screen)
                    else:
                        self._log("HARVEST", "waiting_for_growth")
                        if self._wait(10.0):
                            break

                if self._stop_requested():
                    break
                screen = self.get_target()
                if self.check_silo_is_full(screen):
                    self._log("SILO", "full")
                    self.silo_is_full = True

                if self._wait(3.0):
                    break
            self._log("SESSION", "stopped")
            return True
        except Exception as error:
            self._log("SESSION", "worker_error", error=repr(error))
            raise
        finally:
            self._loop_lock.release()
